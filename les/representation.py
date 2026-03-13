"""
Velocity representation via the random LES integral formula.

Implements the discretised form of the filtered velocity (Qian eq. 56
for R^d, no walls):

    U(x, t_k) = sum_eta  s^d  chi(x - Y^eta_{t_k})  u_0(eta)
              + sum_eta  s^d  chi(x - Y^eta_{t_k})  accum_G^eta

where  accum_G^eta = sum_{j=1}^{k}  dt * G(Y^eta_{t_{j-1}}, t_{j-1})
is the per-particle accumulated forcing integral.

Key differences from the previous implementation
-------------------------------------------------
1. **Direct particle sum** — the deposition is *not* divided by a
   denominator (no conditional average).  The integral representation
   is a direct weighted sum  sum_p  w_p chi(x - Y_p) v_p .

2. **Full accumulated history** — each particle carries its frozen
   initial velocity u_0(eta) and the running integral of G along its
   trajectory.  The velocity at a grid point is reconstructed from
   *all* history, not just the previous time-step's Eulerian field.

3. **Milstein SDE step** — particle positions are advanced with the
   Milstein scheme (Qian eq. 55) via the function
   ``advance_particles_milstein`` in ``les.particles``.
"""

import numpy as np

from les.grid import Grid2D
from les.filter import gaussian_kernel_2d, stencil_half_width_xy
from les.interpolation import (
    bilinear_interpolate_vector,
    bilinear_interpolate_tensor,
)
from les.differential_operators import grad_velocity
from les.particles import (
    advance_particles_milstein,
    advance_particles_euler_maruyama,
    clip_positions_to_box,
)


# ------------------------------------------------------------------ #
#  Core deposition: direct weighted sum  (NO denominator division)
# ------------------------------------------------------------------ #

def deposit_particle_sum(
    grid: Grid2D,
    positions: np.ndarray,
    carried_values: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
) -> np.ndarray:
    """
    Deposit particle-carried vector values onto the grid by direct
    weighted summation (Qian eq. 56).

        field(x) = sum_p  w_p * chi(x - Y_p) * v_p

    There is deliberately **no denominator normalisation**.

    Parameters
    ----------
    grid : Grid2D
    positions : (Np, 2)
        Current particle positions  Y^eta_{t_k}.
    carried_values : (Np, 2)
        Value carried by each particle (either u_0 or accumulated_G).
    weights : (Np,)
        Quadrature weight  s^d  per particle.
    sigma : float
        Width of the Gaussian filter  chi.
    cutoff_sigma : float
        Truncation radius in units of sigma.

    Returns
    -------
    field : (Ny, Nx, 2)
        Reconstructed vector field on the grid.
    """
    _validate_positions(positions)
    _validate_vector_values(carried_values, "carried_values")
    _validate_weights(weights, positions.shape[0])

    if carried_values.shape != positions.shape:
        raise ValueError("carried_values must have shape (Np, 2).")

    field = grid.zeros_vector()

    hx, hy = stencil_half_width_xy(
        dx=grid.dx,
        dy=grid.dy,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    x0 = grid.x[0]
    y0 = grid.y[0]
    radius2 = (cutoff_sigma * sigma) ** 2

    for p in range(positions.shape[0]):
        xp = positions[p, 0]
        yp = positions[p, 1]

        if not (np.isfinite(xp) and np.isfinite(yp)):
            continue

        vp0 = carried_values[p, 0]
        vp1 = carried_values[p, 1]
        wp = weights[p]

        i_center = int(np.round((xp - x0) / grid.dx))
        j_center = int(np.round((yp - y0) / grid.dy))

        i_min = max(0, i_center - hx)
        i_max = min(grid.Nx - 1, i_center + hx)
        j_min = max(0, j_center - hy)
        j_max = min(grid.Ny - 1, j_center + hy)

        xs = grid.x[i_min:i_max + 1]
        ys = grid.y[j_min:j_max + 1]

        Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
        DX = Xs - xp
        DY = Ys - yp

        K = gaussian_kernel_2d(DX, DY, sigma=sigma)
        mask = DX * DX + DY * DY <= radius2
        K = np.where(mask, K, 0.0)

        field[j_min:j_max + 1, i_min:i_max + 1, 0] += wp * K * vp0
        field[j_min:j_max + 1, i_min:i_max + 1, 1] += wp * K * vp1

    return field


# ------------------------------------------------------------------ #
#  Legacy deposition (retained for testing / comparison)
# ------------------------------------------------------------------ #

def deposit_conditional_expectation(
    grid: Grid2D,
    arrival_positions: np.ndarray,
    carried_values: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deposit particle-carried vector values onto the grid.

    Returns both numerator and denominator for backward compatibility
    with diagnostics.

    Returns
    -------
    numerator   : (Ny, Nx, 2)
    denominator : (Ny, Nx)
    """
    _validate_positions(arrival_positions)
    _validate_vector_values(carried_values, "carried_values")
    _validate_weights(weights, arrival_positions.shape[0])

    if carried_values.shape != arrival_positions.shape:
        raise ValueError("carried_values must have shape (Np, 2).")

    numerator = grid.zeros_vector()
    denominator = grid.zeros_scalar()

    hx, hy = stencil_half_width_xy(
        dx=grid.dx,
        dy=grid.dy,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    x0 = grid.x[0]
    y0 = grid.y[0]
    radius2 = (cutoff_sigma * sigma) ** 2

    for p in range(arrival_positions.shape[0]):
        xp = arrival_positions[p, 0]
        yp = arrival_positions[p, 1]
        vp0 = carried_values[p, 0]
        vp1 = carried_values[p, 1]
        wp = weights[p]

        i_center = int(np.round((xp - x0) / grid.dx))
        j_center = int(np.round((yp - y0) / grid.dy))

        i_min = max(0, i_center - hx)
        i_max = min(grid.Nx - 1, i_center + hx)
        j_min = max(0, j_center - hy)
        j_max = min(grid.Ny - 1, j_center + hy)

        xs = grid.x[i_min:i_max + 1]
        ys = grid.y[j_min:j_max + 1]

        Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
        DX = Xs - xp
        DY = Ys - yp

        K = gaussian_kernel_2d(DX, DY, sigma=sigma)
        mask = DX * DX + DY * DY <= radius2
        K = np.where(mask, K, 0.0)

        numerator[j_min:j_max + 1, i_min:i_max + 1, 0] += wp * K * vp0
        numerator[j_min:j_max + 1, i_min:i_max + 1, 1] += wp * K * vp1
        denominator[j_min:j_max + 1, i_min:i_max + 1] += wp * K

    return numerator, denominator


def conditional_average_from_particles(
    grid: Grid2D,
    arrival_positions: np.ndarray,
    carried_values: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
    mass_tol: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalised conditional average (LEGACY — retained for diagnostics).

    Returns
    -------
    field      : (Ny, Nx, 2)
    mass_field : (Ny, Nx)
    """
    numerator, denominator = deposit_conditional_expectation(
        grid=grid,
        arrival_positions=arrival_positions,
        carried_values=carried_values,
        weights=weights,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    field = grid.zeros_vector()
    mask = denominator > mass_tol

    field[..., 0][mask] = numerator[..., 0][mask] / denominator[mask]
    field[..., 1][mask] = numerator[..., 1][mask] / denominator[mask]

    return field, denominator


# ------------------------------------------------------------------ #
#  Full accumulated-history velocity update  (Qian eq. 42 / 56)
# ------------------------------------------------------------------ #

def accumulated_history_velocity_update(
    grid: Grid2D,
    particle_state: dict[str, np.ndarray],
    U_n: np.ndarray,
    g_n: np.ndarray,
    dt: float,
    nu: float,
    sigma: float,
    cutoff_sigma: float,
    rng: np.random.Generator,
    clip_to_box: bool = False,
    use_milstein: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    One full time-step of the random LES for flows in R^d.

    Implements Qian's equations (41)-(44) / (54)-(56) faithfully:

    1.  Interpolate  U(Y_p, t_k)  and  G(Y_p, t_k)  at each particle.
    2.  Accumulate   accum_G_p  +=  dt * G(Y_p, t_k)
    3.  Advance particle positions via the Milstein scheme (eq. 55).
    4.  Reconstruct the Eulerian velocity by direct particle sum (eq. 56):

            U(x, t_{k+1}) = sum_p  w_p chi(x - Y_p^{new})  u0_p
                           + sum_p  w_p chi(x - Y_p^{new})  accum_G_p

    Parameters
    ----------
    grid : Grid2D
    particle_state : dict
        Must contain keys  'positions', 'weights', 'u0_carried',
        'accumulated_G'  as produced by ``initialize_particle_history``.
        **This dict is mutated in-place** (positions and accumulated_G
        are updated).
    U_n : (Ny, Nx, 2)
        Current Eulerian velocity field  U(x, t_k).
    g_n : (Ny, Nx, 2)
        Current  G = -grad P + F  on the grid at time  t_k.
    dt : float
    nu : float
    sigma : float
        Gaussian filter width.
    cutoff_sigma : float
        Truncation radius in multiples of sigma.
    rng : Generator
    clip_to_box : bool
    use_milstein : bool
        If True (default) use Milstein; if False fall back to
        Euler-Maruyama (for testing / comparison).

    Returns
    -------
    U_np1 : (Ny, Nx, 2)
        Velocity field at  t_{k+1}.
    particle_state : dict
        The *same* dict, with updated  'positions' and 'accumulated_G'.
    """
    positions = particle_state["positions"]
    weights = particle_state["weights"]
    u0_carried = particle_state["u0_carried"]
    accumulated_G = particle_state["accumulated_G"]

    _validate_positions(positions)
    _validate_grid_vector_field(grid, U_n, "U_n")
    _validate_grid_vector_field(grid, g_n, "g_n")

    # -------------------------------------------------------------- #
    # 1.  Interpolate U and G at each particle position
    # -------------------------------------------------------------- #
    u_at_particles = bilinear_interpolate_vector(grid, U_n, positions)
    g_at_particles = bilinear_interpolate_vector(grid, g_n, positions)

    # -------------------------------------------------------------- #
    # 2.  Accumulate the G integral  (Qian: sum_{j=1}^{k} dt * G)
    #     We accumulate *before* moving the particles because eq. (56)
    #     evaluates  G(Y^eta_{t_{j-1}}, t_{j-1})  — i.e. at the OLD
    #     position before the step from  t_{j-1}  to  t_j.
    # -------------------------------------------------------------- #
    accumulated_G += dt * g_at_particles

    # -------------------------------------------------------------- #
    # 3.  Advance particles  Y_{t_k} -> Y_{t_{k+1}}
    # -------------------------------------------------------------- #
    if use_milstein:
        gradU = grad_velocity(U_n, grid.dx, grid.dy)   # (Ny, Nx, 2, 2)
        gradU_at_particles = bilinear_interpolate_tensor(
            grid, gradU, positions,
        )                                                # (Np, 2, 2)

        new_positions = advance_particles_milstein(
            positions=positions,
            drift=u_at_particles,
            drift_jacobian=gradU_at_particles,
            dt=dt,
            nu=nu,
            rng=rng,
        )
    else:
        new_positions = advance_particles_euler_maruyama(
            positions=positions,
            drift=u_at_particles,
            dt=dt,
            nu=nu,
            rng=rng,
        )

    if clip_to_box:
        new_positions = clip_positions_to_box(new_positions, grid)

    # -------------------------------------------------------------- #
    # 4.  Reconstruct velocity  (Qian eq. 56)
    #     U(x, t_{k+1}) = initial_term(x)  +  forcing_term(x)
    # -------------------------------------------------------------- #
    initial_term = deposit_particle_sum(
        grid=grid,
        positions=new_positions,
        carried_values=u0_carried,
        weights=weights,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    forcing_term = deposit_particle_sum(
        grid=grid,
        positions=new_positions,
        carried_values=accumulated_G,
        weights=weights,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    U_np1 = initial_term + forcing_term

    # -------------------------------------------------------------- #
    # 5.  Update particle state in-place
    # -------------------------------------------------------------- #
    particle_state["positions"] = new_positions
    # accumulated_G was already mutated above (+=)
    # u0_carried and weights are unchanged

    return U_np1, particle_state


# ------------------------------------------------------------------ #
#  Legacy single-step update  (retained for comparison / testing)
# ------------------------------------------------------------------ #

def one_step_velocity_update(
    grid: Grid2D,
    positions_n: np.ndarray,
    U_n: np.ndarray,
    g_n: np.ndarray,
    weights: np.ndarray,
    dt: float,
    nu: float,
    sigma: float,
    cutoff_sigma: float,
    rng: np.random.Generator,
    clip_to_box: bool = False,
    mass_tol: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    LEGACY — single-step normalised-conditional-average update.

    Retained so that existing tests and the old ``main.py`` still run.
    New code should use ``accumulated_history_velocity_update`` instead.

    Returns
    -------
    U_np1, positions_np1, mass_field, E_u_given_arrival, E_g_given_arrival
    """
    _validate_positions(positions_n)
    _validate_grid_vector_field(grid, U_n, "U_n")
    _validate_grid_vector_field(grid, g_n, "g_n")
    _validate_weights(weights, positions_n.shape[0])

    u_at_particles = bilinear_interpolate_vector(grid, U_n, positions_n)
    g_at_particles = bilinear_interpolate_vector(grid, g_n, positions_n)

    positions_np1 = advance_particles_euler_maruyama(
        positions=positions_n,
        drift=u_at_particles,
        dt=dt,
        nu=nu,
        rng=rng,
    )

    if clip_to_box:
        positions_np1 = _clip_positions_to_box(positions_np1, grid)

    E_u_given_arrival, mass_u = conditional_average_from_particles(
        grid=grid,
        arrival_positions=positions_np1,
        carried_values=u_at_particles,
        weights=weights,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
        mass_tol=mass_tol,
    )

    E_g_given_arrival, mass_g = conditional_average_from_particles(
        grid=grid,
        arrival_positions=positions_np1,
        carried_values=g_at_particles,
        weights=weights,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
        mass_tol=mass_tol,
    )

    U_np1 = E_u_given_arrival + dt * E_g_given_arrival

    mass_field = 0.5 * (mass_u + mass_g)
    low_mass_mask = mass_field <= mass_tol
    if np.any(low_mass_mask):
        U_np1[low_mass_mask, :] = 0.0

    return U_np1, positions_np1, mass_field, E_u_given_arrival, E_g_given_arrival


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _clip_positions_to_box(positions: np.ndarray, grid: Grid2D) -> np.ndarray:
    clipped = positions.copy()

    eps_x = 1.0e-12 * max(1.0, grid.dx)
    eps_y = 1.0e-12 * max(1.0, grid.dy)

    clipped[:, 0] = np.clip(clipped[:, 0], -grid.L_box, grid.L_box - eps_x)
    clipped[:, 1] = np.clip(clipped[:, 1], -grid.L_box, grid.L_box - eps_y)

    return clipped


def _validate_positions(positions: np.ndarray) -> None:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected positions to have shape (Np, 2).")


def _validate_vector_values(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected {name} to have shape (Np, 2).")


def _validate_weights(weights: np.ndarray, n_particles: int) -> None:
    if weights.ndim != 1 or weights.shape[0] != n_particles:
        raise ValueError("Expected weights to have shape (Np,).")


def _validate_grid_vector_field(
    grid: Grid2D,
    field: np.ndarray,
    name: str,
) -> None:
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")