import numpy as np

from les.grid import Grid2D
from les.filter import gaussian_kernel_2d, stencil_half_width_xy
from les.interpolation import bilinear_interpolate_vector
from les.particles import advance_particles_euler_maruyama


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

    Returns:
        numerator   shape (Ny, Nx, 2)
        denominator shape (Ny, Nx)
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
    Compute a conditional average field from deposited particles.

    Returns:
        field      shape (Ny, Nx, 2)
        mass_field shape (Ny, Nx)
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
    One stochastic velocity update.

    Returns:
        U_np1
        positions_np1
        mass_u
        E_u_given_arrival
        E_g_given_arrival
    """
    _validate_positions(positions_n)
    _validate_grid_vector_field(grid, U_n, "U_n")
    _validate_grid_vector_field(grid, g_n, "g_n")
    _validate_weights(weights, positions_n.shape[0])

    if weights.shape[0] != positions_n.shape[0]:
        raise ValueError("weights must have shape (Np,).")

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