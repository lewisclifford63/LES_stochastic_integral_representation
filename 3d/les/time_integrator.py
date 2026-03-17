"""
Time integration for the 3D random LES stochastic integral representation method.

Based on Qian et al. (2025) integral representation with accumulated history of pressure
gradient for 3D Lagrangian particle tracking. Oxford Mathematics Masters dissertation.

Primary entry point
-------------------
``advance_velocity_one_step_qian``
    Implements one full step of Qian's scheme (eqs. 41-44 / 54-56) in 3D:
        1. Filter U_n before computing pressure gradient (Qian eq. 36/39)
        2. Compute G_n = -grad P_n + F_n from filtered Eulerian field
        3. Call ``accumulated_history_velocity_update`` which:
           a. Interpolates U and G at particle positions (3D trilinear)
           b. Accumulates G into each particle's running integral
           c. Advances particles via 3D Milstein (eq. 55)
           d. Reconstructs U_{n+1} by 3D Gaussian particle sum (eq. 56)

``run_time_loop_qian``
    Wraps the above in a time loop with snapshot saving.

Dimensional conventions
-----------------------
Scalar fields:   (Nz, Ny, Nx)
Vector fields:   (Nz, Ny, Nx, 3) with components [u, v, w] (x, y, z directions)
Tensor fields:   (Nz, Ny, Nx, 3, 3)
Particle state:  positions (Np, 3), weights (Np,)

Grid axes:
  - axis 0: z-direction (vertical)
  - axis 1: y-direction (lateral)
  - axis 2: x-direction (lateral)
  - vector component indices: 0 = u (x), 1 = v (y), 2 = w (z)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from les.grid import CartesianGrid3D
from les.pressure import compute_g_field
from les.representation import (
    accumulated_history_velocity_update,
    one_step_velocity_update,
)
from les.les_filtering import apply_spatial_filter


class _QianStepResult(TypedDict):
    """Result dictionary from a single Qian step."""
    U_np1: np.ndarray
    particle_state: dict[str, np.ndarray]
    source_n: np.ndarray
    grad_p_n: np.ndarray
    g_n: np.ndarray


# ================================================================== #
#  Pressure / RHS helper
# ================================================================== #

def compute_step_rhs(
    grid: CartesianGrid3D,
    U_n: np.ndarray,
    F_n: np.ndarray,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the pressure Poisson source, grad_p, and g = -grad_p + F
    for the current Eulerian velocity field in 3D.

    Parameters
    ----------
    grid : CartesianGrid3D
        Uniform Cartesian grid on [-L_box, L_box]^3
    U_n : ndarray, shape (Nz, Ny, Nx, 3)
        Eulerian velocity field at time t_n
    F_n : ndarray, shape (Nz, Ny, Nx, 3)
        External forcing
    pressure_cutoff_radius : float, optional
        Cutoff radius for pressure Poisson kernel
    pressure_softening : float, default 1.0e-10
        Softening parameter for pressure regularization

    Returns
    -------
    source : ndarray, shape (Nz, Ny, Nx)
        Pressure Poisson source = -div(u ⊗ u)
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        Pressure gradient
    g_n : ndarray, shape (Nz, Ny, Nx, 3)
        RHS field g = -grad_p + F
    """
    _validate_vector_field(grid, U_n, "U_n")
    _validate_vector_field(grid, F_n, "F_n")

    source, grad_p, g_n = compute_g_field(
        grid=grid,
        U=U_n,
        F=F_n,
        cutoff_radius=pressure_cutoff_radius,
        softening=pressure_softening,
    )

    return source, grad_p, g_n


# ================================================================== #
#  New primary interface (Qian's full scheme in 3D)
# ================================================================== #

def advance_velocity_one_step_qian(
    grid: CartesianGrid3D,
    particle_state: dict[str, np.ndarray],
    U_n: np.ndarray,
    F_n: np.ndarray,
    dt: float,
    nu: float,
    filter_width: float,
    filter_cutoff_sigma: float,
    rng: np.random.Generator,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
    clip_to_box: bool = False,
    use_milstein: bool = True,
) -> _QianStepResult:
    """
    Advance the 3D velocity by one stochastic step using the full
    accumulated-history scheme (Qian eqs. 41-44 / 54-56).

    The algorithm:
      1. Filter U_n via LES filter (suppresses grid-scale noise feedback)
      2. Compute G_n = -grad P_n + F_n from filtered velocity
      3. Accumulate G_n into particle history and advance particles via Milstein
      4. Reconstruct U_{n+1} by 3D Gaussian particle sum

    Parameters
    ----------
    grid : CartesianGrid3D
        Uniform Cartesian grid
    particle_state : dict
        Mutable particle state from ``initialize_particle_history``.
        Keys: 'positions' (Np, 3), 'weights' (Np,), 'u0_carried' (Np, 3),
        'accumulated_G' (Np, 3). Updated in-place by this call.
    U_n : ndarray, shape (Nz, Ny, Nx, 3)
        Eulerian velocity at time t_k
    F_n : ndarray, shape (Nz, Ny, Nx, 3)
        External forcing at time t_k
    dt : float
        Time step
    nu : float
        Kinematic viscosity
    filter_width : float
        LES filter width (kernel standard deviation)
    filter_cutoff_sigma : float
        LES filter cutoff (in units of sigma)
    rng : np.random.Generator
        Random number generator
    pressure_cutoff_radius : float, optional
        Cutoff radius for pressure computation
    pressure_softening : float, default 1.0e-10
        Pressure regularization parameter
    clip_to_box : bool, default False
        If True, clip particle positions to domain box
    use_milstein : bool, default True
        If True, use Milstein scheme; otherwise Euler-Maruyama

    Returns
    -------
    dict with keys:
        'U_np1'           (Nz, Ny, Nx, 3)  velocity at t_{k+1}
        'particle_state'  dict              updated particle state
        'source_n'        (Nz, Ny, Nx)     pressure Poisson source
        'grad_p_n'        (Nz, Ny, Nx, 3)  pressure gradient
        'g_n'             (Nz, Ny, Nx, 3)  G = -grad P + F
    """
    # 1. Filter U_n before computing the pressure gradient.
    #
    #    The filtered NSE (Qian eq. 36/39) operates on the filtered velocity u_tilde.
    #    The pressure Poisson source involves quadratic products of velocity gradients
    #    (eq. 44), so grid-scale stochastic noise in the raw reconstruction would be
    #    amplified quadratically and fed back into G. Applying the LES filter here
    #    is consistent with the filtered formulation and suppresses this feedback.
    #
    #    The raw (unfiltered) U_n is still passed to the particle update for drift
    #    interpolation and Milstein gradients.
    U_n_filtered = apply_spatial_filter(
        grid,
        U_n,
        filter_width,
        filter_cutoff_sigma,
        normalize=True,
    )

    # 2. Compute G_n = -grad P_n + F_n from the FILTERED velocity
    source, grad_p, g_n = compute_step_rhs(
        grid=grid,
        U_n=U_n_filtered,
        F_n=F_n,
        pressure_cutoff_radius=pressure_cutoff_radius,
        pressure_softening=pressure_softening,
    )

    # 3. Accumulate history, advance particles (3D Milstein), reconstruct U_{n+1}
    U_np1, particle_state = accumulated_history_velocity_update(
        grid=grid,
        particle_state=particle_state,
        U_n=U_n,
        g_n=g_n,
        dt=dt,
        nu=nu,
        sigma=filter_width,
        cutoff_sigma=filter_cutoff_sigma,
        rng=rng,
        clip_to_box=clip_to_box,
        use_milstein=use_milstein,
    )

    return {
        "U_np1": U_np1,
        "particle_state": particle_state,
        "source_n": source,
        "grad_p_n": grad_p,
        "g_n": g_n,
    }


def run_time_loop_qian(
    grid: CartesianGrid3D,
    particle_state: dict[str, np.ndarray],
    U_0: np.ndarray,
    forcing_function,
    num_steps: int,
    dt: float,
    nu: float,
    filter_width: float,
    filter_cutoff_sigma: float,
    rng: np.random.Generator,
    t0: float = 0.0,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
    clip_to_box: bool = False,
    use_milstein: bool = True,
    save_every: int = 1,
) -> dict[str, list]:
    """
    Run the full 3D random LES time loop using Qian's accumulated-history
    scheme and return snapshot history.

    The loop iteratively calls ``advance_velocity_one_step_qian`` and saves
    snapshots according to ``save_every``.

    Parameters
    ----------
    grid : CartesianGrid3D
        Uniform Cartesian grid
    particle_state : dict
        Initial particle state from ``initialize_particle_history``.
        Mutated across steps. Keys: 'positions', 'weights', 'u0_carried', 'accumulated_G'.
    U_0 : ndarray, shape (Nz, Ny, Nx, 3)
        Initial Eulerian velocity
    forcing_function : callable
        Signature: forcing_function(grid, t) -> (Nz, Ny, Nx, 3)
        Returns forcing field at time t
    num_steps : int
        Number of time steps to integrate
    dt : float
        Time step
    nu : float
        Kinematic viscosity
    filter_width : float
        LES filter width
    filter_cutoff_sigma : float
        LES filter cutoff
    rng : np.random.Generator
        Random number generator
    t0 : float, default 0.0
        Initial time
    pressure_cutoff_radius : float, optional
        Pressure cutoff radius
    pressure_softening : float, default 1.0e-10
        Pressure softening parameter
    clip_to_box : bool, default False
        Clip particles to domain
    use_milstein : bool, default True
        Use Milstein scheme
    save_every : int, default 1
        Save a snapshot every this many steps

    Returns
    -------
    history : dict[str, list]
        'times'      list of float
        'U'          list of (Nz, Ny, Nx, 3) velocity snapshots
        'positions'  list of (Np, 3) particle position snapshots
        'source'     list of (Nz, Ny, Nx) pressure source snapshots
        'grad_p'     list of (Nz, Ny, Nx, 3) pressure gradient snapshots
        'g'          list of (Nz, Ny, Nx, 3) RHS snapshots
    """
    _validate_vector_field(grid, U_0, "U_0")

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    U_n = U_0.copy()
    t_n = float(t0)

    history = {
        "times": [t_n],
        "U": [U_n.copy()],
        "positions": [particle_state["positions"].copy()],
        "source": [],
        "grad_p": [],
        "g": [],
    }

    for step in range(1, num_steps + 1):
        F_n = forcing_function(grid, t_n)

        result = advance_velocity_one_step_qian(
            grid=grid,
            particle_state=particle_state,
            U_n=U_n,
            F_n=F_n,
            dt=dt,
            nu=nu,
            filter_width=filter_width,
            filter_cutoff_sigma=filter_cutoff_sigma,
            rng=rng,
            pressure_cutoff_radius=pressure_cutoff_radius,
            pressure_softening=pressure_softening,
            clip_to_box=clip_to_box,
            use_milstein=use_milstein,
        )

        U_n = result["U_np1"]
        particle_state = result["particle_state"]
        t_n = t0 + step * dt

        if step % save_every == 0 or step == num_steps:
            history["times"].append(t_n)
            history["U"].append(U_n.copy())
            history["positions"].append(particle_state["positions"].copy())
            history["source"].append(result["source_n"].copy())
            history["grad_p"].append(result["grad_p_n"].copy())
            history["g"].append(result["g_n"].copy())

    return history


# ================================================================== #
#  Legacy interface (retained for backward compatibility)
# ================================================================== #

def advance_velocity_one_step(
    grid: CartesianGrid3D,
    positions_n: np.ndarray,
    weights: np.ndarray,
    U_n: np.ndarray,
    F_n: np.ndarray,
    dt: float,
    nu: float,
    filter_width: float,
    filter_cutoff_sigma: float,
    rng: np.random.Generator,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
    apply_les_filter: bool = False,
    clip_to_box: bool = False,
    mass_tol: float = 1.0e-14,
) -> dict[str, np.ndarray]:
    """
    LEGACY — single-step normalised-conditional-average update (3D).

    Uses the old Euler-Maruyama integrator and normalised deposition.
    Retained so that existing tests and old code paths still run.

    Parameters
    ----------
    grid : CartesianGrid3D
        Uniform Cartesian grid
    positions_n : ndarray, shape (Np, 3)
        Particle positions at time t_n
    weights : ndarray, shape (Np,)
        Particle weights
    U_n : ndarray, shape (Nz, Ny, Nx, 3)
        Eulerian velocity
    F_n : ndarray, shape (Nz, Ny, Nx, 3)
        Forcing
    dt, nu, filter_width, filter_cutoff_sigma : float
    rng : np.random.Generator
    pressure_cutoff_radius : float, optional
    pressure_softening : float
    apply_les_filter : bool
        If True, filter U_{n+1} after reconstruction
    clip_to_box : bool
    mass_tol : float

    Returns
    -------
    dict with keys:
        'U_np1'           (Nz, Ny, Nx, 3)
        'U_np1_raw'       (Nz, Ny, Nx, 3)
        'positions_np1'   (Np, 3)
        'weights_np1'     (Np,)
        'source_n'        (Nz, Ny, Nx)
        'grad_p_n'        (Nz, Ny, Nx, 3)
        'g_n'             (Nz, Ny, Nx, 3)
        'mass_field'      (Nz, Ny, Nx)
        'E_u_given_arrival'  (Nz, Ny, Nx, 3)
        'E_g_given_arrival'  (Nz, Ny, Nx, 3)
    """
    _validate_positions(positions_n)
    _validate_weights(weights, positions_n.shape[0])
    _validate_vector_field(grid, U_n, "U_n")
    _validate_vector_field(grid, F_n, "F_n")

    source, grad_p, g_n = compute_step_rhs(
        grid=grid,
        U_n=U_n,
        F_n=F_n,
        pressure_cutoff_radius=pressure_cutoff_radius,
        pressure_softening=pressure_softening,
    )

    U_np1_raw, positions_np1, mass_field, E_u, E_g = one_step_velocity_update(
        grid=grid,
        positions_n=positions_n,
        U_n=U_n,
        g_n=g_n,
        weights=weights,
        dt=dt,
        nu=nu,
        sigma=filter_width,
        cutoff_sigma=filter_cutoff_sigma,
        rng=rng,
        clip_to_box=clip_to_box,
        mass_tol=mass_tol,
    )

    if apply_les_filter:
        U_np1 = apply_spatial_filter(
            grid=grid,
            U=U_np1_raw,
            sigma=filter_width,
            cutoff_sigma=filter_cutoff_sigma,
            normalize=True,
        )
    else:
        U_np1 = U_np1_raw

    return {
        "U_np1": U_np1,
        "U_np1_raw": U_np1_raw,
        "positions_np1": positions_np1,
        "weights_np1": weights.copy(),
        "source_n": source,
        "grad_p_n": grad_p,
        "g_n": g_n,
        "mass_field": mass_field,
        "E_u_given_arrival": E_u,
        "E_g_given_arrival": E_g,
    }


def run_time_loop(
    grid: CartesianGrid3D,
    positions_0: np.ndarray,
    weights_0: np.ndarray,
    U_0: np.ndarray,
    forcing_function,
    num_steps: int,
    dt: float,
    nu: float,
    filter_width: float,
    filter_cutoff_sigma: float,
    rng: np.random.Generator,
    t0: float = 0.0,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
    apply_les_filter: bool = False,
    clip_to_box: bool = False,
    mass_tol: float = 1.0e-14,
    save_every: int = 1,
) -> dict[str, list]:
    """
    LEGACY — run a full 3D time loop using the old single-step scheme.

    Parameters
    ----------
    Similar to ``run_time_loop_qian``, but uses ``advance_velocity_one_step``.

    Returns
    -------
    history : dict[str, list]
        'times'      list of float
        'U'          list of (Nz, Ny, Nx, 3)
        'positions'  list of (Np, 3)
        'weights'    list of (Np,)
        'source'     list of (Nz, Ny, Nx)
        'grad_p'     list of (Nz, Ny, Nx, 3)
        'g'          list of (Nz, Ny, Nx, 3)
        'mass_field' list of (Nz, Ny, Nx)
    """
    _validate_positions(positions_0)
    _validate_weights(weights_0, positions_0.shape[0])
    _validate_vector_field(grid, U_0, "U_0")

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    positions_n = positions_0.copy()
    weights_n = weights_0.copy()
    U_n = U_0.copy()
    t_n = float(t0)

    history = {
        "times": [t_n],
        "U": [U_n.copy()],
        "positions": [positions_n.copy()],
        "weights": [weights_n.copy()],
        "source": [],
        "grad_p": [],
        "g": [],
        "mass_field": [],
    }

    for step in range(1, num_steps + 1):
        F_n = forcing_function(grid, t_n)

        result = advance_velocity_one_step(
            grid=grid,
            positions_n=positions_n,
            weights=weights_n,
            U_n=U_n,
            F_n=F_n,
            dt=dt,
            nu=nu,
            filter_width=filter_width,
            filter_cutoff_sigma=filter_cutoff_sigma,
            rng=rng,
            pressure_cutoff_radius=pressure_cutoff_radius,
            pressure_softening=pressure_softening,
            apply_les_filter=apply_les_filter,
            clip_to_box=clip_to_box,
            mass_tol=mass_tol,
        )

        U_n = result["U_np1"]
        positions_n = result["positions_np1"]
        weights_n = result["weights_np1"]
        t_n = t0 + step * dt

        if step % save_every == 0 or step == num_steps:
            history["times"].append(t_n)
            history["U"].append(U_n.copy())
            history["positions"].append(positions_n.copy())
            history["weights"].append(weights_n.copy())
            history["source"].append(result["source_n"].copy())
            history["grad_p"].append(result["grad_p_n"].copy())
            history["g"].append(result["g_n"].copy())
            history["mass_field"].append(result["mass_field"].copy())

    return history


# ================================================================== #
#  Validation helpers
# ================================================================== #

def _validate_positions(positions: np.ndarray) -> None:
    """Validate that positions have shape (Np, 3)."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected positions to have shape (Np, 3).")


def _validate_weights(weights: np.ndarray, n_particles: int) -> None:
    """Validate that weights have shape (Np,)."""
    if weights.ndim != 1 or weights.shape[0] != n_particles:
        raise ValueError("Expected weights to have shape (Np,).")


def _validate_vector_field(
    grid: CartesianGrid3D,
    field: np.ndarray,
    name: str,
) -> None:
    """Validate that vector field has shape (Nz, Ny, Nx, 3)."""
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")
