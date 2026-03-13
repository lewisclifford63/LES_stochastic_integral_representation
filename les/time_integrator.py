"""
Time integration for the random LES method.

Primary entry point
-------------------
``advance_velocity_one_step_qian``
    Implements one full step of Qian's scheme (eqs. 41-44 / 54-56):
        1. Compute  G_n = -grad P_n + F_n  from the current Eulerian field.
        2. Call ``accumulated_history_velocity_update`` which:
           a. Interpolates U and G at particle positions.
           b. Accumulates  G  into each particle's running integral.
           c. Advances particles via Milstein (eq. 55).
           d. Reconstructs U_{n+1} by direct particle sum (eq. 56).

``run_time_loop_qian``
    Wraps the above in a time loop with snapshot saving.

The iterative procedure matches Qian's diagram on page 13:

    U_0, Y_0  -->  Y_1
                   + G_0, U_0  -->  U_1
                                    + dU_1/dx, F  -->  G_1
    U_1, Y_1  -->  Y_2
                   + G_1, U_0  -->  U_2   ...

Legacy functions ``advance_velocity_one_step`` and ``run_time_loop``
are retained unchanged so that existing tests still pass.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from les.grid import Grid2D
from les.pressure import compute_g_field
from les.representation import (
    accumulated_history_velocity_update,
    one_step_velocity_update,
)
from les.les_filtering import apply_spatial_filter


class _QianStepResult(TypedDict):
    U_np1: np.ndarray
    particle_state: dict[str, np.ndarray]
    source_n: np.ndarray
    grad_p_n: np.ndarray
    g_n: np.ndarray


# ================================================================== #
#  Pressure / RHS helper  (unchanged)
# ================================================================== #

def compute_step_rhs(
    grid: Grid2D,
    U_n: np.ndarray,
    F_n: np.ndarray,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the pressure Poisson source, grad_p, and  g = -grad_p + F
    for the current Eulerian velocity field.
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
#  New primary interface  (Qian's full scheme)
# ================================================================== #

def advance_velocity_one_step_qian(
    grid: Grid2D,
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
    Advance the velocity by one stochastic step using the full
    accumulated-history scheme (Qian eqs. 41-44 / 54-56).

    Parameters
    ----------
    grid : Grid2D
    particle_state : dict
        Mutable particle state produced by
        ``les.particles.initialize_particle_history``.  Contains keys
        'positions', 'weights', 'u0_carried', 'accumulated_G'.
        **Updated in-place** by this call.
    U_n : (Ny, Nx, 2)
        Eulerian velocity at time  t_k .
    F_n : (Ny, Nx, 2)
        External forcing at time  t_k .
    dt, nu, filter_width, filter_cutoff_sigma : float
    rng : Generator
    pressure_cutoff_radius : float | None
    pressure_softening : float
    clip_to_box : bool
    use_milstein : bool

    Returns
    -------
    dict with keys:
        'U_np1'           (Ny, Nx, 2)  velocity at  t_{k+1}
        'particle_state'  dict          updated particle state
        'source_n'        (Ny, Nx)      pressure Poisson source
        'grad_p_n'        (Ny, Nx, 2)   pressure gradient
        'g_n'             (Ny, Nx, 2)   G = -grad P + F
    """
    # 1. Filter U_n before computing the pressure gradient.
    #
    #    The filtered NSE (Qian eq. 36/39) operates on the filtered
    #    velocity  u_tilde.  The pressure Poisson source involves
    #    quadratic products of velocity gradients (eq. 44), so
    #    grid-scale stochastic noise in the raw reconstruction would
    #    be amplified quadratically and fed back into G.  Applying the
    #    LES filter here is consistent with the filtered formulation
    #    and suppresses this noise feedback loop.
    #
    #    The raw (unfiltered) U_n is still passed to the particle
    #    update for drift interpolation and Milstein gradients.
    U_n_filtered = apply_spatial_filter(
        grid=grid,
        U=U_n,
        sigma=filter_width,
        cutoff_sigma=filter_cutoff_sigma,
        normalize=True,
    )

    # 2. Compute  G_n = -grad P_n + F_n  from the FILTERED velocity
    source, grad_p, g_n = compute_step_rhs(
        grid=grid,
        U_n=U_n_filtered,
        F_n=F_n,
        pressure_cutoff_radius=pressure_cutoff_radius,
        pressure_softening=pressure_softening,
    )

    # 3. Accumulate history, advance particles, reconstruct U_{n+1}
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
    grid: Grid2D,
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
    Run the full random LES time loop using Qian's accumulated-history
    scheme and return snapshot history.

    Parameters
    ----------
    particle_state : dict
        Initial particle state from ``initialize_particle_history``.
        Mutated across steps.
    U_0 : (Ny, Nx, 2)
        Initial Eulerian velocity.
    forcing_function : callable(grid, t) -> (Ny, Nx, 2)
    Other parameters are the same as ``advance_velocity_one_step_qian``.

    Returns
    -------
    history : dict[str, list]
        'times'      list of float
        'U'          list of (Ny, Nx, 2) velocity snapshots
        'positions'  list of (Np, 2)     particle position snapshots
        'source'     list of (Ny, Nx)
        'grad_p'     list of (Ny, Nx, 2)
        'g'          list of (Ny, Nx, 2)
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
#  Legacy interface  (retained for backward compatibility)
# ================================================================== #

def advance_velocity_one_step(
    grid: Grid2D,
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
    LEGACY — single-step normalised-conditional-average update.

    Uses the old Euler-Maruyama integrator and normalised deposition.
    Retained so that existing tests and the old ``main.py`` still run.
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
    grid: Grid2D,
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
    LEGACY — run a full time loop using the old single-step scheme.
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
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected positions to have shape (Np, 2).")


def _validate_weights(weights: np.ndarray, n_particles: int) -> None:
    if weights.ndim != 1 or weights.shape[0] != n_particles:
        raise ValueError("Expected weights to have shape (Np,).")


def _validate_vector_field(
    grid: Grid2D,
    field: np.ndarray,
    name: str,
) -> None:
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")