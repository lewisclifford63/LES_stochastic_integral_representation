"""
DNS time integration for the padded-box 2D Navier-Stokes reference.

Uses the classical fourth-order Runge-Kutta (RK4) scheme for time
advancement, with an FFT-based pressure projection after each full
step to enforce the divergence-free constraint.

The non-pressure RHS is:

    R(U, F) = -(U · nabla)U  +  nu * Delta U  +  F

One RK4 step computes:

    k1 = R(U_n,         F_n)
    k2 = R(U_n + dt/2 * k1, F_{n+1/2})
    k3 = R(U_n + dt/2 * k2, F_{n+1/2})
    k4 = R(U_n + dt   * k3, F_{n+1})

    U_star = U_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
    U_{n+1} = project(U_star)

For time-independent or slowly varying forcing, the distinction between
F_n, F_{n+1/2}, F_{n+1} is minor.  We evaluate F at the appropriate
sub-step time for correctness.
"""

import numpy as np

from les.grid import Grid2D
from les.differential_operators import convective_term, laplacian_vector
from dns.poisson import project_velocity


# ================================================================== #
#  Non-pressure RHS
# ================================================================== #

def explicit_navier_stokes_rhs(
    U: np.ndarray,
    F: np.ndarray,
    dx: float,
    dy: float,
    nu: float,
) -> np.ndarray:
    """
    Compute the explicit non-pressure RHS:

        R(U, F) = -(U · nabla)U  +  nu * Delta U  +  F
    """
    _validate_vector_field(U, "U")
    _validate_vector_field(F, "F")

    adv = convective_term(U, dx, dy)
    diff = laplacian_vector(U, dx, dy)

    rhs = -adv + nu * diff + F
    return rhs


# ================================================================== #
#  Single-step integrators
# ================================================================== #

def advance_velocity_rk4(
    grid: Grid2D,
    U_n: np.ndarray,
    forcing_function,
    t_n: float,
    dt: float,
    nu: float,
) -> dict[str, np.ndarray]:
    """
    One classical RK4 step + divergence-free projection.

    Parameters
    ----------
    grid : Grid2D
    U_n : (Ny, Nx, 2)
        Velocity at time t_n.
    forcing_function : callable(grid, t) -> (Ny, Nx, 2)
        External forcing evaluated at arbitrary time.
    t_n : float
        Current time.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    dict with keys:
        'U_np1'   projected velocity at t_{n+1}
        'U_star'  pre-projection velocity
        'p_corr'  pressure correction field
    """
    _validate_grid_vector_field(grid, U_n, "U_n")

    dx, dy = grid.dx, grid.dy

    # Stage 1:  k1 = R(U_n, F(t_n))
    F1 = forcing_function(grid, t_n)
    k1 = explicit_navier_stokes_rhs(U_n, F1, dx, dy, nu)

    # Stage 2:  k2 = R(U_n + dt/2 * k1,  F(t_n + dt/2))
    U2 = U_n + 0.5 * dt * k1
    F2 = forcing_function(grid, t_n + 0.5 * dt)
    k2 = explicit_navier_stokes_rhs(U2, F2, dx, dy, nu)

    # Stage 3:  k3 = R(U_n + dt/2 * k2,  F(t_n + dt/2))
    U3 = U_n + 0.5 * dt * k2
    k3 = explicit_navier_stokes_rhs(U3, F2, dx, dy, nu)

    # Stage 4:  k4 = R(U_n + dt * k3,  F(t_n + dt))
    U4 = U_n + dt * k3
    F4 = forcing_function(grid, t_n + dt)
    k4 = explicit_navier_stokes_rhs(U4, F4, dx, dy, nu)

    # Combine
    U_star = U_n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Project onto divergence-free
    U_np1, p_corr = project_velocity(U_star, dx=dx, dy=dy)

    return {
        "U_np1": U_np1,
        "U_star": U_star,
        "p_corr": p_corr,
    }


def advance_velocity_one_step(
    grid: Grid2D,
    U_n: np.ndarray,
    F_n: np.ndarray,
    dt: float,
    nu: float,
) -> dict[str, np.ndarray]:
    """
    LEGACY — one explicit Euler + projection step.

    Retained for backward compatibility with tests.
    """
    _validate_grid_vector_field(grid, U_n, "U_n")
    _validate_grid_vector_field(grid, F_n, "F_n")

    rhs_n = explicit_navier_stokes_rhs(
        U=U_n,
        F=F_n,
        dx=grid.dx,
        dy=grid.dy,
        nu=nu,
    )

    U_star = U_n + dt * rhs_n
    U_np1, p_corr = project_velocity(U_star, dx=grid.dx, dy=grid.dy)

    return {
        "U_np1": U_np1,
        "U_star": U_star,
        "rhs_n": rhs_n,
        "p_corr": p_corr,
    }


# ================================================================== #
#  Time loop
# ================================================================== #

def run_time_loop(
    grid: Grid2D,
    U_0: np.ndarray,
    forcing_function,
    num_steps: int,
    dt: float,
    nu: float,
    t0: float = 0.0,
    save_every: int = 1,
) -> dict[str, list]:
    """
    Run the DNS time loop using RK4 and store snapshots.

    The interface is identical to the old Euler version so that
    ``dns/main.py`` does not need any changes.
    """
    _validate_grid_vector_field(grid, U_0, "U_0")

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    U_n = U_0.copy()
    t_n = float(t0)

    history = {
        "times": [t_n],
        "U": [U_n.copy()],
        "U_star": [],
        "rhs": [],
        "p_corr": [],
    }

    for step in range(1, num_steps + 1):
        result = advance_velocity_rk4(
            grid=grid,
            U_n=U_n,
            forcing_function=forcing_function,
            t_n=t_n,
            dt=dt,
            nu=nu,
        )

        U_n = result["U_np1"]
        t_n = t0 + step * dt

        if step % save_every == 0 or step == num_steps:
            history["times"].append(t_n)
            history["U"].append(U_n.copy())
            history["U_star"].append(result["U_star"].copy())
            # Store empty rhs for interface compatibility
            history["rhs"].append(np.zeros_like(U_n))
            history["p_corr"].append(result["p_corr"].copy())

    return history


# ================================================================== #
#  Validation helpers
# ================================================================== #

def _validate_vector_field(U: np.ndarray, name: str) -> None:
    if U.ndim != 3 or U.shape[-1] != 2:
        raise ValueError(f"Expected {name} with shape (Ny, Nx, 2).")


def _validate_grid_vector_field(
    grid: Grid2D,
    U: np.ndarray,
    name: str,
) -> None:
    if U.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")