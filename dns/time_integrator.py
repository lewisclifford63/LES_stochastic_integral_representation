import numpy as np

from les.grid import Grid2D
from les.differential_operators import convective_term, laplacian_vector
from dns.poisson import project_velocity


def explicit_navier_stokes_rhs(
    U: np.ndarray,
    F: np.ndarray,
    dx: float,
    dy: float,
    nu: float,
) -> np.ndarray:
    """
    Compute the explicit non-pressure RHS:
        - (U dot grad) U + nu * Delta U + F
    """
    _validate_vector_field(U, "U")
    _validate_vector_field(F, "F")

    adv = convective_term(U, dx, dy)
    diff = laplacian_vector(U, dx, dy)

    rhs = -adv + nu * diff + F
    return rhs


def advance_velocity_one_step(
    grid: Grid2D,
    U_n: np.ndarray,
    F_n: np.ndarray,
    dt: float,
    nu: float,
) -> dict[str, np.ndarray]:
    """
    One explicit Euler + projection step.

    Step:
        U_star = U_n + dt * RHS(U_n)
        U_np1  = projection(U_star)
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
    Run the DNS time loop and store snapshots.
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
        F_n = forcing_function(grid, t_n)

        result = advance_velocity_one_step(
            grid=grid,
            U_n=U_n,
            F_n=F_n,
            dt=dt,
            nu=nu,
        )

        U_n = result["U_np1"]
        t_n = t0 + step * dt

        if step % save_every == 0 or step == num_steps:
            history["times"].append(t_n)
            history["U"].append(U_n.copy())
            history["U_star"].append(result["U_star"].copy())
            history["rhs"].append(result["rhs_n"].copy())
            history["p_corr"].append(result["p_corr"].copy())

    return history


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