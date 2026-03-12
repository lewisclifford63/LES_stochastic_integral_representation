import numpy as np

from les.grid import Grid2D
from les.pressure import compute_g_field
from les.representation import one_step_velocity_update
from les.les_filtering import apply_spatial_filter


def compute_step_rhs(
    grid: Grid2D,
    U_n: np.ndarray,
    F_n: np.ndarray,
    pressure_cutoff_radius: float | None = None,
    pressure_softening: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute source, grad_p, and g for the current step.
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
    Advance the velocity by one stochastic step.
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
    Run a full time loop and store snapshots.
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