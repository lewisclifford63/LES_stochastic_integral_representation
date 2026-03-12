import numpy as np

from les.grid import Grid2D


def initialize_particles_from_grid(
    grid: Grid2D,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize particle positions from grid nodes.
    Returns positions and weights.
    """
    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
    else:
        positions = grid.coordinates_as_particles()

    weights = np.full(positions.shape[0], grid.cell_area, dtype=np.float64)
    return positions, weights


def initialize_particles_with_field_values(
    grid: Grid2D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize particles from grid nodes and sample a vector field there.

    Returns positions, carried values, and weights.
    """
    _validate_vector_field(field, "field")

    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
        values = field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 2).copy()

    weights = np.full(positions.shape[0], grid.cell_area, dtype=np.float64)
    return positions, values, weights


def particle_count(positions: np.ndarray) -> int:
    """Return the number of particles."""
    _validate_positions(positions)
    return int(positions.shape[0])


def particles_in_trusted_region(
    positions: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """Boolean mask for particles in the trusted region."""
    _validate_positions(positions)
    return grid.points_in_trusted_region(positions)


def particles_in_padded_box(
    positions: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """Boolean mask for particles in the padded computational box."""
    _validate_positions(positions)
    return grid.points_in_box(positions)


def advance_particles_euler_maruyama(
    positions: np.ndarray,
    drift: np.ndarray,
    dt: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One Euler-Maruyama step:
        X_{n+1} = X_n + drift * dt + sqrt(2 * nu * dt) * N(0, I)
    """
    _validate_positions(positions)
    _validate_vector_field_flat(drift, "drift")

    if drift.shape != positions.shape:
        raise ValueError("drift must have the same shape as positions.")

    if dt < 0.0:
        raise ValueError("dt must be non-negative.")

    if nu < 0.0:
        raise ValueError("nu must be non-negative.")

    noise = np.sqrt(2.0 * nu * dt) * rng.standard_normal(size=positions.shape)
    return positions + dt * drift + noise


def clip_positions_to_box(
    positions: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """
    Clip particle positions into the padded box.
    """
    _validate_positions(positions)

    clipped = positions.copy()
    eps_x = 1.0e-12 * max(1.0, grid.dx)
    eps_y = 1.0e-12 * max(1.0, grid.dy)

    clipped[:, 0] = np.clip(clipped[:, 0], -grid.L_box, grid.L_box - eps_x)
    clipped[:, 1] = np.clip(clipped[:, 1], -grid.L_box, grid.L_box - eps_y)

    return clipped


def count_particles_outside_box(
    positions: np.ndarray,
    grid: Grid2D,
) -> int:
    """Count particles lying outside the padded box."""
    _validate_positions(positions)
    inside = grid.points_in_box(positions)
    return int(np.sum(~inside))


def fraction_particles_outside_box(
    positions: np.ndarray,
    grid: Grid2D,
) -> float:
    """Fraction of particles lying outside the padded box."""
    _validate_positions(positions)

    if positions.shape[0] == 0:
        return 0.0

    return count_particles_outside_box(positions, grid) / positions.shape[0]


def replicate_particles(
    positions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    copies: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicate particles and divide weights evenly across copies.
    """
    _validate_positions(positions)
    _validate_vector_field_flat(values, "values")
    _validate_weights(weights, positions.shape[0])

    if values.shape != positions.shape:
        raise ValueError("values must have the same shape as positions.")

    if copies < 1:
        raise ValueError("copies must be at least 1.")

    if copies == 1:
        return positions.copy(), values.copy(), weights.copy()

    positions_rep = np.repeat(positions, copies, axis=0)
    values_rep = np.repeat(values, copies, axis=0)
    weights_rep = np.repeat(weights / copies, copies, axis=0)

    return positions_rep, values_rep, weights_rep


def _validate_positions(positions: np.ndarray) -> None:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected positions to have shape (Np, 2).")


def _validate_vector_field(field: np.ndarray, name: str) -> None:
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(f"Expected {name} to have shape (Ny, Nx, 2).")


def _validate_vector_field_flat(field: np.ndarray, name: str) -> None:
    if field.ndim != 2 or field.shape[1] != 2:
        raise ValueError(f"Expected {name} to have shape (Np, 2).")


def _validate_weights(weights: np.ndarray, n_particles: int) -> None:
    if weights.ndim != 1 or weights.shape[0] != n_particles:
        raise ValueError("Expected weights to have shape (Np,).")