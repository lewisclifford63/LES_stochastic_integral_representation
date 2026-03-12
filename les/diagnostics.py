import numpy as np

from les.grid import Grid2D
from les.differential_operators import divergence


def kinetic_energy(U: np.ndarray, grid: Grid2D) -> float:
    """
    Total kinetic energy on the full padded box.
    """
    _validate_vector_field(grid, U, "U")

    speed2 = U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1]
    return 0.5 * float(np.sum(speed2) * grid.cell_area)


def trusted_kinetic_energy(U: np.ndarray, grid: Grid2D) -> float:
    """
    Kinetic energy restricted to the trusted interior.
    """
    _validate_vector_field(grid, U, "U")

    speed2 = U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1]
    speed2_trusted = grid.restrict_to_trusted(speed2)
    return 0.5 * float(np.sum(speed2_trusted) * grid.cell_area)


def max_speed(U: np.ndarray) -> float:
    """
    Maximum speed on the full grid.
    """
    _validate_vector_field_shape(U)

    speed = np.sqrt(U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1])
    return float(np.max(speed))


def mean_speed(U: np.ndarray) -> float:
    """
    Mean speed on the full grid.
    """
    _validate_vector_field_shape(U)

    speed = np.sqrt(U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1])
    return float(np.mean(speed))


def trusted_speed_stats(U: np.ndarray, grid: Grid2D) -> tuple[float, float]:
    """
    Max and mean speed on the trusted interior.
    """
    _validate_vector_field(grid, U, "U")

    speed = np.sqrt(U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1])
    speed_trusted = grid.restrict_to_trusted(speed)

    max_val = float(np.max(speed_trusted))
    mean_val = float(np.mean(speed_trusted))
    return max_val, mean_val


def divergence_stats(U: np.ndarray, grid: Grid2D) -> tuple[float, float]:
    """
    Max and mean absolute divergence on the full padded box.
    """
    _validate_vector_field(grid, U, "U")

    divU = divergence(U, grid.dx, grid.dy)
    max_val = float(np.max(np.abs(divU)))
    mean_val = float(np.mean(np.abs(divU)))
    return max_val, mean_val


def trusted_divergence_stats(U: np.ndarray, grid: Grid2D) -> tuple[float, float]:
    """
    Max and mean absolute divergence on the trusted interior.
    """
    _validate_vector_field(grid, U, "U")

    divU = divergence(U, grid.dx, grid.dy)
    div_trusted = grid.restrict_to_trusted(divU)

    max_val = float(np.max(np.abs(div_trusted)))
    mean_val = float(np.mean(np.abs(div_trusted)))
    return max_val, mean_val


def relative_l2_change(
    U_old: np.ndarray,
    U_new: np.ndarray,
) -> float:
    """
    Relative L2 change between two velocity fields.
    """
    _validate_vector_field_shape(U_old)
    _validate_vector_field_shape(U_new)

    if U_old.shape != U_new.shape:
        raise ValueError("Velocity fields must have the same shape.")

    num = np.sqrt(np.sum((U_new - U_old) ** 2))
    den = np.sqrt(np.sum(U_old ** 2))

    if den == 0.0:
        return 0.0

    return float(num / den)


def trusted_relative_l2_change(
    U_old: np.ndarray,
    U_new: np.ndarray,
    grid: Grid2D,
) -> float:
    """
    Relative L2 change restricted to the trusted interior.
    """
    _validate_vector_field(grid, U_old, "U_old")
    _validate_vector_field(grid, U_new, "U_new")

    U_old_t = grid.restrict_to_trusted(U_old)
    U_new_t = grid.restrict_to_trusted(U_new)

    num = np.sqrt(np.sum((U_new_t - U_old_t) ** 2))
    den = np.sqrt(np.sum(U_old_t ** 2))

    if den == 0.0:
        return 0.0

    return float(num / den)


def particle_box_escape_fraction(
    positions: np.ndarray,
    grid: Grid2D,
) -> float:
    """
    Fraction of particles lying outside the padded box.
    """
    _validate_positions(positions)

    if positions.shape[0] == 0:
        return 0.0

    inside = grid.points_in_box(positions)
    return float(np.sum(~inside) / positions.shape[0])


def particle_trusted_fraction(
    positions: np.ndarray,
    grid: Grid2D,
) -> float:
    """
    Fraction of particles lying in the trusted interior.
    """
    _validate_positions(positions)

    if positions.shape[0] == 0:
        return 0.0

    inside = grid.points_in_trusted_region(positions)
    return float(np.sum(inside) / positions.shape[0])


def _validate_vector_field(
    grid: Grid2D,
    field: np.ndarray,
    name: str,
) -> None:
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")


def _validate_vector_field_shape(field: np.ndarray) -> None:
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError("Expected vector field with shape (Ny, Nx, 2).")


def _validate_positions(positions: np.ndarray) -> None:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected positions with shape (Np, 2).")