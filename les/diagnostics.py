"""
Diagnostics for the random LES simulation.

Includes Qian's normalised incompressibility test (Section 4.3.1):

    test(t) = (1/N) * sum_x  |div u(x,t)| / ||grad u(x,t)||_2

where ||grad u||_2 is the Frobenius norm of the velocity gradient tensor.
This measures how much of the velocity gradient is compressive versus
shearing/rotational.  A value much less than 1 indicates approximate
incompressibility.
"""

import numpy as np

from les.grid import Grid2D
from les.differential_operators import divergence, grad_velocity


# ------------------------------------------------------------------ #
#  Kinetic energy
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Speed
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Raw divergence stats  (existing)
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Qian's normalised incompressibility test  (Section 4.3.1)
# ------------------------------------------------------------------ #

def _pointwise_normalised_divergence(
    U: np.ndarray,
    dx: float,
    dy: float,
    grad_norm_floor: float = 1.0e-14,
) -> np.ndarray:
    """
    Compute the pointwise ratio  |div u(x)| / ||grad u(x)||_2
    at every grid point.

    Parameters
    ----------
    U : (Ny, Nx, 2)
    dx, dy : float
    grad_norm_floor : float
        Minimum value of ||grad u|| used in the denominator to avoid
        division by zero in quiescent regions.

    Returns
    -------
    ratio : (Ny, Nx)
        Pointwise  |div u| / max(||grad u||_2, floor).
    """
    divU = divergence(U, dx, dy)                         # (Ny, Nx)
    gradU = grad_velocity(U, dx, dy)                     # (Ny, Nx, 2, 2)

    # Frobenius norm:  ||grad u||_2 = sqrt( sum_{i,j} (du_i/dx_j)^2 )
    gradU_norm = np.sqrt(np.sum(gradU * gradU, axis=(-2, -1)))  # (Ny, Nx)

    # Floor the denominator to avoid 0/0 in uniform regions
    safe_denom = np.maximum(gradU_norm, grad_norm_floor)

    return np.abs(divU) / safe_denom


def incompressibility_test(
    U: np.ndarray,
    grid: Grid2D,
    grad_norm_floor: float = 1.0e-14,
) -> float:
    """
    Qian's incompressibility diagnostic on the full padded box:

        test(t) = (1/N) sum_x  |div u(x)| / ||grad u(x)||_2

    A value much less than 1 indicates approximate incompressibility.

    Parameters
    ----------
    U : (Ny, Nx, 2)
    grid : Grid2D
    grad_norm_floor : float
        Floor for ||grad u|| to avoid 0/0.

    Returns
    -------
    test_value : float
    """
    _validate_vector_field(grid, U, "U")

    ratio = _pointwise_normalised_divergence(
        U, grid.dx, grid.dy, grad_norm_floor,
    )
    return float(np.mean(ratio))


def trusted_incompressibility_test(
    U: np.ndarray,
    grid: Grid2D,
    grad_norm_floor: float = 1.0e-14,
) -> float:
    """
    Qian's incompressibility diagnostic restricted to the trusted interior.

    Returns
    -------
    test_value : float
    """
    _validate_vector_field(grid, U, "U")

    ratio = _pointwise_normalised_divergence(
        U, grid.dx, grid.dy, grad_norm_floor,
    )
    ratio_trusted = grid.restrict_to_trusted(ratio)
    return float(np.mean(ratio_trusted))


# ------------------------------------------------------------------ #
#  Relative L2 change
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Particle diagnostics
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Validation helpers
# ------------------------------------------------------------------ #

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