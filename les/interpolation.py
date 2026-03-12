import numpy as np

from les.grid import Grid2D


def bilinear_interpolate_scalar(
    grid: Grid2D,
    field: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of a scalar field at particle positions.

    Parameters
    ----------
    grid
        Padded computational grid.
    field
        Scalar field of shape (Ny, Nx).
    positions
        Particle positions of shape (Np, 2).

    Returns
    -------
    values : ndarray, shape (Np,)
        Interpolated scalar values.
    """
    _validate_scalar_field(grid, field)
    positions = _validate_positions(positions)

    j, i = grid.cell_indices(positions)

    x0 = grid.x[i]
    x1 = grid.x[i + 1]
    y0 = grid.y[j]
    y1 = grid.y[j + 1]

    x = positions[:, 0]
    y = positions[:, 1]

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    f00 = field[j, i]
    f10 = field[j, i + 1]
    f01 = field[j + 1, i]
    f11 = field[j + 1, i + 1]

    values = (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )
    return values


def bilinear_interpolate_vector(
    grid: Grid2D,
    field: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of a vector field at particle positions.

    Parameters
    ----------
    field
        Vector field of shape (Ny, Nx, 2).

    Returns
    -------
    values : ndarray, shape (Np, 2)
        Interpolated vector values.
    """
    _validate_vector_field(grid, field)
    positions = _validate_positions(positions)

    j, i = grid.cell_indices(positions)

    x0 = grid.x[i]
    x1 = grid.x[i + 1]
    y0 = grid.y[j]
    y1 = grid.y[j + 1]

    x = positions[:, 0]
    y = positions[:, 1]

    tx = ((x - x0) / (x1 - x0))[:, None]
    ty = ((y - y0) / (y1 - y0))[:, None]

    f00 = field[j, i, :]
    f10 = field[j, i + 1, :]
    f01 = field[j + 1, i, :]
    f11 = field[j + 1, i + 1, :]

    values = (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )
    return values


def bilinear_interpolate_tensor(
    grid: Grid2D,
    field: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of a 2x2 tensor field at particle positions.

    Parameters
    ----------
    field
        Tensor field of shape (Ny, Nx, 2, 2).

    Returns
    -------
    values : ndarray, shape (Np, 2, 2)
        Interpolated tensor values.
    """
    _validate_tensor_field(grid, field)
    positions = _validate_positions(positions)

    j, i = grid.cell_indices(positions)

    x0 = grid.x[i]
    x1 = grid.x[i + 1]
    y0 = grid.y[j]
    y1 = grid.y[j + 1]

    x = positions[:, 0]
    y = positions[:, 1]

    tx = ((x - x0) / (x1 - x0))[:, None, None]
    ty = ((y - y0) / (y1 - y0))[:, None, None]

    f00 = field[j, i, :, :]
    f10 = field[j, i + 1, :, :]
    f01 = field[j + 1, i, :, :]
    f11 = field[j + 1, i + 1, :, :]

    values = (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )
    return values


def sample_scalar_on_grid_nodes(
    grid: Grid2D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a scalar field on grid nodes and return positions and values.

    Returns
    -------
    positions : ndarray, shape (Np, 2)
    values    : ndarray, shape (Np,)
    """
    _validate_scalar_field(grid, field)

    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
        values = field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.ravel().copy()

    return positions, values


def sample_vector_on_grid_nodes(
    grid: Grid2D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a vector field on grid nodes and return positions and values.

    Returns
    -------
    positions : ndarray, shape (Np, 2)
    values    : ndarray, shape (Np, 2)
    """
    _validate_vector_field(grid, field)

    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
        values = field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 2).copy()

    return positions, values


def sample_tensor_on_grid_nodes(
    grid: Grid2D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a tensor field on grid nodes and return positions and values.

    Returns
    -------
    positions : ndarray, shape (Np, 2)
    values    : ndarray, shape (Np, 2, 2)
    """
    _validate_tensor_field(grid, field)

    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
        values = field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 2, 2).copy()

    return positions, values


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Expected positions with shape (Np, 2), got {positions.shape}."
        )
    return positions


def _validate_scalar_field(grid: Grid2D, field: np.ndarray) -> None:
    if field.shape != grid.shape:
        raise ValueError(
            f"Expected scalar field with shape {grid.shape}, got {field.shape}."
        )


def _validate_vector_field(grid: Grid2D, field: np.ndarray) -> None:
    if field.shape != grid.vector_shape:
        raise ValueError(
            f"Expected vector field with shape {grid.vector_shape}, got {field.shape}."
        )


def _validate_tensor_field(grid: Grid2D, field: np.ndarray) -> None:
    if field.shape != grid.tensor_shape:
        raise ValueError(
            f"Expected tensor field with shape {grid.tensor_shape}, got {field.shape}."
        )