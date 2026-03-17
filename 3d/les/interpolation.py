"""
Trilinear interpolation utilities for 3D LES calculations.

Extends 2D bilinear interpolation to 3D trilinear interpolation for scalar,
vector, and tensor fields on uniform Cartesian grids.

Field conventions:
  - Scalar fields: shape (Nz, Ny, Nx)
  - Vector fields: shape (Nz, Ny, Nx, 3) with components [u_x, u_y, u_z]
  - Tensor fields: shape (Nz, Ny, Nx, 3, 3) with gradU[..., i, j] = du_i/dx_j

Grid indexing:
  - axis 0: z-direction
  - axis 1: y-direction
  - axis 2: x-direction
"""

import numpy as np

from les.grid import CartesianGrid3D


def trilinear_interpolate_scalar(
    grid: CartesianGrid3D,
    field: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Trilinear interpolation of a scalar field at particle positions.

    Interpolates field values at arbitrary positions within the computational
    domain using 8-corner trilinear basis functions. The interpolation weights
    are computed from fractional distances within the cell.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Scalar field of shape (Nz, Ny, Nx).
    positions
        Particle positions of shape (Np, 3) with columns [x, y, z].

    Returns
    -------
    values : ndarray, shape (Np,)
        Interpolated scalar values at each position.

    Notes
    -----
    Uses 8-point trilinear interpolation:
        f_interp = sum over 8 corners of (weight * field_value)
    where weights depend on fractional cell coordinates (tx, ty, tz).
    """
    _validate_scalar_field(grid, field)
    positions = _validate_positions(positions)

    k0, j0, i0 = grid.cell_indices(positions)
    # Periodic wrapping for upper corner indices
    k1 = (k0 + 1) % grid.Nz
    j1 = (j0 + 1) % grid.Ny
    i1 = (i0 + 1) % grid.Nx

    # Grid node coordinates of cell origin (lower-left-back)
    x0 = grid.x[i0]
    y0 = grid.y[j0]
    z0 = grid.z[k0]

    # Particle coordinates
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Fractional coordinates within cell [0, 1]
    # Use dx directly (not x1-x0) because x1 may have wrapped periodically
    tx = (x - x0) / grid.dx
    ty = (y - y0) / grid.dy
    tz = (z - z0) / grid.dz
    # Clamp to [0,1] for numerical safety (particles exactly on boundary)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)
    tz = np.clip(tz, 0.0, 1.0)

    # 8 corner values: ordered as (k_idx, j_idx, i_idx) in binary
    f000 = field[k0, j0, i0]  # (0, 0, 0)
    f001 = field[k0, j0, i1]  # (0, 0, 1)
    f010 = field[k0, j1, i0]  # (0, 1, 0)
    f011 = field[k0, j1, i1]  # (0, 1, 1)
    f100 = field[k1, j0, i0]  # (1, 0, 0)
    f101 = field[k1, j0, i1]  # (1, 0, 1)
    f110 = field[k1, j1, i0]  # (1, 1, 0)
    f111 = field[k1, j1, i1]  # (1, 1, 1)

    # Trilinear interpolation
    values = (
        (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * f000
        + tx * (1.0 - ty) * (1.0 - tz) * f001
        + (1.0 - tx) * ty * (1.0 - tz) * f010
        + tx * ty * (1.0 - tz) * f011
        + (1.0 - tx) * (1.0 - ty) * tz * f100
        + tx * (1.0 - ty) * tz * f101
        + (1.0 - tx) * ty * tz * f110
        + tx * ty * tz * f111
    )
    return values


def trilinear_interpolate_vector(
    grid: CartesianGrid3D,
    field: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Trilinear interpolation of a vector field at particle positions.

    Interpolates each component of the vector field independently using
    trilinear basis functions. The same interpolation weights are used
    for all three components.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Vector field of shape (Nz, Ny, Nx, 3) with components [u_x, u_y, u_z].
    positions
        Particle positions of shape (Np, 3) with columns [x, y, z].

    Returns
    -------
    values : ndarray, shape (Np, 3)
        Interpolated vector values at each position.
    """
    _validate_vector_field(grid, field)
    positions = _validate_positions(positions)

    k0, j0, i0 = grid.cell_indices(positions)
    k1 = (k0 + 1) % grid.Nz
    j1 = (j0 + 1) % grid.Ny
    i1 = (i0 + 1) % grid.Nx

    x0 = grid.x[i0]
    y0 = grid.y[j0]
    z0 = grid.z[k0]

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Fractional coordinates with broadcasting for vector components
    tx = np.clip((x - x0) / grid.dx, 0.0, 1.0)[:, None]
    ty = np.clip((y - y0) / grid.dy, 0.0, 1.0)[:, None]
    tz = np.clip((z - z0) / grid.dz, 0.0, 1.0)[:, None]

    # 8 corner values with all 3 components
    f000 = field[k0, j0, i0, :]
    f001 = field[k0, j0, i1, :]
    f010 = field[k0, j1, i0, :]
    f011 = field[k0, j1, i1, :]
    f100 = field[k1, j0, i0, :]
    f101 = field[k1, j0, i1, :]
    f110 = field[k1, j1, i0, :]
    f111 = field[k1, j1, i1, :]

    values = (
        (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * f000
        + tx * (1.0 - ty) * (1.0 - tz) * f001
        + (1.0 - tx) * ty * (1.0 - tz) * f010
        + tx * ty * (1.0 - tz) * f011
        + (1.0 - tx) * (1.0 - ty) * tz * f100
        + tx * (1.0 - ty) * tz * f101
        + (1.0 - tx) * ty * tz * f110
        + tx * ty * tz * f111
    )
    return values


def trilinear_interpolate_tensor(
    grid: CartesianGrid3D,
    field: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Trilinear interpolation of a 3x3 tensor field at particle positions.

    Interpolates each component (i, j) of the tensor field independently using
    trilinear basis functions. The same interpolation weights are used for
    all nine tensor components.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Tensor field of shape (Nz, Ny, Nx, 3, 3).
    positions
        Particle positions of shape (Np, 3) with columns [x, y, z].

    Returns
    -------
    values : ndarray, shape (Np, 3, 3)
        Interpolated tensor values at each position.
    """
    _validate_tensor_field(grid, field)
    positions = _validate_positions(positions)

    k0, j0, i0 = grid.cell_indices(positions)
    k1 = (k0 + 1) % grid.Nz
    j1 = (j0 + 1) % grid.Ny
    i1 = (i0 + 1) % grid.Nx

    x0 = grid.x[i0]
    y0 = grid.y[j0]
    z0 = grid.z[k0]

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Fractional coordinates with broadcasting for tensor components
    tx = np.clip((x - x0) / grid.dx, 0.0, 1.0)[:, None, None]
    ty = np.clip((y - y0) / grid.dy, 0.0, 1.0)[:, None, None]
    tz = np.clip((z - z0) / grid.dz, 0.0, 1.0)[:, None, None]

    # 8 corner values with all tensor components
    f000 = field[k0, j0, i0, :, :]
    f001 = field[k0, j0, i1, :, :]
    f010 = field[k0, j1, i0, :, :]
    f011 = field[k0, j1, i1, :, :]
    f100 = field[k1, j0, i0, :, :]
    f101 = field[k1, j0, i1, :, :]
    f110 = field[k1, j1, i0, :, :]
    f111 = field[k1, j1, i1, :, :]

    values = (
        (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * f000
        + tx * (1.0 - ty) * (1.0 - tz) * f001
        + (1.0 - tx) * ty * (1.0 - tz) * f010
        + tx * ty * (1.0 - tz) * f011
        + (1.0 - tx) * (1.0 - ty) * tz * f100
        + tx * (1.0 - ty) * tz * f101
        + (1.0 - tx) * ty * tz * f110
        + tx * ty * tz * f111
    )
    return values


def sample_scalar_on_grid_nodes(
    grid: CartesianGrid3D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a scalar field on grid nodes and return positions and values.

    Flattens the 3D field and returns grid node coordinates as particle
    positions. Useful for exporting field data to particle representation.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Scalar field of shape (Nz, Ny, Nx).
    trusted_only
        If True, only sample points in the trusted region.

    Returns
    -------
    positions : ndarray, shape (Np, 3)
        Grid node coordinates [x, y, z].
    values : ndarray, shape (Np,)
        Scalar field values at grid nodes.
    """
    _validate_scalar_field(grid, field)

    if trusted_only:
        positions = grid.coordinates_as_particles()
        trusted_mask = grid.points_in_trusted_region(positions)
        return positions[trusted_mask], field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.ravel().copy()
        return positions, values


def sample_vector_on_grid_nodes(
    grid: CartesianGrid3D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a vector field on grid nodes and return positions and values.

    Flattens the 3D field (keeping vector components) and returns grid node
    coordinates as particle positions. Useful for exporting field data.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Vector field of shape (Nz, Ny, Nx, 3).
    trusted_only
        If True, only sample points in the trusted region.

    Returns
    -------
    positions : ndarray, shape (Np, 3)
        Grid node coordinates [x, y, z].
    values : ndarray, shape (Np, 3)
        Vector field values at grid nodes.
    """
    _validate_vector_field(grid, field)

    if trusted_only:
        positions = grid.coordinates_as_particles()
        trusted_mask = grid.points_in_trusted_region(positions)
        return positions[trusted_mask], field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 3).copy()
        return positions, values


def sample_tensor_on_grid_nodes(
    grid: CartesianGrid3D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a tensor field on grid nodes and return positions and values.

    Flattens the 3D field (keeping tensor components) and returns grid node
    coordinates as particle positions. Useful for exporting field data.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Tensor field of shape (Nz, Ny, Nx, 3, 3).
    trusted_only
        If True, only sample points in the trusted region.

    Returns
    -------
    positions : ndarray, shape (Np, 3)
        Grid node coordinates [x, y, z].
    values : ndarray, shape (Np, 3, 3)
        Tensor field values at grid nodes.
    """
    _validate_tensor_field(grid, field)

    if trusted_only:
        positions = grid.coordinates_as_particles()
        trusted_mask = grid.points_in_trusted_region(positions)
        return positions[trusted_mask], field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 3, 3).copy()
        return positions, values


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    """
    Validate and cast particle positions to float64.

    Parameters
    ----------
    positions : ndarray
        Particle positions to validate.

    Returns
    -------
    positions : ndarray, dtype float64
        Validated positions of shape (Np, 3).

    Raises
    ------
    ValueError
        If positions do not have shape (Np, 3).
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected positions with shape (Np, 3), got {positions.shape}."
        )
    return positions


def _validate_scalar_field(grid: CartesianGrid3D, field: np.ndarray) -> None:
    """
    Validate scalar field shape.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Field to validate.

    Raises
    ------
    ValueError
        If field shape does not match grid.shape (Nz, Ny, Nx).
    """
    if field.ndim != 3:
        raise ValueError(
            f"Expected scalar field with ndim=3, got {field.ndim}."
        )
    if field.shape != grid.shape:
        raise ValueError(
            f"Expected scalar field with shape {grid.shape}, got {field.shape}."
        )


def _validate_vector_field(grid: CartesianGrid3D, field: np.ndarray) -> None:
    """
    Validate vector field shape.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Field to validate.

    Raises
    ------
    ValueError
        If field shape does not match grid.vector_shape (Nz, Ny, Nx, 3).
    """
    if field.ndim != 4:
        raise ValueError(
            f"Expected vector field with ndim=4, got {field.ndim}."
        )
    if field.shape[-1] != 3:
        raise ValueError(
            f"Expected vector field with 3 components, got {field.shape[-1]}."
        )
    if field.shape != grid.vector_shape:
        raise ValueError(
            f"Expected vector field with shape {grid.vector_shape}, got {field.shape}."
        )


def _validate_tensor_field(grid: CartesianGrid3D, field: np.ndarray) -> None:
    """
    Validate tensor field shape.

    Parameters
    ----------
    grid
        3D Cartesian grid object (CartesianGrid3D).
    field
        Field to validate.

    Raises
    ------
    ValueError
        If field shape does not match grid.tensor_shape (Nz, Ny, Nx, 3, 3).
    """
    if field.ndim != 5:
        raise ValueError(
            f"Expected tensor field with ndim=5, got {field.ndim}."
        )
    if field.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected tensor field with shape (..., 3, 3), got {field.shape}."
        )
    if field.shape != grid.tensor_shape:
        raise ValueError(
            f"Expected tensor field with shape {grid.tensor_shape}, got {field.shape}."
        )
