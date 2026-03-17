"""
3D Initial Conditions Module
=============================
Provides 3D velocity field initialization for Large Eddy Simulation using
stochastic integral representations (Qian et al., 2025).

Convention:
- Scalar fields: shape (Nz, Ny, Nx)
- Vector fields: shape (Nz, Ny, Nx, 3) where last dimension is (u_x, u_y, u_z)
- Grid coordinates: X, Y, Z from meshgrid(z, y, x, indexing="ij")
"""

import numpy as np
from les.grid import Grid3D


def taylor_green_velocity_3d(grid: Grid3D) -> np.ndarray:
    """
    Classic 3D Taylor-Green vortex initial condition.

    The velocity field is:
        u_x(x, y, z) =  sin(k*x) * cos(k*y) * cos(k*z)
        u_y(x, y, z) = -cos(k*x) * sin(k*y) * cos(k*z)
        u_z(x, y, z) = 0

    where k = π / L_box with L_box the domain half-width.

    This field is exactly divergence-free:
        ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
        = k*cos(k*x)*cos(k*y)*cos(k*z) - k*cos(k*x)*cos(k*y)*cos(k*z) + 0 = 0

    Parameters
    ----------
    grid : Grid3D
        Grid instance with attributes X, Y, Z of shape (Nz, Ny, Nx).

    Returns
    -------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).

    References
    ----------
    Qian et al. (2025) - 3D LES stochastic integral representation.
    """
    # Compute wavenumber using grid's actual domain size
    k = np.pi / grid.L_box

    # Allocate velocity field
    velocity = grid.zeros_vector()

    # Compute velocity components
    velocity[..., 0] = (np.sin(k * grid.X) *
                        np.cos(k * grid.Y) *
                        np.cos(k * grid.Z))
    velocity[..., 1] = (-np.cos(k * grid.X) *
                        np.sin(k * grid.Y) *
                        np.cos(k * grid.Z))
    velocity[..., 2] = 0.0

    _validate_vector_field(velocity)
    return velocity


def gaussian_vortex_velocity_3d(grid: Grid3D,
                                 strength: float = 52.5,
                                 sigma: float = 1.57,
                                 center: tuple = (0.0, 0.0, 0.0)) -> np.ndarray:
    """
    3D localized Gaussian vortex: axisymmetric swirl in xy-plane.

    The velocity field models a vortex core centered at (x0, y0, z0):
        r² = (X - x0)² + (Y - y0)² + (Z - z0)²
        factor = strength * exp(-r² / (2*σ²))

        u_x = -(Y - y0) * factor
        u_y =  (X - x0) * factor
        u_z = 0

    This represents an irrotational vortex filament in the z-direction,
    localized in space by Gaussian envelope.

    Parameters
    ----------
    grid : Grid3D
        Grid instance with attributes X, Y, Z of shape (Nz, Ny, Nx).
    strength : float
        Vortex strength (circulation-like parameter). Default 52.5 (Qian scale).
    sigma : float
        Gaussian width parameter. Default 1.57 (Qian scale).
    center : tuple
        Vortex center (x0, y0, z0). Default (0, 0, 0).

    Returns
    -------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).

    References
    ----------
    Qian et al. (2025) - Localized vortex parametrization.
    """
    x0, y0, z0 = center

    # Compute squared distance from vortex center
    r_squared = ((grid.X - x0)**2 +
                 (grid.Y - y0)**2 +
                 (grid.Z - z0)**2)

    # Gaussian envelope factor
    factor = strength * np.exp(-r_squared / (2.0 * sigma**2))

    # Allocate velocity field
    velocity = grid.zeros_vector()

    # Swirl pattern in xy-plane
    velocity[..., 0] = -(grid.Y - y0) * factor
    velocity[..., 1] = (grid.X - x0) * factor
    velocity[..., 2] = 0.0

    _validate_vector_field(velocity)
    return velocity


def two_gaussian_vortices_velocity_3d(grid: Grid3D,
                                       strength1: float = 52.5,
                                       strength2: float = 52.5,
                                       sigma1: float = 1.57,
                                       sigma2: float = 1.57,
                                       center1: tuple = (-1.5, 0.0, 0.0),
                                       center2: tuple = (1.5, 0.0, 0.0)) -> np.ndarray:
    """
    Superposition of two 3D Gaussian vortices.

    Combines two localized vortex structures at different centers.
    Useful for testing vortex-vortex interactions and reconnection phenomena.

    Parameters
    ----------
    grid : Grid3D
        Grid instance.
    strength1, strength2 : float
        Vortex strengths.
    sigma1, sigma2 : float
        Gaussian widths.
    center1, center2 : tuple
        Vortex centers (x, y, z).

    Returns
    -------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    """
    vel1 = gaussian_vortex_velocity_3d(grid, strength1, sigma1, center1)
    vel2 = gaussian_vortex_velocity_3d(grid, strength2, sigma2, center2)

    velocity = vel1 + vel2

    _validate_vector_field(velocity)
    return velocity


def compute_divergence(velocity: np.ndarray,
                      dx: float = 1.0,
                      dy: float = 1.0,
                      dz: float = 1.0) -> np.ndarray:
    """
    Compute velocity field divergence using central differences.

    ∇·u = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    divergence : np.ndarray
        Scalar field of shape (Nz, Ny, Nx).
    """
    _validate_vector_field(velocity)
    Nz, Ny, Nx = velocity.shape[:3]

    divergence = np.zeros((Nz, Ny, Nx), dtype=velocity.dtype)

    # ∂u_x/∂x (central differences, periodic BC)
    divergence += (np.roll(velocity[..., 0], -1, axis=2) -
                   np.roll(velocity[..., 0],  1, axis=2)) / (2.0 * dx)

    # ∂u_y/∂y (central differences, periodic BC)
    divergence += (np.roll(velocity[..., 1], -1, axis=1) -
                   np.roll(velocity[..., 1],  1, axis=1)) / (2.0 * dy)

    # ∂u_z/∂z (central differences, periodic BC)
    divergence += (np.roll(velocity[..., 2], -1, axis=0) -
                   np.roll(velocity[..., 2],  1, axis=0)) / (2.0 * dz)

    return divergence


def compute_speed(velocity: np.ndarray) -> np.ndarray:
    """
    Compute pointwise velocity magnitude.

    speed = sqrt(u_x² + u_y² + u_z²)

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).

    Returns
    -------
    speed : np.ndarray
        Scalar field of shape (Nz, Ny, Nx).
    """
    _validate_vector_field(velocity)
    speed = np.sqrt(velocity[..., 0]**2 +
                    velocity[..., 1]**2 +
                    velocity[..., 2]**2)
    return speed


def trusted_max_divergence(velocity: np.ndarray,
                          dx: float = 1.0,
                          dy: float = 1.0,
                          dz: float = 1.0,
                          restrict_to_trusted: bool = True) -> float:
    """
    Maximum absolute divergence (should be ~0 for incompressible flow).

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    dx, dy, dz : float
        Grid spacings.
    restrict_to_trusted : bool
        If True, excludes boundary layers (first/last grid point in each dim).

    Returns
    -------
    max_div : float
        Maximum absolute divergence value.
    """
    divergence = compute_divergence(velocity, dx, dy, dz)

    if restrict_to_trusted:
        divergence = divergence[1:-1, 1:-1, 1:-1]

    return np.max(np.abs(divergence))


def trusted_mean_abs_divergence(velocity: np.ndarray,
                               dx: float = 1.0,
                               dy: float = 1.0,
                               dz: float = 1.0,
                               restrict_to_trusted: bool = True) -> float:
    """
    Mean absolute divergence (should be ~0 for incompressible flow).

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    dx, dy, dz : float
        Grid spacings.
    restrict_to_trusted : bool
        If True, excludes boundary layers.

    Returns
    -------
    mean_div : float
        Mean absolute divergence value.
    """
    divergence = compute_divergence(velocity, dx, dy, dz)

    if restrict_to_trusted:
        divergence = divergence[1:-1, 1:-1, 1:-1]

    return np.mean(np.abs(divergence))


def trusted_speed(velocity: np.ndarray,
                 restrict_to_trusted: bool = True) -> np.ndarray:
    """
    Compute pointwise speed, optionally restricting to interior points.

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    restrict_to_trusted : bool
        If True, returns speed only at interior points.

    Returns
    -------
    speed : np.ndarray
        Scalar field, shape (Nz, Ny, Nx) or (Nz-2, Ny-2, Nx-2).
    """
    speed = compute_speed(velocity)

    if restrict_to_trusted:
        speed = speed[1:-1, 1:-1, 1:-1]

    return speed


def _validate_vector_field(velocity: np.ndarray) -> None:
    """
    Validate that velocity has correct shape (*, *, *, 3).

    Parameters
    ----------
    velocity : np.ndarray
        Field to validate.

    Raises
    ------
    ValueError
        If shape[-1] != 3.
    """
    if velocity.shape[-1] != 3:
        raise ValueError(
            f"Vector field must have shape (..., 3), got {velocity.shape}"
        )
