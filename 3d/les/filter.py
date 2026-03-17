"""
3D Gaussian kernel utilities for LES filtering on R^3.

The Gaussian kernel in 3D is:
    G(x, y, z; σ) = (2π σ²)^{-3/2} exp(-(x² + y² + z²) / (2σ²))

Discrete convolution uses finite-support stencils computed from
local kernel evaluations on the grid.
"""

import numpy as np
from .config import SimulationConfig


def gaussian_kernel_3d(dx: float, dy: float, dz: float, sigma: float) -> float:
    r"""
    Evaluate the 3D Gaussian kernel at a point offset (dx, dy, dz).

    Parameters
    ----------
    dx, dy, dz : float
        Spatial offsets in x, y, z directions.
    sigma : float
        Gaussian width parameter (standard deviation).

    Returns
    -------
    kernel_value : float
        G(dx, dy, dz; σ) = (2π σ²)^{-3/2} exp(-(dx² + dy² + dz²) / (2σ²))
    """
    normalization = (2.0 * np.pi * sigma**2) ** (-1.5)
    r_squared = dx**2 + dy**2 + dz**2
    return normalization * np.exp(-r_squared / (2.0 * sigma**2))


def stencil_half_width_xyz(
    config: SimulationConfig, sigma: float
) -> tuple[int, int, int]:
    """
    Compute the half-width of the stencil in each direction.

    The stencil extends |offset| <= cutoff_sigma * sigma in each direction.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with grid spacings.
    sigma : float
        Gaussian width parameter.

    Returns
    -------
    hx, hy, hz : int
        Half-widths (number of grid cells) in x, y, z directions.
    """
    cutoff = config.filter_cutoff_sigma * sigma
    hx = int(np.ceil(cutoff / config.L_box * config.Nx / 2.0))
    hy = int(np.ceil(cutoff / config.L_box * config.Ny / 2.0))
    hz = int(np.ceil(cutoff / config.L_box * config.Nz / 2.0))
    return hx, hy, hz


def make_local_kernel(
    grid_spacing_xyz: tuple[float, float, float],
    half_widths_xyz: tuple[int, int, int],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a 3D Gaussian stencil kernel.

    Parameters
    ----------
    grid_spacing_xyz : tuple[float, float, float]
        Grid spacings (dx, dy, dz).
    half_widths_xyz : tuple[int, int, int]
        Half-widths (hx, hy, hz) in grid cells.
    sigma : float
        Gaussian width parameter.

    Returns
    -------
    offsets_x : ndarray, shape (2*hx + 1,)
        x-offsets in physical coordinates.
    offsets_y : ndarray, shape (2*hy + 1,)
        y-offsets in physical coordinates.
    offsets_z : ndarray, shape (2*hz + 1,)
        z-offsets in physical coordinates.
    kernel_3d : ndarray, shape (2*hz + 1, 2*hy + 1, 2*hx + 1)
        3D kernel evaluated on the stencil grid.
    """
    dx, dy, dz = grid_spacing_xyz
    hx, hy, hz = half_widths_xyz

    # Offset arrays in physical coordinates
    offsets_x = np.arange(-hx, hx + 1) * dx
    offsets_y = np.arange(-hy, hy + 1) * dy
    offsets_z = np.arange(-hz, hz + 1) * dz

    # Construct 3D kernel: shape (2*hz + 1, 2*hy + 1, 2*hx + 1)
    kernel_3d = np.zeros((2 * hz + 1, 2 * hy + 1, 2 * hx + 1))

    for kk, oz in enumerate(offsets_z):
        for jj, oy in enumerate(offsets_y):
            for ii, ox in enumerate(offsets_x):
                kernel_3d[kk, jj, ii] = gaussian_kernel_3d(ox, oy, oz, sigma)

    return offsets_x, offsets_y, offsets_z, kernel_3d


def truncate_kernel_by_radius(
    kernel_3d: np.ndarray,
    grid_spacing_xyz: tuple[float, float, float],
    half_widths_xyz: tuple[int, int, int],
    cutoff_radius: float,
) -> np.ndarray:
    """
    Truncate a 3D kernel by setting to zero points beyond a radius.

    Parameters
    ----------
    kernel_3d : ndarray, shape (2*hz + 1, 2*hy + 1, 2*hx + 1)
        Input kernel.
    grid_spacing_xyz : tuple[float, float, float]
        Grid spacings (dx, dy, dz).
    half_widths_xyz : tuple[int, int, int]
        Half-widths (hx, hy, hz) in grid cells.
    cutoff_radius : float
        Points r > cutoff_radius are zeroed.

    Returns
    -------
    truncated : ndarray
        Kernel with points beyond cutoff_radius set to zero.
    """
    dx, dy, dz = grid_spacing_xyz
    hx, hy, hz = half_widths_xyz
    cutoff_sq = cutoff_radius**2

    offsets_x = np.arange(-hx, hx + 1) * dx
    offsets_y = np.arange(-hy, hy + 1) * dy
    offsets_z = np.arange(-hz, hz + 1) * dz

    truncated = kernel_3d.copy()

    for kk, oz in enumerate(offsets_z):
        for jj, oy in enumerate(offsets_y):
            for ii, ox in enumerate(offsets_x):
                r_sq = ox**2 + oy**2 + oz**2
                if r_sq > cutoff_sq:
                    truncated[kk, jj, ii] = 0.0

    return truncated


def normalize_discrete_kernel(
    kernel_3d: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """
    Normalize kernel to integrate (sum) to 1 over the grid.

    Parameters
    ----------
    kernel_3d : ndarray, shape (2*hz + 1, 2*hy + 1, 2*hx + 1)
        Input kernel (unnormalized).
    dx, dy, dz : float
        Grid spacings in each direction.

    Returns
    -------
    normalized : ndarray
        Kernel scaled so that sum(kernel * dx * dy * dz) ≈ 1.
    """
    cell_volume = dx * dy * dz
    integral = np.sum(kernel_3d) * cell_volume
    if integral > 0.0:
        return kernel_3d / integral
    else:
        return kernel_3d.copy()
