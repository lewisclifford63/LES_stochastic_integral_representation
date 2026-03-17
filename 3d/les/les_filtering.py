"""
3D spatial filtering for LES stochastic integral representation.
Extends 2D Gaussian filtering to 3D domain.
Oxford Mathematics Masters dissertation, Qian et al. (2025).

The 3D isotropic Gaussian kernel:
    G(r; sigma) = (2*pi*sigma^2)^{-3/2} exp(-|r|^2 / (2*sigma^2))

For efficiency on small grids, we precompute the full 3D stencil kernel
and apply it via slicing (no scipy dependency).
"""

import numpy as np
from les.grid import Grid3D
from les.filter import gaussian_kernel_3d


def _compute_half_widths(sigma: float, cutoff_sigma: float,
                         dx: float, dy: float, dz: float) -> tuple[int, int, int]:
    """Compute stencil half-widths (in grid cells) for each direction."""
    cutoff = cutoff_sigma * sigma
    hx = int(np.ceil(cutoff / dx))
    hy = int(np.ceil(cutoff / dy))
    hz = int(np.ceil(cutoff / dz))
    return hz, hy, hx


def _build_kernel_3d(sigma: float, dx: float, dy: float, dz: float,
                     hz: int, hy: int, hx: int) -> np.ndarray:
    """
    Build a 3D Gaussian stencil kernel of shape (2*hz+1, 2*hy+1, 2*hx+1).
    Uses vectorised evaluation of gaussian_kernel_3d.
    """
    oz = np.arange(-hz, hz + 1) * dz
    oy = np.arange(-hy, hy + 1) * dy
    ox = np.arange(-hx, hx + 1) * dx

    # Meshgrid for vectorised kernel evaluation
    OZ, OY, OX = np.meshgrid(oz, oy, ox, indexing="ij")
    K = gaussian_kernel_3d(OX, OY, OZ, sigma)
    return K


def apply_spatial_filter(grid, U, sigma, cutoff_sigma, normalize=True):
    """
    Apply 3D Gaussian spatial filter to a 3D vector field.

    Uses scipy.ndimage.convolve if available, otherwise falls back
    to a direct stencil approach.

    Parameters
    ----------
    grid : Grid3D
        Grid object with spacing dx, dy, dz and domain bounds.
    U : ndarray of shape (Nz, Ny, Nx, 3)
        Input vector field [u, v, w] at grid points.
    sigma : float
        Isotropic filter width (standard deviation of Gaussian).
    cutoff_sigma : float
        Truncate Gaussian kernel at this many sigma.
    normalize : bool, optional
        If True, normalize by kernel mass. Default True.

    Returns
    -------
    U_bar : ndarray of shape (Nz, Ny, Nx, 3)
        Filtered vector field.
    """
    hz, hy, hx = _compute_half_widths(sigma, cutoff_sigma,
                                       grid.dx, grid.dy, grid.dz)

    # Build kernel
    K = _build_kernel_3d(sigma, grid.dx, grid.dy, grid.dz, hz, hy, hx)

    cell_vol = grid.cell_volume

    if normalize:
        mass = np.sum(K) * cell_vol
        if mass > 0:
            K = K / mass

    # Try scipy for fast convolution
    try:
        from scipy.ndimage import convolve
        U_bar = np.zeros_like(U)
        for c in range(3):
            U_bar[..., c] = convolve(U[..., c], K * cell_vol, mode='constant', cval=0.0)
        return U_bar
    except ImportError:
        pass

    # Fallback: direct stencil (works on small grids)
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    U_bar = np.zeros_like(U)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                # Stencil bounds (clipped to domain)
                k0, k1 = max(0, k - hz), min(Nz - 1, k + hz)
                j0, j1 = max(0, j - hy), min(Ny - 1, j + hy)
                i0, i1 = max(0, i - hx), min(Nx - 1, i + hx)

                # Corresponding kernel indices
                kk0, kk1 = k0 - (k - hz), k1 - (k - hz)
                jj0, jj1 = j0 - (j - hy), j1 - (j - hy)
                ii0, ii1 = i0 - (i - hx), i1 - (i - hx)

                K_local = K[kk0:kk1+1, jj0:jj1+1, ii0:ii1+1]
                patch = U[k0:k1+1, j0:j1+1, i0:i1+1, :]

                local_mass = np.sum(K_local) * cell_vol
                if normalize and local_mass > 0:
                    K_norm = K_local / local_mass
                else:
                    K_norm = K_local

                for c in range(3):
                    U_bar[k, j, i, c] = np.sum(K_norm * patch[..., c]) * cell_vol

    return U_bar


def apply_spatial_filter_scalar(grid, field, sigma, cutoff_sigma, normalize=True):
    """
    Apply 3D Gaussian spatial filter to a scalar field.

    Parameters
    ----------
    grid : Grid3D
        Grid object.
    field : ndarray of shape (Nz, Ny, Nx)
        Input scalar field.
    sigma : float
        Filter width.
    cutoff_sigma : float
        Truncation parameter.
    normalize : bool, optional
        Default True.

    Returns
    -------
    field_bar : ndarray of shape (Nz, Ny, Nx)
        Filtered scalar field.
    """
    hz, hy, hx = _compute_half_widths(sigma, cutoff_sigma,
                                       grid.dx, grid.dy, grid.dz)

    K = _build_kernel_3d(sigma, grid.dx, grid.dy, grid.dz, hz, hy, hx)
    cell_vol = grid.cell_volume

    if normalize:
        mass = np.sum(K) * cell_vol
        if mass > 0:
            K = K / mass

    try:
        from scipy.ndimage import convolve
        return convolve(field, K * cell_vol, mode='constant', cval=0.0)
    except ImportError:
        pass

    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    field_bar = np.zeros_like(field)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                k0, k1 = max(0, k - hz), min(Nz - 1, k + hz)
                j0, j1 = max(0, j - hy), min(Ny - 1, j + hy)
                i0, i1 = max(0, i - hx), min(Nx - 1, i + hx)

                kk0, kk1 = k0 - (k - hz), k1 - (k - hz)
                jj0, jj1 = j0 - (j - hy), j1 - (j - hy)
                ii0, ii1 = i0 - (i - hx), i1 - (i - hx)

                K_local = K[kk0:kk1+1, jj0:jj1+1, ii0:ii1+1]
                patch = field[k0:k1+1, j0:j1+1, i0:i1+1]

                local_mass = np.sum(K_local) * cell_vol
                if normalize and local_mass > 0:
                    K_norm = K_local / local_mass
                else:
                    K_norm = K_local

                field_bar[k, j, i] = np.sum(K_norm * patch) * cell_vol

    return field_bar
