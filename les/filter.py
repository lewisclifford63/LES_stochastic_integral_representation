import numpy as np


def gaussian_kernel_2d(dx: np.ndarray, dy: np.ndarray, sigma: float) -> np.ndarray:
    """
    2D Gaussian filter kernel:
        chi(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))

    Parameters
    ----------
    dx, dy
        Arrays of relative offsets in x and y.
    sigma
        Filter width parameter.

    Returns
    -------
    np.ndarray
        Gaussian kernel evaluated at the given offsets.
    """
    r2 = dx * dx + dy * dy
    prefactor = 1.0 / (2.0 * np.pi * sigma * sigma)
    return prefactor * np.exp(-0.5 * r2 / (sigma * sigma))


def cutoff_radius(sigma: float, cutoff_sigma: float) -> float:
    """
    Physical cutoff radius used to truncate the Gaussian for efficiency.

    The truncated kernel is supported only on the ball
        sqrt(x^2 + y^2) <= cutoff_sigma * sigma.
    """
    return cutoff_sigma * sigma


def stencil_half_width(h: float, sigma: float, cutoff_sigma: float) -> int:
    """
    Number of grid cells to include on either side of the central point.

    This gives a square stencil large enough to contain the truncated
    Gaussian support.
    """
    radius = cutoff_radius(sigma, cutoff_sigma)
    return int(np.ceil(radius / h))


def stencil_half_width_xy(
    dx: float,
    dy: float,
    sigma: float,
    cutoff_sigma: float,
) -> tuple[int, int]:
    """
    Direction-wise stencil half-widths for anisotropic grid spacings.

    Useful on rectangular grids where dx != dy.
    """
    radius = cutoff_radius(sigma, cutoff_sigma)
    hx = int(np.ceil(radius / dx))
    hy = int(np.ceil(radius / dy))
    return hx, hy


def make_local_kernel(
    dx: float,
    dy: float,
    sigma: float,
    cutoff_sigma: float,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a truncated discrete Gaussian kernel on a local Cartesian stencil.

    Parameters
    ----------
    dx, dy
        Grid spacings.
    sigma
        Gaussian width.
    cutoff_sigma
        Truncation radius in units of sigma.
    normalize
        If True, normalize the discrete kernel so that its weighted discrete
        integral is 1, i.e.
            sum K_ij * dx * dy = 1.

    Returns
    -------
    offsets_x : np.ndarray
        1D array of x-offsets used in the stencil.
    offsets_y : np.ndarray
        1D array of y-offsets used in the stencil.
    kernel : np.ndarray
        2D truncated kernel on the stencil.
    """
    hx, hy = stencil_half_width_xy(dx=dx, dy=dy, sigma=sigma, cutoff_sigma=cutoff_sigma)

    offsets_x = np.arange(-hx, hx + 1, dtype=np.float64) * dx
    offsets_y = np.arange(-hy, hy + 1, dtype=np.float64) * dy

    DX, DY = np.meshgrid(offsets_x, offsets_y, indexing="xy")
    K = gaussian_kernel_2d(DX, DY, sigma=sigma)

    radius = cutoff_radius(sigma, cutoff_sigma)
    mask = (DX * DX + DY * DY) <= radius * radius
    K = np.where(mask, K, 0.0)

    if normalize:
        mass = np.sum(K) * dx * dy
        if mass <= 0.0:
            raise ValueError("Discrete Gaussian kernel has non-positive mass.")
        K = K / mass

    return offsets_x, offsets_y, K