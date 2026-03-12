import numpy as np


def gaussian_kernel_2d(dx: np.ndarray, dy: np.ndarray, sigma: float) -> np.ndarray:
    """
    2D Gaussian kernel.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    r2 = dx * dx + dy * dy
    prefactor = 1.0 / (2.0 * np.pi * sigma * sigma)
    return prefactor * np.exp(-0.5 * r2 / (sigma * sigma))


def cutoff_radius(sigma: float, cutoff_sigma: float) -> float:
    """
    Physical cutoff radius for the truncated Gaussian.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if cutoff_sigma <= 0.0:
        raise ValueError("cutoff_sigma must be positive.")

    return cutoff_sigma * sigma


def stencil_half_width(h: float, sigma: float, cutoff_sigma: float) -> int:
    """
    Stencil half-width for a grid with spacing h.
    """
    if h <= 0.0:
        raise ValueError("h must be positive.")

    radius = cutoff_radius(sigma, cutoff_sigma)
    return int(np.ceil(radius / h))


def stencil_half_width_xy(
    dx: float,
    dy: float,
    sigma: float,
    cutoff_sigma: float,
) -> tuple[int, int]:
    """
    Stencil half-widths in x and y.
    """
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive.")

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
    Build a truncated discrete Gaussian kernel on a local stencil.

    Returns:
        offsets_x, offsets_y, kernel
    """
    hx, hy = stencil_half_width_xy(
        dx=dx,
        dy=dy,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    offsets_x = np.arange(-hx, hx + 1, dtype=np.float64) * dx
    offsets_y = np.arange(-hy, hy + 1, dtype=np.float64) * dy

    DX, DY = np.meshgrid(offsets_x, offsets_y, indexing="xy")
    K = gaussian_kernel_2d(DX, DY, sigma=sigma)

    radius = cutoff_radius(sigma, cutoff_sigma)
    mask = DX * DX + DY * DY <= radius * radius
    K = np.where(mask, K, 0.0)

    if normalize:
        mass = np.sum(K) * dx * dy
        if mass <= 0.0:
            raise ValueError("Kernel mass must be positive.")
        K = K / mass

    return offsets_x, offsets_y, K


def truncate_kernel_by_radius(
    DX: np.ndarray,
    DY: np.ndarray,
    K: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Zero out kernel values outside the given radius.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    mask = DX * DX + DY * DY <= radius * radius
    return np.where(mask, K, 0.0)


def normalize_discrete_kernel(
    K: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Normalize a discrete kernel so that sum(K) * dx * dy = 1.
    """
    mass = np.sum(K) * dx * dy
    if mass <= 0.0:
        raise ValueError("Kernel mass must be positive.")

    return K / mass