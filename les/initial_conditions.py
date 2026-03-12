import numpy as np
from les.grid import Grid2D


def taylor_green_velocity(grid: Grid2D) -> np.ndarray:
    """
    Smooth divergence-free 2D initial condition.

    On the padded box, we scale the sinusoidal structure using the full
    computational half-width L_box, not the trusted half-width. This keeps the
    field smooth across the whole numerical domain.

    u(x, y) = ( sin(k y), -sin(k x) ),    k = pi / L_box
    """
    k = np.pi / grid.L_box

    u = np.sin(k * grid.Y)
    v = -np.sin(k * grid.X)

    U0 = np.zeros((grid.Ny, grid.Nx, 2), dtype=np.float64)
    U0[..., 0] = u
    U0[..., 1] = v
    return U0


def gaussian_vortex_velocity(
    grid: Grid2D,
    strength: float = 1.0,
    sigma: float = 0.4,
) -> np.ndarray:
    """
    Smooth swirling field centered at the origin.

    This decays away from the origin, so it is naturally compatible with the
    padded-box approach and is often a better unbounded-domain test field than
    a globally oscillatory one.
    """
    r2 = grid.X**2 + grid.Y**2
    factor = strength * np.exp(-r2 / (2.0 * sigma**2))

    U0 = np.zeros((grid.Ny, grid.Nx, 2), dtype=np.float64)
    U0[..., 0] = -grid.Y * factor
    U0[..., 1] = grid.X * factor
    return U0


def compact_gaussian_vortex_velocity(
    grid: Grid2D,
    strength: float = 1.0,
    sigma: float = 0.4,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Shifted Gaussian vortex centered at `center`.

    Useful if you later want multiple localized structures while still trusting
    only the interior region.
    """
    x0, y0 = center
    Xc = grid.X - x0
    Yc = grid.Y - y0

    r2 = Xc**2 + Yc**2
    factor = strength * np.exp(-r2 / (2.0 * sigma**2))

    U0 = np.zeros((grid.Ny, grid.Nx, 2), dtype=np.float64)
    U0[..., 0] = -Yc * factor
    U0[..., 1] = Xc * factor
    return U0


def compute_divergence(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Central-difference divergence of a vector field U[..., 0], U[..., 1].

    Uses periodic-style differences via np.roll. In the padded-box approach,
    this is mainly acceptable because we only trust the interior region away
    from the artificial box boundary.
    """
    ux = U[..., 0]
    uy = U[..., 1]

    dux_dx = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dx)
    duy_dy = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dy)

    return dux_dx + duy_dy


def compute_speed(U: np.ndarray) -> np.ndarray:
    """
    Pointwise speed |U|.
    """
    return np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)


def trusted_max_abs_divergence(U: np.ndarray, grid: Grid2D) -> float:
    """
    Maximum absolute divergence restricted to the trusted interior.
    """
    div = compute_divergence(U, dx=grid.dx, dy=grid.dy)
    div_trusted = grid.restrict_to_trusted(div)
    return float(np.max(np.abs(div_trusted)))


def trusted_mean_abs_divergence(U: np.ndarray, grid: Grid2D) -> float:
    """
    Mean absolute divergence restricted to the trusted interior.
    """
    div = compute_divergence(U, dx=grid.dx, dy=grid.dy)
    div_trusted = grid.restrict_to_trusted(div)
    return float(np.mean(np.abs(div_trusted)))


def trusted_speed(U: np.ndarray, grid: Grid2D) -> np.ndarray:
    """
    Speed restricted to the trusted interior.
    """
    return grid.restrict_to_trusted(compute_speed(U))