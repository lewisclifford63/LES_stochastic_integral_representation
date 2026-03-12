import numpy as np

from les.grid import Grid2D


def taylor_green_velocity(grid: Grid2D) -> np.ndarray:
    """
    Smooth divergence-free 2D initial condition on the padded computational box.

    We scale the sinusoidal structure using the full computational half-width
    L_box so that the field is smooth across the whole padded domain.

        u(x, y) = ( sin(k y), -sin(k x) ),    k = pi / L_box
    """
    k = np.pi / grid.L_box

    U0 = grid.zeros_vector()
    U0[..., 0] = np.sin(k * grid.Y)
    U0[..., 1] = -np.sin(k * grid.X)
    return U0


def gaussian_vortex_velocity(
    grid: Grid2D,
    strength: float = 1.0,
    sigma: float = 0.4,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Smooth swirling field centered at `center`.

    This is a localized field, so it is naturally well suited to the padded-box
    approximation of R^2.
    """
    x0, y0 = center
    Xc = grid.X - x0
    Yc = grid.Y - y0

    r2 = Xc**2 + Yc**2
    factor = strength * np.exp(-r2 / (2.0 * sigma**2))

    U0 = grid.zeros_vector()
    U0[..., 0] = -Yc * factor
    U0[..., 1] = Xc * factor
    return U0


def two_gaussian_vortices_velocity(
    grid: Grid2D,
    strength_1: float = 1.0,
    strength_2: float = -1.0,
    sigma_1: float = 0.35,
    sigma_2: float = 0.35,
    center_1: tuple[float, float] = (-0.75, 0.0),
    center_2: tuple[float, float] = (0.75, 0.0),
) -> np.ndarray:
    """
    Superposition of two localized Gaussian vortices.

    Useful for testing interactions while remaining in a smooth, localized
    whole-space-style setting.
    """
    U1 = gaussian_vortex_velocity(
        grid=grid,
        strength=strength_1,
        sigma=sigma_1,
        center=center_1,
    )
    U2 = gaussian_vortex_velocity(
        grid=grid,
        strength=strength_2,
        sigma=sigma_2,
        center=center_2,
    )
    return U1 + U2


def compute_divergence(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Central-difference divergence of a vector field U[..., 0], U[..., 1].

    Uses periodic-style differences via np.roll. In the padded-box approach,
    this is acceptable so long as diagnostics are primarily interpreted on the
    trusted interior.
    """
    _validate_vector_field(U)

    ux = U[..., 0]
    uy = U[..., 1]

    dux_dx = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dx)
    duy_dy = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dy)

    return dux_dx + duy_dy


def compute_speed(U: np.ndarray) -> np.ndarray:
    """
    Pointwise speed |U|.
    """
    _validate_vector_field(U)
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


def _validate_vector_field(U: np.ndarray) -> None:
    if U.ndim != 3 or U.shape[-1] != 2:
        raise ValueError(
            f"Expected vector field with shape (Ny, Nx, 2), got {U.shape}."
        )