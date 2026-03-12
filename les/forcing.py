import numpy as np

from les.grid import Grid2D


def zero_forcing(grid: Grid2D, t: float) -> np.ndarray:
    """
    Zero forcing field.
    """
    _ = t
    return grid.zeros_vector()


def constant_forcing(
    grid: Grid2D,
    t: float,
    fx: float = 0.0,
    fy: float = 0.0,
) -> np.ndarray:
    """
    Constant forcing field.
    """
    _ = t

    F = grid.zeros_vector()
    F[..., 0] = fx
    F[..., 1] = fy
    return F


def time_dependent_constant_forcing(
    grid: Grid2D,
    t: float,
    amplitude_x: float = 0.0,
    amplitude_y: float = 0.0,
    frequency: float = 1.0,
) -> np.ndarray:
    """
    Spatially constant but time-dependent forcing.
    """
    factor = np.cos(frequency * t)

    F = grid.zeros_vector()
    F[..., 0] = amplitude_x * factor
    F[..., 1] = amplitude_y * factor
    return F


def gaussian_forcing(
    grid: Grid2D,
    t: float,
    amplitude: tuple[float, float] = (1.0, 0.0),
    sigma: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Localized Gaussian forcing field.
    """
    _ = t

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    x0, y0 = center
    ax, ay = amplitude

    Xc = grid.X - x0
    Yc = grid.Y - y0
    r2 = Xc * Xc + Yc * Yc

    factor = np.exp(-r2 / (2.0 * sigma * sigma))

    F = grid.zeros_vector()
    F[..., 0] = ax * factor
    F[..., 1] = ay * factor
    return F


def swirling_gaussian_forcing(
    grid: Grid2D,
    t: float,
    strength: float = 1.0,
    sigma: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Localized swirling Gaussian forcing field.
    """
    _ = t

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    x0, y0 = center

    Xc = grid.X - x0
    Yc = grid.Y - y0
    r2 = Xc * Xc + Yc * Yc

    factor = strength * np.exp(-r2 / (2.0 * sigma * sigma))

    F = grid.zeros_vector()
    F[..., 0] = -Yc * factor
    F[..., 1] = Xc * factor
    return F


def manufactured_forcing_from_velocity(
    grid: Grid2D,
    t: float,
    velocity_function,
    dt: float = 1.0e-6,
) -> np.ndarray:
    """
    Placeholder helper for future manufactured-solution tests.

    Currently returns velocity_function(grid, t). This keeps the interface
    available without yet enforcing a full manufactured forcing formula.
    """
    _ = dt
    return velocity_function(grid, t)


def make_zero_forcing():
    """
    Return a callable forcing function F(grid, t).
    """
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return zero_forcing(grid, t)

    return forcing


def make_constant_forcing(fx: float = 0.0, fy: float = 0.0):
    """
    Return a callable constant forcing function F(grid, t).
    """
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return constant_forcing(grid, t, fx=fx, fy=fy)

    return forcing


def make_time_dependent_constant_forcing(
    amplitude_x: float = 0.0,
    amplitude_y: float = 0.0,
    frequency: float = 1.0,
):
    """
    Return a callable time-dependent constant forcing function F(grid, t).
    """
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return time_dependent_constant_forcing(
            grid,
            t,
            amplitude_x=amplitude_x,
            amplitude_y=amplitude_y,
            frequency=frequency,
        )

    return forcing


def make_gaussian_forcing(
    amplitude: tuple[float, float] = (1.0, 0.0),
    sigma: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
):
    """
    Return a callable localized Gaussian forcing function F(grid, t).
    """
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return gaussian_forcing(
            grid,
            t,
            amplitude=amplitude,
            sigma=sigma,
            center=center,
        )

    return forcing


def make_swirling_gaussian_forcing(
    strength: float = 1.0,
    sigma: float = 0.5,
    center: tuple[float, float] = (0.0, 0.0),
):
    """
    Return a callable swirling Gaussian forcing function F(grid, t).
    """
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return swirling_gaussian_forcing(
            grid,
            t,
            strength=strength,
            sigma=sigma,
            center=center,
        )

    return forcing