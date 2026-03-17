"""
3D Forcing Functions Module
============================
Provides time-dependent and spatially-varying force fields for 3D Large Eddy
Simulation via stochastic integral representations (Qian et al., 2025).

Convention:
- All forcing functions return vector fields of shape (Nz, Ny, Nx, 3).
- Force components: (f_x, f_y, f_z) along the last dimension.
- Grid coordinates: X, Y, Z from meshgrid(z, y, x, indexing="ij").
"""

import numpy as np
from les.grid import Grid3D


def zero_forcing(grid: Grid3D, t: float = 0.0) -> np.ndarray:
    """
    Zero forcing field (unforced dynamics).

    Parameters
    ----------
    grid : Grid3D
        Grid instance with attributes X, Y, Z of shape (Nz, Ny, Nx).
    t : float
        Current time (unused, kept for consistent interface).

    Returns
    -------
    forcing : np.ndarray
        Zero vector field of shape (Nz, Ny, Nx, 3).
    """
    return grid.zeros_vector()


def constant_forcing(grid: Grid3D,
                    t: float = 0.0,
                    fx: float = 0.0,
                    fy: float = 0.0,
                    fz: float = 0.0) -> np.ndarray:
    """
    Uniform constant force field.

    f(x, y, z, t) = (f_x, f_y, f_z) everywhere.

    Parameters
    ----------
    grid : Grid3D
        Grid instance.
    t : float
        Current time (unused, kept for consistent interface).
    fx, fy, fz : float
        Force components (constant in space and time).

    Returns
    -------
    forcing : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    """
    forcing = grid.zeros_vector()
    forcing[..., 0] = fx
    forcing[..., 1] = fy
    forcing[..., 2] = fz
    return forcing


def time_dependent_constant_forcing(grid: Grid3D,
                                   t: float,
                                   ax: float = 1.0,
                                   ay: float = 1.0,
                                   az: float = 0.0,
                                   omega_x: float = 1.0,
                                   omega_y: float = 1.0,
                                   omega_z: float = 1.0,
                                   phase_x: float = 0.0,
                                   phase_y: float = 0.0,
                                   phase_z: float = 0.0) -> np.ndarray:
    """
    Spatially uniform force with harmonic time-dependence.

    f_i(t) = a_i * sin(ω_i * t + φ_i) for i ∈ {x, y, z}

    Parameters
    ----------
    grid : Grid3D
        Grid instance.
    t : float
        Current time.
    ax, ay, az : float
        Force amplitudes.
    omega_x, omega_y, omega_z : float
        Angular frequencies.
    phase_x, phase_y, phase_z : float
        Phase shifts.

    Returns
    -------
    forcing : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    """
    forcing = grid.zeros_vector()
    forcing[..., 0] = ax * np.sin(omega_x * t + phase_x)
    forcing[..., 1] = ay * np.sin(omega_y * t + phase_y)
    forcing[..., 2] = az * np.sin(omega_z * t + phase_z)
    return forcing


def gaussian_forcing(grid: Grid3D,
                    t: float = 0.0,
                    amplitude: tuple = (1.0, 1.0, 0.0),
                    sigma: float = 1.57,
                    center: tuple = (0.0, 0.0, 0.0)) -> np.ndarray:
    """
    3D localized Gaussian force field.

    Applies force components (a_x, a_y, a_z) modulated by Gaussian envelope
    centered at (x0, y0, z0):

        f_i(x, y, z) = a_i * exp(-r² / (2*σ²))

    where r² = (X - x0)² + (Y - y0)² + (Z - z0)²

    Parameters
    ----------
    grid : Grid3D
        Grid instance.
    t : float
        Current time (unused, kept for consistent interface).
    amplitude : tuple
        Force amplitudes (a_x, a_y, a_z).
    sigma : float
        Gaussian width.
    center : tuple
        Gaussian center (x0, y0, z0).

    Returns
    -------
    forcing : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    """
    ax, ay, az = amplitude
    x0, y0, z0 = center

    # Squared distance from center
    r_squared = ((grid.X - x0)**2 +
                 (grid.Y - y0)**2 +
                 (grid.Z - z0)**2)

    # Gaussian envelope
    envelope = np.exp(-r_squared / (2.0 * sigma**2))

    # Apply to each component
    forcing = grid.zeros_vector()
    forcing[..., 0] = ax * envelope
    forcing[..., 1] = ay * envelope
    forcing[..., 2] = az * envelope

    return forcing


def swirling_gaussian_forcing(grid: Grid3D,
                              t: float = 0.0,
                              strength: float = 52.5,
                              sigma: float = 1.57,
                              center: tuple = (0.0, 0.0, 0.0)) -> np.ndarray:
    """
    3D localized Gaussian swirling force field.

    Models a localized forcing vortex with swirl pattern in the xy-plane:

        r² = (X - x0)² + (Y - y0)² + (Z - z0)²
        factor = strength * exp(-r² / (2*σ²))

        f_x = -(Y - y0) * factor
        f_y =  (X - x0) * factor
        f_z = 0

    This pattern induces rotational acceleration in the xy-plane.

    Parameters
    ----------
    grid : Grid3D
        Grid instance.
    t : float
        Current time (unused, kept for consistent interface).
    strength : float
        Force strength (vortex-like parameter). Default 52.5 (Qian scale).
    sigma : float
        Gaussian width. Default 1.57 (Qian scale).
    center : tuple
        Force center (x0, y0, z0).

    Returns
    -------
    forcing : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    """
    x0, y0, z0 = center

    # Squared distance
    r_squared = ((grid.X - x0)**2 +
                 (grid.Y - y0)**2 +
                 (grid.Z - z0)**2)

    # Gaussian envelope
    factor = strength * np.exp(-r_squared / (2.0 * sigma**2))

    # Allocate forcing
    forcing = grid.zeros_vector()

    # Swirl pattern
    forcing[..., 0] = -(grid.Y - y0) * factor
    forcing[..., 1] = (grid.X - x0) * factor
    forcing[..., 2] = 0.0

    return forcing


def make_zero_forcing() -> callable:
    """
    Factory function for zero forcing.

    Returns
    -------
    forcing_fn : callable
        Function with signature forcing_fn(grid, t) → np.ndarray.
    """
    def forcing_fn(grid: Grid3D, t: float) -> np.ndarray:
        return zero_forcing(grid, t)
    return forcing_fn


def make_constant_forcing(fx: float = 0.0,
                         fy: float = 0.0,
                         fz: float = 0.0) -> callable:
    """
    Factory function for constant forcing.

    Parameters
    ----------
    fx, fy, fz : float
        Force components.

    Returns
    -------
    forcing_fn : callable
        Function with signature forcing_fn(grid, t) → np.ndarray.
    """
    def forcing_fn(grid: Grid3D, t: float) -> np.ndarray:
        return constant_forcing(grid, t, fx, fy, fz)
    return forcing_fn


def make_time_dependent_constant_forcing(ax: float = 1.0,
                                        ay: float = 1.0,
                                        az: float = 0.0,
                                        omega_x: float = 1.0,
                                        omega_y: float = 1.0,
                                        omega_z: float = 1.0,
                                        phase_x: float = 0.0,
                                        phase_y: float = 0.0,
                                        phase_z: float = 0.0) -> callable:
    """
    Factory function for time-dependent constant forcing.

    Parameters
    ----------
    ax, ay, az : float
        Amplitudes.
    omega_x, omega_y, omega_z : float
        Frequencies.
    phase_x, phase_y, phase_z : float
        Phases.

    Returns
    -------
    forcing_fn : callable
        Function with signature forcing_fn(grid, t) → np.ndarray.
    """
    def forcing_fn(grid: Grid3D, t: float) -> np.ndarray:
        return time_dependent_constant_forcing(
            grid, t, ax, ay, az, omega_x, omega_y, omega_z,
            phase_x, phase_y, phase_z
        )
    return forcing_fn


def make_gaussian_forcing(amplitude: tuple = (1.0, 1.0, 0.0),
                         sigma: float = 1.57,
                         center: tuple = (0.0, 0.0, 0.0)) -> callable:
    """
    Factory function for Gaussian forcing.

    Parameters
    ----------
    amplitude : tuple
        Force amplitudes (a_x, a_y, a_z).
    sigma : float
        Gaussian width.
    center : tuple
        Gaussian center.

    Returns
    -------
    forcing_fn : callable
        Function with signature forcing_fn(grid, t) → np.ndarray.
    """
    def forcing_fn(grid: Grid3D, t: float) -> np.ndarray:
        return gaussian_forcing(grid, t, amplitude, sigma, center)
    return forcing_fn


def make_swirling_gaussian_forcing(strength: float = 52.5,
                                  sigma: float = 1.57,
                                  center: tuple = (0.0, 0.0, 0.0)) -> callable:
    """
    Factory function for swirling Gaussian forcing.

    Parameters
    ----------
    strength : float
        Force strength.
    sigma : float
        Gaussian width.
    center : tuple
        Force center.

    Returns
    -------
    forcing_fn : callable
        Function with signature forcing_fn(grid, t) → np.ndarray.
    """
    def forcing_fn(grid: Grid3D, t: float) -> np.ndarray:
        return swirling_gaussian_forcing(grid, t, strength, sigma, center)
    return forcing_fn
