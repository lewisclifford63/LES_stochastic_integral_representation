import numpy as np
from les.grid import Grid2D


def initialize_particles_from_grid(
    grid: Grid2D,
    U0: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise particles from the Eulerian grid.

    Parameters
    ----------
    grid
        Full padded computational grid.
    U0
        Initial Eulerian velocity field of shape (Ny, Nx, 2).
    trusted_only
        If False, place one particle at every grid node in the full padded box.
        If True, place particles only in the trusted interior region.

    Returns
    -------
    positions : ndarray, shape (Np, 2)
        Particle positions.
    amplitudes : ndarray, shape (Np, 2)
        Particle-carried vector amplitudes, initially U0 at the particle location.
    weights : ndarray, shape (Np,)
        Quadrature weights, initially equal to the grid cell area.
    """
    if U0.shape != (grid.Ny, grid.Nx, 2):
        raise ValueError(
            f"Expected U0.shape == {(grid.Ny, grid.Nx, 2)}, got {U0.shape}."
        )

    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
        amplitudes = U0[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        amplitudes = U0.reshape(-1, 2).copy()

    weights = np.full(positions.shape[0], grid.cell_area, dtype=np.float64)
    return positions, amplitudes, weights


def initialize_particles_from_trusted_grid(
    grid: Grid2D,
    U0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: initialise particles only in the trusted interior.
    """
    return initialize_particles_from_grid(grid=grid, U0=U0, trusted_only=True)


def particle_count(positions: np.ndarray) -> int:
    """
    Number of particles.
    """
    return int(positions.shape[0])


def particles_in_trusted_region(positions: np.ndarray, grid: Grid2D) -> np.ndarray:
    """
    Boolean mask selecting particles that lie in the trusted interior region.
    """
    if grid.L_trust is None:
        return np.ones(positions.shape[0], dtype=bool)

    x = positions[:, 0]
    y = positions[:, 1]
    return (np.abs(x) <= grid.L_trust) & (np.abs(y) <= grid.L_trust)