"""
Particle module for the 3D random LES method.

Implements the Milstein SDE integrator (Qian eq. 55 in 3D) and the per-particle
accumulated-history data structures required by the full integral
representation (Qian eq. 42/56 for flows in R^3).

Each particle carries:
    - its current position  Y^eta_{t_k}  (3D vector)
    - the initial velocity  u_0(eta)  sampled at its birth grid node (3D vector)
    - the running integral  sum_{j=1}^{k} dt * G(Y^eta_{t_{j-1}}, t_{j-1})  (3D vector)

These are stored as plain numpy arrays whose rows are indexed identically
so that row p always refers to the same physical particle.
"""

import numpy as np

from les.grid import CartesianGrid3D


# ------------------------------------------------------------------ #
#  Initialisation
# ------------------------------------------------------------------ #

def initialize_particles_from_grid(
    grid: CartesianGrid3D,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize particle positions from grid nodes.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    trusted_only : bool, optional
        If True, initialize only from trusted region nodes.

    Returns
    -------
    positions : ndarray, shape (Np, 3)
        Particle positions [x_i, y_i, z_i].
    weights   : ndarray, shape (Np,)
        Each weight equals the cell volume  s^d  (Qian's s^3 in 3-D).
    """
    if trusted_only:
        positions = grid.coordinates_as_particles()
        # Apply trust mask: keep only trusted nodes
        Np = grid.Nz * grid.Ny * grid.Nx
        mask = np.zeros(Np, dtype=bool)
        idx = 0
        for k in range(grid.Nz):
            for j in range(grid.Ny):
                for i in range(grid.Nx):
                    if grid.trust_mask[k, j, i]:
                        mask[idx] = True
                    idx += 1
        positions = positions[mask]
    else:
        positions = grid.coordinates_as_particles()

    weights = np.full(positions.shape[0], grid.cell_volume, dtype=np.float64)
    return positions, weights


def initialize_particles_with_field_values(
    grid: CartesianGrid3D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize particles from grid nodes and sample a vector field there.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    field : ndarray, shape (Nz, Ny, Nx, 3)
        Vector field with 3 components [u, v, w].
    trusted_only : bool, optional
        If True, initialize only from trusted region nodes.

    Returns
    -------
    positions : ndarray, shape (Np, 3)
        Particle positions [x_i, y_i, z_i].
    values    : ndarray, shape (Np, 3)
        Sampled field values at particles.
    weights   : ndarray, shape (Np,)
        Cell volume weights.
    """
    _validate_vector_field(field, "field")

    if trusted_only:
        positions = grid.coordinates_as_particles()
        # Flatten field to (Np, 3)
        field_flat = field.reshape(-1, 3).copy()
        # Apply trust mask
        Np = grid.Nz * grid.Ny * grid.Nx
        mask = np.zeros(Np, dtype=bool)
        idx = 0
        for k in range(grid.Nz):
            for j in range(grid.Ny):
                for i in range(grid.Nx):
                    if grid.trust_mask[k, j, i]:
                        mask[idx] = True
                    idx += 1
        positions = positions[mask]
        values = field_flat[mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 3).copy()

    weights = np.full(positions.shape[0], grid.cell_volume, dtype=np.float64)
    return positions, values, weights


def initialize_particle_history(
    grid: CartesianGrid3D,
    U0: np.ndarray,
    trusted_only: bool = False,
) -> dict[str, np.ndarray]:
    """
    Build the full particle state needed for Qian's accumulated-history
    representation (eq. 42/56 for flows in R^d).

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    U0 : ndarray, shape (Nz, Ny, Nx, 3)
        Initial velocity field with 3 components [u, v, w].
    trusted_only : bool, optional
        If True, initialize only from trusted region nodes.

    Returns
    -------
    dict with keys:
        positions       (Np, 3)   current particle positions Y^eta_{t_k}
        weights         (Np,)     quadrature weight  s^d  per particle
        u0_carried      (Np, 3)   initial velocity u_0(eta) frozen at birth
        accumulated_G   (Np, 3)   running sum  sum_{j=1}^{k} dt * G(Y, t_{j-1})
    """
    positions, weights = initialize_particles_from_grid(
        grid=grid,
        trusted_only=trusted_only,
    )

    _validate_vector_field(U0, "U0")

    if trusted_only:
        # Flatten U0 and apply trust mask
        U0_flat = U0.reshape(-1, 3).copy()
        Np = grid.Nz * grid.Ny * grid.Nx
        mask = np.zeros(Np, dtype=bool)
        idx = 0
        for k in range(grid.Nz):
            for j in range(grid.Ny):
                for i in range(grid.Nx):
                    if grid.trust_mask[k, j, i]:
                        mask[idx] = True
                    idx += 1
        u0_carried = U0_flat[mask].copy()
    else:
        u0_carried = U0.reshape(-1, 3).copy()

    accumulated_G = np.zeros_like(u0_carried)

    return {
        "positions": positions,
        "weights": weights,
        "u0_carried": u0_carried,
        "accumulated_G": accumulated_G,
    }


# ------------------------------------------------------------------ #
#  Queries
# ------------------------------------------------------------------ #

def particle_count(positions: np.ndarray) -> int:
    """Return the number of particles."""
    _validate_positions(positions)
    return int(positions.shape[0])


def particles_in_trusted_region(
    positions: np.ndarray,
    grid: CartesianGrid3D,
) -> np.ndarray:
    """Boolean mask for particles in the trusted region."""
    _validate_positions(positions)
    return grid.points_in_trusted_region(positions)


def particles_in_padded_box(
    positions: np.ndarray,
    grid: CartesianGrid3D,
) -> np.ndarray:
    """Boolean mask for particles in the padded computational box."""
    _validate_positions(positions)
    return grid.points_in_box(positions)


# ------------------------------------------------------------------ #
#  SDE integrators
# ------------------------------------------------------------------ #

def advance_particles_euler_maruyama(
    positions: np.ndarray,
    drift: np.ndarray,
    dt: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One Euler-Maruyama step  (Qian eq. 54):

        Y_{k+1} = Y_k  +  dt * U(Y_k, t_k)  +  sqrt(2 nu dt) N(0, I_3)

    Parameters
    ----------
    positions : ndarray, shape (Np, 3)
        Current particle positions.
    drift : ndarray, shape (Np, 3)
        Velocity  U(Y_k, t_k)  interpolated at each particle.
    dt : float
        Time-step size  delta.
    nu : float
        Kinematic viscosity.
    rng : Generator
        NumPy random generator.

    Returns
    -------
    new_positions : ndarray, shape (Np, 3)

    Notes
    -----
    Retained for comparison / testing against Milstein integrator.
    """
    _validate_positions(positions)
    _validate_vector_field_flat(drift, "drift")

    if drift.shape != positions.shape:
        raise ValueError("drift must have the same shape as positions.")
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")
    if nu < 0.0:
        raise ValueError("nu must be non-negative.")

    noise = np.sqrt(2.0 * nu * dt) * rng.standard_normal(size=positions.shape)
    return positions + dt * drift + noise


def advance_particles_milstein(
    positions: np.ndarray,
    drift: np.ndarray,
    drift_jacobian: np.ndarray,
    dt: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One Milstein step  (Qian eq. 55 in 3D):

        Y_{k+1} = Y_k
                 + dt * U(Y_k)
                 + sqrt(2 nu) * dB
                 + 0.5 * (nabla U)(Y_k) . U(Y_k)  * [(dB)^2 - dt]

    Parameters
    ----------
    positions : ndarray, shape (Np, 3)
        Current particle positions.
    drift : ndarray, shape (Np, 3)
        Velocity  U(Y_k, t_k)  interpolated at each particle.
    drift_jacobian : ndarray, shape (Np, 3, 3)
        Velocity gradient  (nabla U)(Y_k, t_k)  interpolated at each particle.
        Convention:  drift_jacobian[p, i, j] = dU_i / dx_j.
    dt : float
        Time-step size  delta.
    nu : float
        Kinematic viscosity.
    rng : Generator
        NumPy random generator.

    Returns
    -------
    new_positions : ndarray, shape (Np, 3)

    Notes
    -----
    The diffusion coefficient sigma = sqrt(2 nu) * I_3 is constant, so the
    classical Milstein correction from the noise-times-noise term vanishes.
    Qian's eq. (55) in 3D includes a drift-derivative correction:

        correction_i = 0.5 * sum_j  (dU_i / dx_j) * (dB_j)^2 - dt)

    This is implemented using einsum for efficiency:
        correction[p, i] = (sigma/2) * sum_j drift_jacobian[p, i, j] * (dW_j^2 - dt)

    where sigma = sqrt(2 nu) and dW = dB / sigma is the standard 3D Brownian
    increment.
    """
    _validate_positions(positions)
    _validate_vector_field_flat(drift, "drift")

    Np = positions.shape[0]
    if drift.shape != (Np, 3):
        raise ValueError("drift must have shape (Np, 3).")
    if drift_jacobian.shape != (Np, 3, 3):
        raise ValueError("drift_jacobian must have shape (Np, 3, 3).")
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")
    if nu < 0.0:
        raise ValueError("nu must be non-negative.")

    # Brownian increment  dB ~ N(0, 2*nu*dt * I_3)
    dB = np.sqrt(2.0 * nu * dt) * rng.standard_normal(size=(Np, 3))

    # Euler-Maruyama part
    new_positions = positions + dt * drift + dB

    # Milstein correction (Qian eq. 55, last line)
    #
    # The SDE is  dY_i = U_i dt + sqrt(2 nu) dW_i  with ADDITIVE (constant)
    # noise coefficient sigma = sqrt(2 nu).  For additive noise the classical
    # Milstein noise-times-noise term vanishes.  The leading stochastic
    # correction beyond Euler-Maruyama comes from the 1.5-order Ito-Taylor
    # expansion of the drift:
    #
    #   correction_i = (sigma / 2) * sum_j (dU_i/dx_j) * (dW_j^2 - dt)
    #
    # where sigma = sqrt(2 nu) is the *diffusion coefficient* and
    # dW = dB / sqrt(2 nu)  is the standard BM increment  ~N(0, dt I_3).
    #
    # Note: using the drift U_j in place of sigma here would overestimate
    # the correction by a factor of |U| / sqrt(2 nu) >> 1 for large
    # velocities, causing numerical blow-up.
    if nu > 0.0:
        sigma = np.sqrt(2.0 * nu)
        dW = dB / sigma                       # shape (Np, 3), standard BM increment
        dW2_minus_dt = dW * dW - dt           # shape (Np, 3)

        # correction[p, i] = (sigma/2) * sum_j drift_jacobian[p, i, j] * (dW_j^2 - dt)
        correction = (sigma / 2.0) * np.einsum("pij,pj->pi", drift_jacobian, dW2_minus_dt)
        new_positions += correction

    return new_positions


# ------------------------------------------------------------------ #
#  Clipping / box utilities
# ------------------------------------------------------------------ #

def clip_positions_to_box(
    positions: np.ndarray,
    grid: CartesianGrid3D,
) -> np.ndarray:
    """
    Clip particle positions into the padded box.

    Parameters
    ----------
    positions : ndarray, shape (Np, 3)
        Particle positions.
    grid : CartesianGrid3D
        3D grid with box bounds.

    Returns
    -------
    clipped : ndarray, shape (Np, 3)
        Clipped positions, all components in [-L_box, L_box).
    """
    _validate_positions(positions)

    clipped = positions.copy()
    eps_x = 1.0e-12 * max(1.0, grid.dx)
    eps_y = 1.0e-12 * max(1.0, grid.dy)
    eps_z = 1.0e-12 * max(1.0, grid.dz)

    clipped[:, 0] = np.clip(clipped[:, 0], -grid.L_box,   grid.L_box   - eps_x)
    clipped[:, 1] = np.clip(clipped[:, 1], -grid.L_box,   grid.L_box   - eps_y)
    clipped[:, 2] = np.clip(clipped[:, 2], -grid.L_box_z, grid.L_box_z - eps_z)

    return clipped


def wrap_positions_periodic(
    positions: np.ndarray,
    grid: CartesianGrid3D,
) -> np.ndarray:
    """
    Wrap particle positions periodically into [-L_box, L_box).

    Consistent with the periodic FFT pressure solver: a particle leaving
    one side re-enters from the opposite side, preserving its accumulated_G
    history and maintaining uniform particle coverage.

    Parameters
    ----------
    positions : ndarray, shape (Np, 3)
        Particle positions (may be outside [-L_box, L_box)).
    grid : CartesianGrid3D
        3D grid with box bounds.

    Returns
    -------
    wrapped : ndarray, shape (Np, 3)
        Wrapped positions, all components in [-L_box, L_box).
    """
    _validate_positions(positions)

    # Use per-axis box half-widths (supports non-cubic grids).
    wrapped = positions.copy()
    for ax, L in enumerate([grid.L_box, grid.L_box, grid.L_box_z]):
        period = 2.0 * L
        wrapped[:, ax] = (positions[:, ax] + L) % period - L
    return wrapped


def count_particles_outside_box(
    positions: np.ndarray,
    grid: CartesianGrid3D,
) -> int:
    """Count particles lying outside the padded box."""
    _validate_positions(positions)
    inside = grid.points_in_box(positions)
    return int(np.sum(~inside))


def fraction_particles_outside_box(
    positions: np.ndarray,
    grid: CartesianGrid3D,
) -> float:
    """Fraction of particles lying outside the padded box."""
    _validate_positions(positions)

    if positions.shape[0] == 0:
        return 0.0

    return count_particles_outside_box(positions, grid) / positions.shape[0]


def replicate_particles(
    positions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    copies: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicate particles and divide weights evenly across copies.

    Parameters
    ----------
    positions : ndarray, shape (Np, 3)
        Particle positions.
    values : ndarray, shape (Np, 3)
        Particle field values.
    weights : ndarray, shape (Np,)
        Particle weights.
    copies : int
        Number of copies of each particle.

    Returns
    -------
    positions_rep : ndarray, shape (copies * Np, 3)
    values_rep : ndarray, shape (copies * Np, 3)
    weights_rep : ndarray, shape (copies * Np,)
    """
    _validate_positions(positions)
    _validate_vector_field_flat(values, "values")
    _validate_weights(weights, positions.shape[0])

    if values.shape != positions.shape:
        raise ValueError("values must have the same shape as positions.")

    if copies < 1:
        raise ValueError("copies must be at least 1.")

    if copies == 1:
        return positions.copy(), values.copy(), weights.copy()

    positions_rep = np.repeat(positions, copies, axis=0)
    values_rep = np.repeat(values, copies, axis=0)
    weights_rep = np.repeat(weights / copies, copies, axis=0)

    return positions_rep, values_rep, weights_rep


# ------------------------------------------------------------------ #
#  Validation helpers
# ------------------------------------------------------------------ #

def _validate_positions(positions: np.ndarray) -> None:
    """Validate that positions have shape (Np, 3)."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected positions to have shape (Np, 3).")


def _validate_vector_field(field: np.ndarray, name: str) -> None:
    """Validate that field has shape (Nz, Ny, Nx, 3)."""
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f"Expected {name} to have shape (Nz, Ny, Nx, 3).")


def _validate_vector_field_flat(field: np.ndarray, name: str) -> None:
    """Validate that field has shape (Np, 3)."""
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError(f"Expected {name} to have shape (Np, 3).")


def _validate_weights(weights: np.ndarray, n_particles: int) -> None:
    """Validate that weights have shape (Np,)."""
    if weights.ndim != 1 or weights.shape[0] != n_particles:
        raise ValueError("Expected weights to have shape (Np,).")
