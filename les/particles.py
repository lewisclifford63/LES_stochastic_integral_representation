"""
Particle module for the random LES method.

Implements the Milstein SDE integrator (Qian eq. 55) and the per-particle
accumulated-history data structures required by the full integral
representation (Qian eq. 42/56).

Each particle carries:
    - its current position  Y^eta_{t_k}
    - the initial velocity  u_0(eta) sampled at its birth grid node
    - the running integral  sum_{j=1}^{k} dt * G(Y^eta_{t_{j-1}}, t_{j-1})

These are stored as plain numpy arrays whose rows are indexed identically
so that row p always refers to the same physical particle.
"""

import numpy as np

from les.grid import Grid2D


# ------------------------------------------------------------------ #
#  Initialisation
# ------------------------------------------------------------------ #

def initialize_particles_from_grid(
    grid: Grid2D,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize particle positions from grid nodes.

    Returns
    -------
    positions : ndarray, shape (Np, 2)
    weights   : ndarray, shape (Np,)
        Each weight equals the cell area  s^d  (Qian's s^2 in 2-D).
    """
    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
    else:
        positions = grid.coordinates_as_particles()

    weights = np.full(positions.shape[0], grid.cell_area, dtype=np.float64)
    return positions, weights


def initialize_particles_with_field_values(
    grid: Grid2D,
    field: np.ndarray,
    trusted_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize particles from grid nodes and sample a vector field there.

    Returns
    -------
    positions : ndarray, shape (Np, 2)
    values    : ndarray, shape (Np, 2)
    weights   : ndarray, shape (Np,)
    """
    _validate_vector_field(field, "field")

    if trusted_only:
        positions = grid.trusted_coordinates_as_particles()
        values = field[grid.trust_mask].copy()
    else:
        positions = grid.coordinates_as_particles()
        values = field.reshape(-1, 2).copy()

    weights = np.full(positions.shape[0], grid.cell_area, dtype=np.float64)
    return positions, values, weights


def initialize_particle_history(
    grid: Grid2D,
    U0: np.ndarray,
    trusted_only: bool = False,
) -> dict[str, np.ndarray]:
    """
    Build the full particle state needed for Qian's accumulated-history
    representation (eq. 42/56 for flows in R^d).

    Returns a dict with keys:
        positions       (Np, 2)   current particle positions Y^eta_{t_k}
        weights         (Np,)     quadrature weight  s^d  per particle
        u0_carried      (Np, 2)   initial velocity u_0(eta) frozen at birth
        accumulated_G   (Np, 2)   running sum  sum_{j=1}^{k} dt * G(Y, t_{j-1})
    """
    positions, weights = initialize_particles_from_grid(
        grid=grid,
        trusted_only=trusted_only,
    )

    _validate_vector_field(U0, "U0")

    if trusted_only:
        u0_carried = U0[grid.trust_mask].copy()
    else:
        u0_carried = U0.reshape(-1, 2).copy()

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
    grid: Grid2D,
) -> np.ndarray:
    """Boolean mask for particles in the trusted region."""
    _validate_positions(positions)
    return grid.points_in_trusted_region(positions)


def particles_in_padded_box(
    positions: np.ndarray,
    grid: Grid2D,
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

        Y_{k+1} = Y_k  +  dt * U(Y_k, t_k)  +  sqrt(2 nu dt) N(0, I)

    Retained for comparison / testing.
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
    One Milstein step  (Qian eq. 55):

        Y_{k+1} = Y_k
                 + dt * U(Y_k)
                 + sqrt(2 nu) * dB
                 + 0.5 * (nabla U)(Y_k) . U(Y_k)  * [(dB)^2 - dt]

    Parameters
    ----------
    positions : (Np, 2)
        Current particle positions.
    drift : (Np, 2)
        Velocity  U(Y_k, t_k)  interpolated at each particle.
    drift_jacobian : (Np, 2, 2)
        Velocity gradient  (nabla U)(Y_k, t_k)  interpolated at each
        particle.  Convention:  drift_jacobian[p, i, j] = dU_i / dx_j.
    dt : float
        Time-step size  delta.
    nu : float
        Kinematic viscosity.
    rng : Generator
        NumPy random generator.

    Returns
    -------
    new_positions : (Np, 2)

    Notes
    -----
    The diffusion coefficient sigma = sqrt(2 nu) * I is constant, so the
    classical Milstein correction from the noise-times-noise term vanishes.
    Qian's eq. (55) instead includes a drift-derivative correction:

        correction_i = 0.5 * sum_j  U_j * (dU_i / dx_j) * [(dB_j)^2 - dt]

    This is the first additional term in the Ito-Taylor expansion of the
    drift beyond the Euler-Maruyama level, applied component-by-component
    to the Brownian increment.
    """
    _validate_positions(positions)
    _validate_vector_field_flat(drift, "drift")

    Np = positions.shape[0]
    if drift.shape != (Np, 2):
        raise ValueError("drift must have shape (Np, 2).")
    if drift_jacobian.shape != (Np, 2, 2):
        raise ValueError("drift_jacobian must have shape (Np, 2, 2).")
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")
    if nu < 0.0:
        raise ValueError("nu must be non-negative.")

    # Brownian increment  dB ~ N(0, dt I)
    dB = np.sqrt(2.0 * nu * dt) * rng.standard_normal(size=(Np, 2))

    # Euler-Maruyama part
    new_positions = positions + dt * drift + dB

    # Milstein correction (Qian eq. 55, last line)
    #   0.5 * U_j * (dU_i/dx_j) * [(dB_j)^2 - dt]
    # where  dB  above already contains the  sqrt(2 nu)  factor, so
    # (dB_j)^2  corresponds to  2 nu * (xi_j)^2 * dt  with xi ~ N(0,1).
    # Qian writes  sqrt(2 nu) (B_{t_k} - B_{t_{k-1}})  for the noise
    # and then  [(B_{t_k}-B_{t_{k-1}})^2 - dt]  for the correction.
    # We must therefore use the *raw* Brownian increment  dW = dB / sqrt(2nu)
    # when computing  (dW)^2 - dt .
    if nu > 0.0:
        dW = dB / np.sqrt(2.0 * nu)          # shape (Np, 2), standard BM increment
        dW2_minus_dt = dW * dW - dt           # shape (Np, 2)

        # correction_i = 0.5 * sum_j  U_j * (dU_i/dx_j) * (dW_j^2 - dt)
        # drift_jacobian[p, i, j] = dU_i/dx_j
        # We want  sum_j  drift[p, j] * drift_jacobian[p, i, j] * corr[p, j]
        #        = sum_j  (drift * corr)[p, j] * drift_jacobian[p, i, j]
        weighted = drift * dW2_minus_dt       # (Np, 2)  element-wise
        # correction[p, i] = sum_j  drift_jacobian[p, i, j] * weighted[p, j]
        correction = np.einsum("pij,pj->pi", drift_jacobian, weighted)
        new_positions += 0.5 * correction

    return new_positions


# ------------------------------------------------------------------ #
#  Clipping / box utilities
# ------------------------------------------------------------------ #

def clip_positions_to_box(
    positions: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """
    Clip particle positions into the padded box.
    """
    _validate_positions(positions)

    clipped = positions.copy()
    eps_x = 1.0e-12 * max(1.0, grid.dx)
    eps_y = 1.0e-12 * max(1.0, grid.dy)

    clipped[:, 0] = np.clip(clipped[:, 0], -grid.L_box, grid.L_box - eps_x)
    clipped[:, 1] = np.clip(clipped[:, 1], -grid.L_box, grid.L_box - eps_y)

    return clipped


def count_particles_outside_box(
    positions: np.ndarray,
    grid: Grid2D,
) -> int:
    """Count particles lying outside the padded box."""
    _validate_positions(positions)
    inside = grid.points_in_box(positions)
    return int(np.sum(~inside))


def fraction_particles_outside_box(
    positions: np.ndarray,
    grid: Grid2D,
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
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected positions to have shape (Np, 2).")


def _validate_vector_field(field: np.ndarray, name: str) -> None:
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(f"Expected {name} to have shape (Ny, Nx, 2).")


def _validate_vector_field_flat(field: np.ndarray, name: str) -> None:
    if field.ndim != 2 or field.shape[1] != 2:
        raise ValueError(f"Expected {name} to have shape (Np, 2).")


def _validate_weights(weights: np.ndarray, n_particles: int) -> None:
    if weights.ndim != 1 or weights.shape[0] != n_particles:
        raise ValueError("Expected weights to have shape (Np,).")