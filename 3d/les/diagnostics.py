"""
3D Diagnostics Module
=====================
Computes kinetic energy, divergence statistics, and incompressibility metrics
for 3D Large Eddy Simulation (Qian et al., 2025).

Convention:
- Scalar fields: shape (Nz, Ny, Nx)
- Vector fields: shape (Nz, Ny, Nx, 3)
- Particle positions: shape (Np, 3) with columns [x, y, z]
"""

import numpy as np
from les.grid import Grid3D


def kinetic_energy(U: np.ndarray, grid: Grid3D) -> float:
    """
    Compute domain-integrated kinetic energy.

    KE = (1/2) ∫ |u|² dV = (1/2) ∑ (u_x² + u_y² + u_z²) * dV

    Integration via cell volumes.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance with dx, dy, dz attributes.

    Returns
    -------
    ke : float
        Total kinetic energy.
    """
    _validate_vector_field(U)

    speed_squared = (U[..., 0]**2 +
                     U[..., 1]**2 +
                     U[..., 2]**2)

    cell_volume = grid.dx * grid.dy * grid.dz
    ke = 0.5 * np.sum(speed_squared) * cell_volume

    return float(ke)


def trusted_kinetic_energy(U: np.ndarray, grid: Grid3D,
                          restrict_to_trusted: bool = True) -> float:
    """
    Compute kinetic energy restricting to the trusted interior region.

    Uses the grid's actual trusted slices (based on L_trust) rather than
    a simple [1:-1] trim.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.
    restrict_to_trusted : bool
        If True, restrict to trusted region.

    Returns
    -------
    ke : float
        Interior kinetic energy.
    """
    _validate_vector_field(U)

    if restrict_to_trusted:
        sz, sy, sx = grid.trusted_slices
        vel_interior = U[sz, sy, sx]
    else:
        vel_interior = U

    speed_squared = (vel_interior[..., 0]**2 +
                     vel_interior[..., 1]**2 +
                     vel_interior[..., 2]**2)

    cell_volume = grid.dx * grid.dy * grid.dz
    ke = 0.5 * np.sum(speed_squared) * cell_volume

    return float(ke)


def max_speed(velocity: np.ndarray) -> float:
    """
    Maximum velocity magnitude in domain.

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).

    Returns
    -------
    max_spd : float
        Maximum |u| = sqrt(u_x² + u_y² + u_z²).
    """
    _validate_vector_field(velocity)
    speed = np.sqrt(velocity[..., 0]**2 +
                    velocity[..., 1]**2 +
                    velocity[..., 2]**2)
    return float(np.max(speed))


def mean_speed(velocity: np.ndarray) -> float:
    """
    Mean velocity magnitude in domain.

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).

    Returns
    -------
    mean_spd : float
        Mean |u|.
    """
    _validate_vector_field(velocity)
    speed = np.sqrt(velocity[..., 0]**2 +
                    velocity[..., 1]**2 +
                    velocity[..., 2]**2)
    return float(np.mean(speed))


def trusted_speed_stats(U: np.ndarray, grid: Grid3D,
                       restrict_to_trusted: bool = True) -> tuple[float, float]:
    """
    Compute speed statistics (max, mean) on interior points.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.
    restrict_to_trusted : bool
        If True, use only interior points.

    Returns
    -------
    max_spd : float
        Maximum speed on interior.
    mean_spd : float
        Mean speed on interior.
    """
    _validate_vector_field(U)

    if restrict_to_trusted:
        sz, sy, sx = grid.trusted_slices
        vel = U[sz, sy, sx]
    else:
        vel = U

    speed = np.sqrt(vel[..., 0]**2 +
                    vel[..., 1]**2 +
                    vel[..., 2]**2)

    return float(np.max(speed)), float(np.mean(speed))


def divergence_stats(U: np.ndarray, grid: Grid3D) -> tuple[float, float]:
    """
    Compute divergence statistics (max, mean).

    For incompressible flow, all should be ~0.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance with dx, dy, dz.

    Returns
    -------
    max_div : float
        Maximum absolute divergence.
    mean_div : float
        Mean absolute divergence.
    """
    _validate_vector_field(U)

    div = _compute_divergence_3d(U, grid.dx, grid.dy, grid.dz)

    return float(np.max(np.abs(div))), float(np.mean(np.abs(div)))


def trusted_divergence_stats(U: np.ndarray, grid: Grid3D,
                            restrict_to_trusted: bool = True) -> tuple[float, float]:
    """
    Compute divergence statistics on interior points only.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.
    restrict_to_trusted : bool
        If True, exclude boundary layers.

    Returns
    -------
    max_div : float
        Maximum absolute divergence.
    mean_div : float
        Mean absolute divergence.
    """
    _validate_vector_field(U)

    div = _compute_divergence_3d(U, grid.dx, grid.dy, grid.dz)

    if restrict_to_trusted:
        sz, sy, sx = grid.trusted_slices
        div = div[sz, sy, sx]

    return float(np.max(np.abs(div))), float(np.mean(np.abs(div)))


def incompressibility_test(U: np.ndarray, grid: Grid3D,
                          grad_norm_floor: float = 1e-14) -> float:
    """
    Test incompressibility: normalized mean|∇·u| / ||∇u||.

    Implements check from Qian et al. (2025) Section 4.3.1.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.
    grad_norm_floor : float
        Floor for gradient norm to avoid division by zero.

    Returns
    -------
    normalized_div : float
        Normalized incompressibility metric.
    """
    _validate_vector_field(U)

    norm_div = _pointwise_normalized_divergence(U, grid.dx, grid.dy, grid.dz)
    return float(np.mean(np.abs(norm_div)))


def trusted_incompressibility_test(U: np.ndarray, grid: Grid3D,
                                  grad_norm_floor: float = 1e-14) -> float:
    """
    Incompressibility test on interior points (Qian et al., 2025, Section 4.3.1).

    Evaluates  test(t) = mean_{x ∈ Ω_int} ( |∇·u(x)| / ‖∇u(x)‖_F )

    The evaluation region Ω_int is the trusted region shrunk by one grid cell
    on all six faces.  This ensures that the central-difference stencils used
    for both |∇·u| and ‖∇u‖_F never reach into the padded computational
    buffer, where the particle deposit is less accurate and would artificially
    inflate the divergence at the trusted-region boundary.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.
    grad_norm_floor : float
        Floor for gradient norm to avoid division by zero.

    Returns
    -------
    normalized_div : float
        Normalized incompressibility metric on interior of trusted region.
    """
    _validate_vector_field(U)

    norm_div = _pointwise_normalized_divergence(U, grid.dx, grid.dy, grid.dz)

    # Shrink trusted slices by 1 cell on each face to avoid stencil
    # contamination from the padded buffer region.
    sz, sy, sx = grid.trusted_slices
    sz_inner = slice(sz.start + 1, sz.stop - 1)
    sy_inner = slice(sy.start + 1, sy.stop - 1)
    sx_inner = slice(sx.start + 1, sx.stop - 1)

    norm_div_interior = norm_div[sz_inner, sy_inner, sx_inner]
    return float(np.mean(np.abs(norm_div_interior)))


def trusted_incompressibility_test_l2(U: np.ndarray, grid: Grid3D) -> float:
    """
    Global-norm form of the incompressibility test (interior region).

    Computes the ratio of L² norms:

        test_L2(t) = ‖∇·u‖_{L²(Ω_int)} / ‖∇u‖_{L²,F(Ω_int)}
                   = √(∑ (∇·u)²) / √(∑ ‖∇u‖²_F)

    This is an alternative to the pointwise-mean form and may match Qian's
    exact definition depending on the paper's notation.  Both are reported
    in the diagnostics so the two formulations can be compared.

    Parameters
    ----------
    U : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.

    Returns
    -------
    test_l2 : float
        Global-L² incompressibility ratio on interior of trusted region.
    """
    _validate_vector_field(U)

    div = _compute_divergence_3d(U, grid.dx, grid.dy, grid.dz)

    sz, sy, sx = grid.trusted_slices
    sz_inner = slice(sz.start + 1, sz.stop - 1)
    sy_inner = slice(sy.start + 1, sy.stop - 1)
    sx_inner = slice(sx.start + 1, sx.stop - 1)

    div_int = div[sz_inner, sy_inner, sx_inner]

    # Frobenius norm of gradient tensor over same interior region
    Nz, Ny, Nx = U.shape[:3]
    grad_frob_sq = np.zeros((Nz, Ny, Nx))
    for comp in range(3):
        grad_frob_sq += (
            ((np.roll(U[..., comp], -1, axis=2) - np.roll(U[..., comp], 1, axis=2)) / (2*grid.dx))**2 +
            ((np.roll(U[..., comp], -1, axis=1) - np.roll(U[..., comp], 1, axis=1)) / (2*grid.dy))**2 +
            ((np.roll(U[..., comp], -1, axis=0) - np.roll(U[..., comp], 1, axis=0)) / (2*grid.dz))**2
        )
    grad_frob_int = grad_frob_sq[sz_inner, sy_inner, sx_inner]

    denom = np.sqrt(np.sum(grad_frob_int))
    if denom < 1e-14:
        return 0.0
    return float(np.sqrt(np.sum(div_int**2)) / denom)


def relative_l2_change(U_old: np.ndarray,
                      U_new: np.ndarray) -> float:
    """
    Compute relative L² change between two velocity fields.

    Δ = ||u_new - u_old||₂ / ||u_old||₂

    Parameters
    ----------
    U_old : np.ndarray
        Previous velocity field (Nz, Ny, Nx, 3).
    U_new : np.ndarray
        Current velocity field (Nz, Ny, Nx, 3).

    Returns
    -------
    rel_change : float
        Relative L² difference.
    """
    _validate_vector_field(U_old)
    _validate_vector_field(U_new)

    diff = U_new - U_old
    diff_norm = np.sqrt(np.sum(diff**2))

    old_norm = np.sqrt(np.sum(U_old**2))

    if old_norm > 0:
        return float(diff_norm / old_norm)
    else:
        return float(diff_norm)


def trusted_relative_l2_change(U_old: np.ndarray,
                              U_new: np.ndarray, grid: Grid3D) -> float:
    """
    Relative L² change computed on interior points only.

    Parameters
    ----------
    U_old : np.ndarray
        Previous velocity field (Nz, Ny, Nx, 3).
    U_new : np.ndarray
        Current velocity field (Nz, Ny, Nx, 3).
    grid : Grid3D
        Grid instance.

    Returns
    -------
    rel_change : float
        Relative L² difference on interior.
    """
    _validate_vector_field(U_old)
    _validate_vector_field(U_new)

    sz, sy, sx = grid.trusted_slices
    v_old = U_old[sz, sy, sx]
    v_new = U_new[sz, sy, sx]

    diff = v_new - v_old
    diff_norm = np.sqrt(np.sum(diff**2))

    old_norm = np.sqrt(np.sum(v_old**2))

    if old_norm > 0:
        return float(diff_norm / old_norm)
    else:
        return float(diff_norm)


def particle_box_escape_fraction(positions: np.ndarray, grid: Grid3D) -> float:
    """
    Compute fraction of particles that have escaped the computational domain.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (Np, 3) with columns [x, y, z].
    grid : Grid3D
        Grid instance with domain bounds.

    Returns
    -------
    escape_fraction : float
        Fraction of particles outside domain (0 to 1).
    """
    _validate_positions(positions)

    escaped = ~grid.points_in_box(positions)

    Np = positions.shape[0]
    if Np == 0:
        return 0.0

    return float(np.sum(escaped) / Np)


def particle_trusted_fraction(positions: np.ndarray, grid: Grid3D) -> float:
    """
    Compute fraction of particles still within domain bounds.

    Complement of particle_box_escape_fraction.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (Np, 3).
    grid : Grid3D
        Grid instance with domain bounds.

    Returns
    -------
    trusted_fraction : float
        Fraction of particles inside domain (0 to 1).
    """
    escape_frac = particle_box_escape_fraction(positions, grid)
    return 1.0 - escape_frac


# ============================================================================
# Private Helper Functions
# ============================================================================


def _compute_divergence_3d(velocity: np.ndarray,
                          dx: float,
                          dy: float,
                          dz: float) -> np.ndarray:
    """
    Compute ∇·u = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z using central differences.

    Assumes periodic boundary conditions (uses np.roll).

    Parameters
    ----------
    velocity : np.ndarray
        Vector field of shape (Nz, Ny, Nx, 3).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    divergence : np.ndarray
        Scalar field of shape (Nz, Ny, Nx).
    """
    Nz, Ny, Nx = velocity.shape[:3]
    div = np.zeros((Nz, Ny, Nx), dtype=velocity.dtype)

    # ∂u_x/∂x (axis=2 is x-direction)
    div += (np.roll(velocity[..., 0], -1, axis=2) -
            np.roll(velocity[..., 0],  1, axis=2)) / (2.0 * dx)

    # ∂u_y/∂y (axis=1 is y-direction)
    div += (np.roll(velocity[..., 1], -1, axis=1) -
            np.roll(velocity[..., 1],  1, axis=1)) / (2.0 * dy)

    # ∂u_z/∂z (axis=0 is z-direction)
    div += (np.roll(velocity[..., 2], -1, axis=0) -
            np.roll(velocity[..., 2],  1, axis=0)) / (2.0 * dz)

    return div


def _pointwise_normalized_divergence(velocity: np.ndarray,
                                    dx: float,
                                    dy: float,
                                    dz: float) -> np.ndarray:
    """
    Compute normalized divergence at each point.

    Used for incompressibility metrics. Divides pointwise divergence by
    local velocity gradient norm (Frobenius norm of 3×3 Jacobian).

    Parameters
    ----------
    velocity : np.ndarray
        Vector field (Nz, Ny, Nx, 3).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    norm_div : np.ndarray
        Normalized divergence field (Nz, Ny, Nx).
    """
    Nz, Ny, Nx = velocity.shape[:3]

    # Compute divergence
    div = _compute_divergence_3d(velocity, dx, dy, dz)

    # Compute Frobenius norm of velocity gradient tensor ∇u (3×3 matrix)
    # ∇u = [∂u_i/∂x_j] where i,j ∈ {x,y,z}
    grad_u_frob = np.zeros((Nz, Ny, Nx))

    # Row 1: ∂u_x/∂x, ∂u_x/∂y, ∂u_x/∂z
    dux_dx = (np.roll(velocity[..., 0], -1, axis=2) -
              np.roll(velocity[..., 0],  1, axis=2)) / (2.0 * dx)
    dux_dy = (np.roll(velocity[..., 0], -1, axis=1) -
              np.roll(velocity[..., 0],  1, axis=1)) / (2.0 * dy)
    dux_dz = (np.roll(velocity[..., 0], -1, axis=0) -
              np.roll(velocity[..., 0],  1, axis=0)) / (2.0 * dz)

    # Row 2: ∂u_y/∂x, ∂u_y/∂y, ∂u_y/∂z
    duy_dx = (np.roll(velocity[..., 1], -1, axis=2) -
              np.roll(velocity[..., 1],  1, axis=2)) / (2.0 * dx)
    duy_dy = (np.roll(velocity[..., 1], -1, axis=1) -
              np.roll(velocity[..., 1],  1, axis=1)) / (2.0 * dy)
    duy_dz = (np.roll(velocity[..., 1], -1, axis=0) -
              np.roll(velocity[..., 1],  1, axis=0)) / (2.0 * dz)

    # Row 3: ∂u_z/∂x, ∂u_z/∂y, ∂u_z/∂z
    duz_dx = (np.roll(velocity[..., 2], -1, axis=2) -
              np.roll(velocity[..., 2],  1, axis=2)) / (2.0 * dx)
    duz_dy = (np.roll(velocity[..., 2], -1, axis=1) -
              np.roll(velocity[..., 2],  1, axis=1)) / (2.0 * dy)
    duz_dz = (np.roll(velocity[..., 2], -1, axis=0) -
              np.roll(velocity[..., 2],  1, axis=0)) / (2.0 * dz)

    # Frobenius norm: ||∇u||²_F = sum of all squared entries
    grad_u_frob = np.sqrt(
        dux_dx**2 + dux_dy**2 + dux_dz**2 +
        duy_dx**2 + duy_dy**2 + duy_dz**2 +
        duz_dx**2 + duz_dy**2 + duz_dz**2
    )

    # Normalized divergence (avoid division by zero)
    norm_div = np.zeros_like(div)
    mask = grad_u_frob > 1e-14
    norm_div[mask] = div[mask] / grad_u_frob[mask]

    return norm_div


def _validate_vector_field(velocity: np.ndarray) -> None:
    """
    Validate vector field shape (*, *, *, 3).

    Parameters
    ----------
    velocity : np.ndarray
        Field to validate.

    Raises
    ------
    ValueError
        If shape[-1] != 3.
    """
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError(
            f"Vector field must have shape (Nz, Ny, Nx, 3), got {velocity.shape}"
        )


def _validate_positions(positions: np.ndarray) -> None:
    """
    Validate particle positions shape (Np, 3).

    Parameters
    ----------
    positions : np.ndarray
        Positions array.

    Raises
    ------
    ValueError
        If shape != (Np, 3).
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Positions must have shape (Np, 3), got {positions.shape}"
        )
