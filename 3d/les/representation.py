"""
3D velocity reconstruction via Qian et al. (2025) integral representation.
Stochastic Lagrangian particle method for LES subgrid modeling.
Oxford Mathematics Masters dissertation.

Direct particle sum (no denominator normalisation):
    U(x) = sum_p  w_p * chi(x - Y_p) * v_p

where chi is the 3D Gaussian kernel:
    chi(r; sigma) = (2*pi*sigma^2)^{-3/2} exp(-|r|^2 / (2*sigma^2))
"""

import numpy as np
from les.grid import Grid3D
from les.filter import gaussian_kernel_3d
from les.interpolation import trilinear_interpolate_vector, trilinear_interpolate_tensor
from les.differential_operators import grad_velocity
from les.particles import advance_particles_milstein, advance_particles_euler_maruyama, clip_positions_to_box, wrap_positions_periodic
from dns.poisson import project_velocity_3d, project_velocity_spectral


def _compute_half_widths(sigma: float, cutoff_sigma: float,
                         dx: float, dy: float, dz: float) -> tuple[int, int, int]:
    """Compute stencil half-widths in grid cells for each direction."""
    cutoff = cutoff_sigma * sigma
    hx = int(np.ceil(cutoff / dx))
    hy = int(np.ceil(cutoff / dy))
    hz = int(np.ceil(cutoff / dz))
    return hz, hy, hx


def deposit_particle_sum(grid, positions, carried_values, weights, sigma, cutoff_sigma):
    """
    Deposit particle vector quantities onto Eulerian grid via 3D Gaussian kernel
    with periodic boundary conditions.

    Direct particle sum (Qian eq. 56, no normalisation):
        U(x) = sum_p  w_p * chi(x - Y_p) * v_p

    Uses periodic wrapping for stencil indices and minimum-image distances,
    ensuring particles near box boundaries contribute correctly to grid nodes
    on the opposite side.  This is essential for maintaining divergence-free
    reconstructions in a periodic domain.

    Parameters
    ----------
    grid : Grid3D
        Grid object with spacing and domain information.
    positions : ndarray of shape (Np, 3)
        Particle positions [x, y, z], assumed wrapped into [-L_box, L_box).
    carried_values : ndarray of shape (Np, 3)
        Vector quantity at each particle (u0 + G_acc).
    weights : ndarray of shape (Np,)
        Weight/volume of each particle (= s^d = cell_volume).
    sigma : float
        Isotropic kernel width.
    cutoff_sigma : float
        Truncate kernel at this many sigma.

    Returns
    -------
    field : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed vector field on Eulerian grid.
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx

    sig = float(sigma) if np.isscalar(sigma) else float(sigma[0])
    hz, hy, hx = _compute_half_widths(sig, cutoff_sigma,
                                       grid.dx, grid.dy, grid.dz)

    field = np.zeros((Nz, Ny, Nx, 3), dtype=np.float64)
    field_flat = field.reshape(-1, 3)   # C-contiguous view, shape (Nz*Ny*Nx, 3)

    # Nearest-grid-node index for every particle.
    # Positions are wrapped to [-L_box, L_box); grid.x[0] = -L_box.
    # This gives indices in [0, Nx) for properly wrapped particles.
    i_centers = np.round((positions[:, 0] - grid.x[0]) / grid.dx).astype(np.int64)
    j_centers = np.round((positions[:, 1] - grid.y[0]) / grid.dy).astype(np.int64)
    k_centers = np.round((positions[:, 2] - grid.z[0]) / grid.dz).astype(np.int64)

    # Pre-scale carried values by weight once: shape (Np, 3)
    wv = weights[:, None] * carried_values

    grid_size = Nz * Ny * Nx

    # Period of the domain in each direction (for minimum-image distances)
    Lx_period = Nx * grid.dx   # = 2 * L_box
    Ly_period = Ny * grid.dy
    Lz_period = Nz * grid.dz

    # Outer loop: stencil offsets only (≤ 7³ = 343 Python iterations)
    for dk in range(-hz, hz + 1):
        for dj in range(-hy, hy + 1):
            for di in range(-hx, hx + 1):
                # Stencil node indices with PERIODIC WRAPPING
                ii = (i_centers + di) % Nx   # (Np,)  always in [0, Nx)
                jj = (j_centers + dj) % Ny   # (Np,)  always in [0, Ny)
                kk = (k_centers + dk) % Nz   # (Np,)  always in [0, Nz)

                # Distance from stencil grid-node to each particle,
                # using minimum-image convention for periodic domain.
                DX = grid.x[ii] - positions[:, 0]
                DY = grid.y[jj] - positions[:, 1]
                DZ = grid.z[kk] - positions[:, 2]

                # Minimum-image: wrap distances to [-L/2, L/2)
                DX = DX - Lx_period * np.round(DX / Lx_period)
                DY = DY - Ly_period * np.round(DY / Ly_period)
                DZ = DZ - Lz_period * np.round(DZ / Lz_period)

                # Gaussian kernel evaluated at all particles simultaneously
                K = gaussian_kernel_3d(DX, DY, DZ, sig)   # (Np,)

                # Contribution: K_p * w_p * v_p, shape (Np, 3)
                contrib = K[:, None] * wv   # (Np, 3)

                # Raveled flat index into (Nz*Ny*Nx,) for scatter-add
                linear_idx = (kk * (Ny * Nx) + jj * Nx + ii).astype(np.int64)

                # Scatter-add via bincount (handles duplicate indices correctly)
                for c in range(3):
                    field_flat[:, c] += np.bincount(
                        linear_idx, weights=contrib[:, c], minlength=grid_size
                    )

    return field


def accumulated_history_velocity_update(grid, particle_state, U_n, g_n, dt, nu,
                                         sigma, cutoff_sigma, rng,
                                         clip_to_box=False, wrap_periodic=False,
                                         use_milstein=True, project_advection=True):
    """
    One step of Qian's accumulated-history scheme (eqs. 42/56).

    Each particle carries:
      - u0_carried: frozen initial velocity at particle birth
      - accumulated_G: running integral of G = (-grad p + F) along trajectory

    The reconstructed velocity at t_{n+1} is:
      U_{n+1}(x) = sum_p w_p * chi(x - Y_p^{n+1}) * [u0_p + G_acc_p]

    Parameters
    ----------
    grid : Grid3D
    particle_state : dict with keys 'positions', 'weights', 'u0_carried', 'accumulated_G'
    U_n : (Nz, Ny, Nx, 3) Eulerian velocity
    g_n : (Nz, Ny, Nx, 3) RHS field g = -grad_p + F
    dt, nu : float
    sigma, cutoff_sigma : float
    rng : np.random.Generator
    clip_to_box : bool
        Legacy clipping (teleports to boundary — not recommended).
    wrap_periodic : bool
        Wrap particles periodically into [-L_box, L_box).  Consistent with
        the periodic FFT pressure solver and ensures every grid cell always
        receives contributions from nearby particles.
    use_milstein : bool
    project_advection : bool
        If True, Hodge-project U_n before using it for particle advection.
        This breaks the positive feedback loop where slightly divergent
        deposit → compressible advection → worse deposit → more divergence.
        The raw (unprojected) deposit U_np1 is still returned for honest
        test(t) measurement.

    Returns
    -------
    U_np1 : (Nz, Ny, Nx, 3)
    particle_state_new : dict
    """
    positions = particle_state["positions"]
    weights = particle_state["weights"]
    u0_carried = particle_state["u0_carried"]
    G_accumulated = particle_state["accumulated_G"]

    # Optionally project U_n to break the divergence feedback loop.
    # The true velocity IS divergence-free, so the SDE drift should be too.
    # We project before interpolating for advection, but keep the raw U_n
    # for the pressure source (which depends on the actual velocity).
    # Use the spectral Leray-Hodge projection (true wavenumbers) for
    # consistency with the spectral K-kernel pressure gradient.
    if project_advection:
        U_advect, _ = project_velocity_spectral(U_n, grid.dx, grid.dy, grid.dz)
    else:
        U_advect = U_n

    # Interpolate fields at particle locations
    U_interp = trilinear_interpolate_vector(grid, U_advect, positions)
    g_interp = trilinear_interpolate_vector(grid, g_n, positions)

    # Accumulate G integral (trapezoidal-like: G_acc += g * dt)
    G_accumulated = G_accumulated + g_interp * dt

    # Advance particle positions
    if use_milstein:
        gradU = grad_velocity(U_advect, grid.dx, grid.dy, grid.dz)
        drift_jac = trilinear_interpolate_tensor(grid, gradU, positions)
        positions_new = advance_particles_milstein(positions, U_interp, drift_jac, dt, nu, rng)
    else:
        positions_new = advance_particles_euler_maruyama(positions, U_interp, dt, nu, rng)

    if wrap_periodic:
        positions_new = wrap_positions_periodic(positions_new, grid)
    elif clip_to_box:
        positions_new = clip_positions_to_box(positions_new, grid)

    # Reconstruct: U_{n+1} = deposit(u0 + G_acc)
    amplitude = u0_carried + G_accumulated
    U_np1 = deposit_particle_sum(grid, positions_new, amplitude, weights, sigma, cutoff_sigma)

    particle_state_new = {
        "positions": positions_new,
        "weights": weights,
        "u0_carried": u0_carried,
        "accumulated_G": G_accumulated,
    }

    return U_np1, particle_state_new


def one_step_velocity_update(grid, positions_n, U_n, g_n, weights, dt, nu,
                              sigma, cutoff_sigma, rng,
                              clip_to_box=False, mass_tol=1e-14):
    """
    Legacy single-step normalised-conditional-average update.

    Returns
    -------
    U_np1_raw, positions_np1, mass_field, E_u, E_g
    """
    U_interp = trilinear_interpolate_vector(grid, U_n, positions_n)
    g_interp = trilinear_interpolate_vector(grid, g_n, positions_n)

    # Euler-Maruyama
    positions_np1 = advance_particles_euler_maruyama(positions_n, U_interp, dt, nu, rng)

    if clip_to_box:
        positions_np1 = clip_positions_to_box(positions_np1, grid)

    # Deposit
    E_u = deposit_particle_sum(grid, positions_np1, U_interp, weights, sigma, cutoff_sigma)
    E_g = deposit_particle_sum(grid, positions_np1, g_interp, weights, sigma, cutoff_sigma)

    # Mass field (scalar deposit)
    mass_field = np.zeros(grid.shape)
    Np = positions_np1.shape[0]
    sig = float(sigma) if np.isscalar(sigma) else float(sigma[0])
    hz, hy, hx = _compute_half_widths(sig, cutoff_sigma, grid.dx, grid.dy, grid.dz)
    for p in range(Np):
        x_p, y_p, z_p = positions_np1[p]
        w_p = weights[p]
        i_c = np.argmin(np.abs(grid.x - x_p))
        j_c = np.argmin(np.abs(grid.y - y_p))
        k_c = np.argmin(np.abs(grid.z - z_p))
        i0, i1 = max(0, i_c - hx), min(grid.Nx - 1, i_c + hx)
        j0, j1 = max(0, j_c - hy), min(grid.Ny - 1, j_c + hy)
        k0, k1 = max(0, k_c - hz), min(grid.Nz - 1, k_c + hz)
        DX = grid.X[k0:k1+1, j0:j1+1, i0:i1+1] - x_p
        DY = grid.Y[k0:k1+1, j0:j1+1, i0:i1+1] - y_p
        DZ = grid.Z[k0:k1+1, j0:j1+1, i0:i1+1] - z_p
        K = gaussian_kernel_3d(DX, DY, DZ, sig)
        mass_field[k0:k1+1, j0:j1+1, i0:i1+1] += w_p * K

    # Normalised reconstruction
    U_np1_raw = np.zeros(grid.vector_shape)
    safe = mass_field > mass_tol
    for c in range(3):
        U_np1_raw[..., c][safe] = (E_u[..., c][safe] + dt * E_g[..., c][safe]) / mass_field[safe]

    return U_np1_raw, positions_np1, mass_field, E_u, E_g
