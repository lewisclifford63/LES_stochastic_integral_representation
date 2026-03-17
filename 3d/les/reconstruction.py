"""
3D particle-to-Eulerian velocity reconstruction and error analysis.
Diagnostics for stochastic integral representation, Qian et al. (2025).
Oxford Mathematics Masters dissertation.
"""

import numpy as np
from les.grid import Grid3D
from les.filter import gaussian_kernel_3d, stencil_half_width_xyz


def reconstruct_velocity_from_particles(positions, amplitudes, weights, grid, sigma_kernel, truncate_radius=2.0):
    """
    Reconstruct 3D velocity field from Lagrangian particle contributions.

    Deposits particle vector amplitudes onto Eulerian grid via 3D Gaussian stencil.

    Parameters
    ----------
    positions : ndarray of shape (Np, 3)
        Particle positions [x, y, z].
    amplitudes : ndarray of shape (Np, 3)
        Vector amplitude (velocity contribution) at each particle.
    weights : ndarray of shape (Np,)
        Weight/volume of each particle.
    grid : Grid3D
        Grid object with spacing and domain information.
    sigma_kernel : float or tuple of floats
        Kernel width. If float, isotropic. If tuple, (sigma_z, sigma_y, sigma_x).
    truncate_radius : float, optional
        Truncate kernel at this many sigma. Default 2.0.

    Returns
    -------
    U_rec : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity field [u, v, w].
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    Np = positions.shape[0]

    # Normalize sigma
    if np.isscalar(sigma_kernel):
        sigma_z, sigma_y, sigma_x = sigma_kernel, sigma_kernel, sigma_kernel
    else:
        sigma_z, sigma_y, sigma_x = sigma_kernel

    # Compute stencil half-widths
    hz, hy, hx = stencil_half_width_xyz(sigma_z, sigma_y, sigma_x,
                                         grid.dz, grid.dy, grid.dx,
                                         truncate_radius)

    U_rec = np.zeros((Nz, Ny, Nx, 3))

    # Loop over particles
    for p in range(Np):
        x_p, y_p, z_p = positions[p, 0], positions[p, 1], positions[p, 2]
        amp_p = amplitudes[p, :]  # [amp_x, amp_y, amp_z]
        w_p = weights[p]

        # Find nearest grid indices
        i_center = np.argmin(np.abs(grid.x - x_p))
        j_center = np.argmin(np.abs(grid.y - y_p))
        k_center = np.argmin(np.abs(grid.z - z_p))

        # Build local 3D stencil bounds
        i_min = max(0, i_center - hx)
        i_max = min(Nx - 1, i_center + hx)
        j_min = max(0, j_center - hy)
        j_max = min(Ny - 1, j_center + hy)
        k_min = max(0, k_center - hz)
        k_max = min(Nz - 1, k_center + hz)

        # Compute distances from particle to grid points in stencil
        Z_stencil = grid.Z[k_min:k_max+1, j_min:j_max+1, i_min:i_max+1]
        Y_stencil = grid.Y[k_min:k_max+1, j_min:j_max+1, i_min:i_max+1]
        X_stencil = grid.X[k_min:k_max+1, j_min:j_max+1, i_min:i_max+1]

        DZ = Z_stencil - z_p
        DY = Y_stencil - y_p
        DX = X_stencil - x_p

        # Compute 3D Gaussian kernel
        K = gaussian_kernel_3d(DX, DY, DZ, sigma_x, sigma_y, sigma_z)

        # Deposit onto all 3 components
        for c in range(3):
            U_rec[k_min:k_max+1, j_min:j_max+1, i_min:i_max+1, c] += w_p * K * amp_p[c]

    return U_rec


def reconstruction_error(U_reference, U_reconstructed):
    """
    Compute pointwise reconstruction error magnitude.

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference (true) velocity field.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity field.

    Returns
    -------
    error_magnitude : ndarray of shape (Nz, Ny, Nx)
        Magnitude of error at each grid point: ||U_ref - U_rec||.
    """
    # Compute difference
    diff = U_reference - U_reconstructed

    # Magnitude: sqrt(du^2 + dv^2 + dw^2)
    err_mag = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2 + diff[..., 2]**2)

    return err_mag


def relative_l2_error(U_reference, U_reconstructed):
    """
    Compute relative L2 error between reference and reconstructed velocity.

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference velocity field.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity field.

    Returns
    -------
    error_rel : float
        Relative L2 error: ||U_ref - U_rec||_L2 / ||U_ref||_L2.
    """
    # Compute difference
    diff = U_reference - U_reconstructed

    # L2 norms (sum over all points and components)
    norm_diff_sq = np.sum(diff[..., 0]**2 + diff[..., 1]**2 + diff[..., 2]**2)
    norm_ref_sq = np.sum(U_reference[..., 0]**2 + U_reference[..., 1]**2 + U_reference[..., 2]**2)

    if norm_ref_sq < 1e-16:
        return 0.0

    error_rel = np.sqrt(norm_diff_sq / norm_ref_sq)
    return error_rel


def relative_l2_error_trusted(U_reference, U_reconstructed, mask=None):
    """
    Compute relative L2 error on trusted/interior region (with optional masking).

    Useful for excluding boundary effects and evaluating interior accuracy.

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference velocity field.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity field.
    mask : ndarray of shape (Nz, Ny, Nx), optional
        Boolean mask (True = trusted region). If None, uses all points.

    Returns
    -------
    error_rel : float
        Relative L2 error on trusted region.
    """
    if mask is None:
        return relative_l2_error(U_reference, U_reconstructed)

    # Apply mask
    U_ref_masked = U_reference[mask, :]
    U_rec_masked = U_reconstructed[mask, :]

    # Compute error
    diff = U_ref_masked - U_rec_masked
    norm_diff_sq = np.sum(diff[..., 0]**2 + diff[..., 1]**2 + diff[..., 2]**2)
    norm_ref_sq = np.sum(U_ref_masked[..., 0]**2 + U_ref_masked[..., 1]**2 + U_ref_masked[..., 2]**2)

    if norm_ref_sq < 1e-16:
        return 0.0

    error_rel = np.sqrt(norm_diff_sq / norm_ref_sq)
    return error_rel


def kinetic_energy_density(U):
    """
    Compute kinetic energy density field: rho * ||U||^2 / 2 (density=1.0).

    Parameters
    ----------
    U : ndarray of shape (Nz, Ny, Nx, 3)
        Velocity field.

    Returns
    -------
    ke_density : ndarray of shape (Nz, Ny, Nx)
        Kinetic energy density.
    """
    ke = 0.5 * (U[..., 0]**2 + U[..., 1]**2 + U[..., 2]**2)
    return ke


def kinetic_energy_error(U_reference, U_reconstructed):
    """
    Compute pointwise kinetic energy error.

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference velocity.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity.

    Returns
    -------
    ke_error : ndarray of shape (Nz, Ny, Nx)
        KE(U_ref) - KE(U_rec).
    """
    ke_ref = kinetic_energy_density(U_reference)
    ke_rec = kinetic_energy_density(U_reconstructed)
    return ke_ref - ke_rec


def relative_kinetic_energy_error(U_reference, U_reconstructed):
    """
    Compute relative kinetic energy error (integral over domain).

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference velocity.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity.

    Returns
    -------
    error_rel : float
        |KE_ref - KE_rec| / KE_ref.
    """
    ke_ref = np.sum(kinetic_energy_density(U_reference))
    ke_rec = np.sum(kinetic_energy_density(U_reconstructed))

    if ke_ref < 1e-16:
        return 0.0

    error_rel = np.abs(ke_ref - ke_rec) / ke_ref
    return error_rel


def vorticity_magnitude(U, grid):
    """
    Compute vorticity magnitude field: ||curl(U)||.

    Uses central differences for curl computation.

    Parameters
    ----------
    U : ndarray of shape (Nz, Ny, Nx, 3)
        Velocity field [u, v, w].
    grid : Grid3D
        Grid object with spacing.

    Returns
    -------
    vort_mag : ndarray of shape (Nz, Ny, Nx)
        Vorticity magnitude at interior points (edges set to zero).
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    vort_mag = np.zeros((Nz, Ny, Nx))

    # Interior region
    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                # Curl components: omega = del x U
                # omega_x = dw/dy - dv/dz
                dw_dy = (U[k, j+1, i, 2] - U[k, j-1, i, 2]) / (2 * grid.dy)
                dv_dz = (U[k+1, j, i, 1] - U[k-1, j, i, 1]) / (2 * grid.dz)
                omega_x = dw_dy - dv_dz

                # omega_y = du/dz - dw/dx
                du_dz = (U[k+1, j, i, 0] - U[k-1, j, i, 0]) / (2 * grid.dz)
                dw_dx = (U[k, j, i+1, 2] - U[k, j, i-1, 2]) / (2 * grid.dx)
                omega_y = du_dz - dw_dx

                # omega_z = dv/dx - du/dy
                dv_dx = (U[k, j, i+1, 1] - U[k, j, i-1, 1]) / (2 * grid.dx)
                du_dy = (U[k, j+1, i, 0] - U[k, j-1, i, 0]) / (2 * grid.dy)
                omega_z = dv_dx - du_dy

                vort_mag[k, j, i] = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    return vort_mag


def vorticity_error(U_reference, U_reconstructed, grid):
    """
    Compute pointwise vorticity magnitude error.

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference velocity.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity.
    grid : Grid3D
        Grid object.

    Returns
    -------
    vort_error : ndarray of shape (Nz, Ny, Nx)
        ||omega_ref|| - ||omega_rec||.
    """
    vort_ref = vorticity_magnitude(U_reference, grid)
    vort_rec = vorticity_magnitude(U_reconstructed, grid)
    return vort_ref - vort_rec


def divergence_field(U, grid):
    """
    Compute velocity divergence field (should be ~0 for incompressible flow).

    Parameters
    ----------
    U : ndarray of shape (Nz, Ny, Nx, 3)
        Velocity field.
    grid : Grid3D
        Grid object with spacing.

    Returns
    -------
    div_u : ndarray of shape (Nz, Ny, Nx)
        Divergence at interior points (edges set to zero).
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    div_u = np.zeros((Nz, Ny, Nx))

    # Interior region
    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                du_dx = (U[k, j, i+1, 0] - U[k, j, i-1, 0]) / (2 * grid.dx)
                dv_dy = (U[k, j+1, i, 1] - U[k, j-1, i, 1]) / (2 * grid.dy)
                dw_dz = (U[k+1, j, i, 2] - U[k-1, j, i, 2]) / (2 * grid.dz)

                div_u[k, j, i] = du_dx + dv_dy + dw_dz

    return div_u


def kinetic_energy_spectrum(U, grid, n_bins=32):
    """
    Compute 1D kinetic energy spectrum |u_hat(k)|^2 via FFT (isotropy assumption).

    Parameters
    ----------
    U : ndarray of shape (Nz, Ny, Nx, 3)
        Velocity field.
    grid : Grid3D
        Grid object.
    n_bins : int, optional
        Number of wavenumber bins.

    Returns
    -------
    k_bins : ndarray of shape (n_bins,)
        Wavenumber bin centers.
    E_k : ndarray of shape (n_bins,)
        Kinetic energy in each wavenumber bin.
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx

    # FFT of each component
    u_hat = np.fft.rfftn(U[..., 0])
    v_hat = np.fft.rfftn(U[..., 1])
    w_hat = np.fft.rfftn(U[..., 2])

    # Energy density in Fourier space
    E_k_full = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)

    # Radial wavenumber
    kz_max = Nz // 2
    ky_max = Ny // 2
    kx_max = Nx // 2 + 1

    # Create wavenumber grids
    kz = np.fft.fftfreq(Nz, d=grid.dz)[:kz_max]
    ky = np.fft.fftfreq(Ny, d=grid.dy)[:ky_max]
    kx = np.fft.rfftfreq(Nx, d=grid.dx)

    # Binning: aggregate by radial wavenumber magnitude
    k_max = np.sqrt(kz_max**2 + ky_max**2 + kx_max**2)
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    E_k = np.zeros(n_bins)
    count = np.zeros(n_bins)

    for iz in range(min(kz_max, Nz)):
        for iy in range(min(ky_max, Ny)):
            for ix in range(kx_max):
                k_mag = np.sqrt(kz[iz]**2 + ky[iy]**2 + kx[ix]**2)
                bin_idx = np.searchsorted(k_bins, k_mag) - 1
                if 0 <= bin_idx < n_bins:
                    E_k[bin_idx] += E_k_full[iz, iy, ix]
                    count[bin_idx] += 1

    # Average over bin
    E_k = np.divide(E_k, count, where=count > 0, out=np.zeros_like(E_k))

    return k_centers, E_k


def component_rmse(U_reference, U_reconstructed, component=0):
    """
    Compute RMSE for a single velocity component.

    Parameters
    ----------
    U_reference : ndarray of shape (Nz, Ny, Nx, 3)
        Reference velocity.
    U_reconstructed : ndarray of shape (Nz, Ny, Nx, 3)
        Reconstructed velocity.
    component : int, optional
        Component index (0=u, 1=v, 2=w).

    Returns
    -------
    rmse : float
        Root mean square error for the specified component.
    """
    diff = U_reference[..., component] - U_reconstructed[..., component]
    rmse = np.sqrt(np.mean(diff**2))
    return rmse
