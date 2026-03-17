"""
3D FFT-based Poisson solver for DNS of incompressible flows.

References:
    Qian et al. (2025) - LES stochastic integral representation

Implements periodic boundary conditions via spectral methods in 3D.
"""

import numpy as np


def fft_wave_numbers(N, L):
    r"""
    Compute FFT wave numbers for periodic domain.

    Parameters
    ----------
    N : int
        Number of grid points
    L : float
        Domain length

    Returns
    -------
    k : ndarray
        Wave numbers k = 2π n / L for n in [0, N/2, -(N/2)+1, ..., -1]
    """
    return 2.0 * np.pi * np.fft.fftfreq(N, d=L/N)


def solve_poisson_periodic_3d(rhs, dx, dy, dz):
    r"""
    Solve 3D Poisson equation ∆φ = rhs with periodic boundary conditions.

    Uses FFT-based spectral method with **modified wavenumbers** that match
    the central-difference Laplacian operator.  This ensures that the
    discrete divergence of the discrete gradient of φ equals rhs exactly
    (to machine precision), which is essential for maintaining ∇·g = 0
    in the pressure computation.

    The central-difference second derivative in direction x has the Fourier
    symbol  -(2 sin(kx·dx/2) / dx)²  rather than  -kx².  Using these
    modified wavenumbers makes the FFT Poisson solve **consistent** with
    the central-difference gradient/divergence operators used elsewhere.

    Parameters
    ----------
    rhs : ndarray
        Right-hand side of Poisson equation, shape (Nz, Ny, Nx)
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    dz : float
        Grid spacing in z-direction

    Returns
    -------
    phi : ndarray
        Solution to Poisson equation, shape (Nz, Ny, Nx)
    """
    Nz, Ny, Nx = rhs.shape

    # Compute wave numbers in each direction
    kx = fft_wave_numbers(Nx, dx * Nx)
    ky = fft_wave_numbers(Ny, dy * Ny)
    kz = fft_wave_numbers(Nz, dz * Nz)

    # Create 3D meshgrid with shape (Nz, Ny, Nx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

    # Modified wavenumbers for the CHAINED central-difference operators:
    #   div(grad(f)) = d/dx(df/dx) + d/dy(df/dy) + d/dz(df/dz)
    # where each d/dx uses the central-difference stencil (f[i+1]-f[i-1])/(2h).
    # Composing two such first derivatives gives Fourier symbol  -sin²(k·h)/h²
    # (NOT the compact Laplacian symbol -4sin²(k·h/2)/h²).
    # Using this ensures div(grad(phi)) = rhs exactly to machine precision,
    # so that g = -grad(p) + F is numerically divergence-free.
    k2_mod = (np.sin(KX * dx) / dx) ** 2 \
           + (np.sin(KY * dy) / dy) ** 2 \
           + (np.sin(KZ * dz) / dz) ** 2

    # Transform RHS to Fourier space
    rhs_hat = np.fft.fftn(rhs)

    # Solve in Fourier space: φ̂ = -r̂ / k̃²
    # Avoid division by zero at k = 0 (set φ̂[0,0,0] = 0)
    phi_hat = np.zeros_like(rhs_hat, dtype=complex)
    mask = k2_mod > 0
    phi_hat[mask] = -rhs_hat[mask] / k2_mod[mask]

    # Transform back to physical space
    phi = np.fft.ifftn(phi_hat).real

    return phi


def project_velocity_3d(U, dx, dy, dz):
    r"""
    Project velocity field onto divergence-free space in 3D.

    Enforces continuity equation ∇·U = 0 by solving for pressure correction:
        ∆p = ∇·U
    Then updates U_proj = U - ∇p.

    Uses central differences for divergence and gradient computations with
    periodic boundary conditions (np.roll).  This is the legacy
    finite-difference-consistent version; see ``project_velocity_spectral``
    for the true spectral Leray-Hodge projection.

    Parameters
    ----------
    U : ndarray
        Velocity field, shape (Nz, Ny, Nx, 3)
        Components are U[..., 0] = u_x, U[..., 1] = u_y, U[..., 2] = u_z
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    dz : float
        Grid spacing in z-direction

    Returns
    -------
    U_proj : ndarray
        Divergence-free velocity field, shape (Nz, Ny, Nx, 3)
    p_corr : ndarray
        Pressure correction field, shape (Nz, Ny, Nx)
    """
    Nz, Ny, Nx = U.shape[:3]

    # Compute divergence ∇·U = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
    # Central differences with periodic BCs via roll
    du_x_dx = (np.roll(U[..., 0], -1, axis=2) - np.roll(U[..., 0], 1, axis=2)) / (2.0 * dx)
    du_y_dy = (np.roll(U[..., 1], -1, axis=1) - np.roll(U[..., 1], 1, axis=1)) / (2.0 * dy)
    du_z_dz = (np.roll(U[..., 2], -1, axis=0) - np.roll(U[..., 2], 1, axis=0)) / (2.0 * dz)

    divU = du_x_dx + du_y_dy + du_z_dz

    # Solve Poisson equation for pressure correction
    p_corr = solve_poisson_periodic_3d(divU, dx, dy, dz)

    # Compute pressure gradients via central differences
    dp_dx = (np.roll(p_corr, -1, axis=2) - np.roll(p_corr, 1, axis=2)) / (2.0 * dx)
    dp_dy = (np.roll(p_corr, -1, axis=1) - np.roll(p_corr, 1, axis=1)) / (2.0 * dy)
    dp_dz = (np.roll(p_corr, -1, axis=0) - np.roll(p_corr, 1, axis=0)) / (2.0 * dz)

    # Project velocity: U_proj = U - ∇p
    U_proj = U.copy()
    U_proj[..., 0] -= dp_dx
    U_proj[..., 1] -= dp_dy
    U_proj[..., 2] -= dp_dz

    return U_proj, p_corr


def project_velocity_spectral(U, dx, dy, dz):
    r"""
    Spectral Leray-Hodge projection onto divergence-free space in 3D.

    Implements the exact spectral projector:
        ℙ[U] = U - ∇(Δ⁻¹ ∇·U)

    entirely in Fourier space using TRUE wavenumbers.  The projected field
    satisfies ∇·ℙ[U] = 0 to machine precision in the spectral sense.

    This is the spectral analogue of ``project_velocity_3d`` but avoids
    the modified-wavenumber / central-difference approximation.  For use
    as the advection velocity in the Qian LES loop when
    ``project_advection=True``, this ensures the SDE drift is spectrally
    divergence-free, consistent with the spectral K-kernel pressure
    gradient.

    In Fourier space the projection is:
        Û_proj_i(k) = Û_i(k) - k_i (k_j Û_j(k)) / |k|²

    which removes the irrotational component mode-by-mode.

    Parameters
    ----------
    U : ndarray
        Velocity field, shape (Nz, Ny, Nx, 3)
    dx, dy, dz : float
        Grid spacings

    Returns
    -------
    U_proj : ndarray
        Spectrally divergence-free velocity field, shape (Nz, Ny, Nx, 3)
    p_corr : ndarray
        Pressure correction field, shape (Nz, Ny, Nx)
    """
    Nz, Ny, Nx = U.shape[:3]

    # True spectral wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

    k2 = KX**2 + KY**2 + KZ**2

    # FFT each velocity component
    Ux_hat = np.fft.fftn(U[..., 0])
    Uy_hat = np.fft.fftn(U[..., 1])
    Uz_hat = np.fft.fftn(U[..., 2])

    # Spectral divergence: k · Û
    divU_hat = KX * Ux_hat + KY * Uy_hat + KZ * Uz_hat

    # Pressure correction in Fourier space: p̂ = divU_hat / |k|²
    # (from Δp = ∇·U → -|k|² p̂ = i k · Û_hat ... but we work with
    #  the projection formula directly)
    mask = k2 > 0

    # Leray projector: Û_proj_i = Û_i - k_i (k · Û) / |k|²
    Kcomps = [KX, KY, KZ]
    U_hats = [Ux_hat, Uy_hat, Uz_hat]

    U_proj = np.zeros_like(U)
    for c in range(3):
        proj_hat = U_hats[c].copy()
        proj_hat[mask] -= Kcomps[c][mask] * divU_hat[mask] / k2[mask]
        U_proj[..., c] = np.fft.ifftn(proj_hat).real

    # Also return a pressure correction for interface compatibility
    p_hat = np.zeros_like(Ux_hat)
    # p̂ = (i k · Û) / |k|²  →  but divU_hat = i k · Û already contains i,
    # Actually: ∇·U in Fourier is i(kx Ûx + ky Ûy + kz Ûz), but we computed
    # divU_hat = kx Ûx + ky Ûy + kz Ûz (without the i), so:
    # Δp = ∇·U  →  -|k|² p̂ = i * divU_hat
    # → p̂ = -i * divU_hat / |k|²
    p_hat[mask] = -1j * divU_hat[mask] / k2[mask]
    p_corr = np.fft.ifftn(p_hat).real

    return U_proj, p_corr
