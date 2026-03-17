"""
Pressure solver module for the 3D random LES method.

Implements the whole-space pressure Poisson equation and associated integral
representation needed for the stochastic integral method (Qian et al. 2025).

The pressure source is:
    source = -sum_{i,j} (du_j/dx_i)(du_i/dx_j) + div(F)

Two implementations of the pressure gradient are provided:

1. compute_pressure_gradient_fft  (RECOMMENDED — O(N³ log N))
   Uses the spectral equivalence of the 3D free-space Green's function.
   In Fourier space the kernel K(r) = (1/4π) r/|r|³ becomes:
       FFT[K_i] = i k_i / |k|²
   so ∇p = IFFT(i k · p̂) where p̂ = -source_hat / |k|².
   Implemented via solve_poisson_periodic_3d from dns.poisson,
   followed by a central-difference spectral gradient.  This is the
   padded-box periodic approximation to the whole-space kernel, and is
   exactly the representation used in the Qian et al. (2025) dissertation.

2. pressure_gradient_from_source  (reference — O(N^6), for small N only)
   Direct quadrature loop over all source–evaluation point pairs.
   Retained for scientific fidelity to the 2D reference code.
"""

import numpy as np

from les.grid import CartesianGrid3D
from les.differential_operators import divergence, double_contraction_gradU


def pressure_source(
    U: np.ndarray,
    F: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """
    Compute the pressure Poisson source in 3D.

    The source for the incompressible Navier-Stokes pressure Poisson equation:
        Δp = -sum_{i,j} (du_j/dx_i)(du_i/dx_j) + div(F)

    Parameters
    ----------
    U : ndarray, shape (Nz, Ny, Nx, 3)
        Velocity field with components [u_x, u_y, u_z].
    F : ndarray, shape (Nz, Ny, Nx, 3)
        Forcing field with components [f_x, f_y, f_z].
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    dz : float
        Grid spacing in z-direction.

    Returns
    -------
    source : ndarray, shape (Nz, Ny, Nx)
        Scalar pressure source field.
    """
    _validate_vector_field(U, "U")
    _validate_vector_field(F, "F")

    nonlinear_term = double_contraction_gradU(U, dx, dy, dz)
    forcing_div = divergence(F, dx, dy, dz)
    return -nonlinear_term + forcing_div


def pressure_kernel_gradient_3d(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    softening: float = 1.0e-10,
) -> np.ndarray:
    """
    3D Green's function gradient for the whole-space pressure Poisson equation.

    The pressure satisfies:
        ∇²p = source

    The solution via whole-space Green's function is:
        p(x) = (1/(4π)) ∫ source(y) / |x - y| dy

    The pressure gradient is:
        ∇p(x) = (1/(4π)) ∫ source(y) * (x - y) / |x - y|³ dy

    This function computes the kernel:
        K(dx, dy, dz) = (1/(4π)) * (dx, dy, dz) / (dx² + dy² + dz²)^{3/2}

    Parameters
    ----------
    dx : ndarray
        x-component of displacement vectors. Shape can be arbitrary but all
        three inputs must broadcast together.
    dy : ndarray
        y-component of displacement vectors.
    dz : ndarray
        z-component of displacement vectors.
    softening : float, optional
        Small regularization to avoid singularity: r² = dx² + dy² + dz² + softening.
        Default 1.0e-10.

    Returns
    -------
    K : ndarray
        Kernel gradient field with shape (..., 3).
        K[..., 0] = K_x component
        K[..., 1] = K_y component
        K[..., 2] = K_z component

    Notes
    -----
    The softening parameter ensures numerical stability at small separations.
    The factor 1/(4π) is the correct normalization for 3D free space.
    """
    r2 = dx * dx + dy * dy + dz * dz + softening
    r3 = np.power(r2, 1.5)
    prefactor = 1.0 / (4.0 * np.pi)

    K = np.zeros(dx.shape + (3,), dtype=np.float64)
    K[..., 0] = prefactor * dx / r3
    K[..., 1] = prefactor * dy / r3
    K[..., 2] = prefactor * dz / r3
    return K


def pressure_gradient_from_source(
    grid: CartesianGrid3D,
    source: np.ndarray,
    cutoff_radius: float | None = None,
    softening: float = 1.0e-10,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Compute grad p on the 3D grid from the source field by direct quadrature.

    This function evaluates:
        ∇p(x_eval) = (1/(4π)) * sum_{source points} source(y) * (x_eval - y) / |x_eval - y|³ * dV

    where dV = dx * dy * dz is the cell volume and the sum is over all grid
    points with optional cutoff and self-exclusion.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    source : ndarray, shape (Nz, Ny, Nx)
        Scalar pressure source field.
    cutoff_radius : float, optional
        If provided, only include source points within this radius of the
        evaluation point.
    softening : float, optional
        Softening parameter to avoid singularities. Default 1.0e-10.
    exclude_self : bool, optional
        If True, exclude the self-cell (evaluation point's own cell).
        Default True.

    Returns
    -------
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        Pressure gradient field with components [∂p/∂x, ∂p/∂y, ∂p/∂z].

    Notes
    -----
    COMPUTATIONAL COST WARNING: This implementation uses a triple nested loop
    over all grid points (as evaluation locations) and quadrature over all
    source grid points. The computational complexity is O((Nz*Ny*Nx)²), which
    becomes prohibitively expensive for large 3D grids (e.g., 128³ or larger).

    For production large-scale 3D simulations, consider:
    - Fast multipole method (FMM)
    - Fourier spectral methods with periodic boundaries
    - Multigrid preconditioners for iterative solvers
    - Domain decomposition with far-field approximations

    This direct quadrature is retained for scientific accuracy and fidelity
    to the 2D reference implementation, which uses the same pattern.
    """
    _validate_scalar_field(grid, source, "source")

    grad_p = np.zeros(grid.vector_shape, dtype=np.float64)
    cell_volume = grid.cell_volume

    source_flat = source.ravel()
    x_src = grid.X.ravel()
    y_src = grid.Y.ravel()
    z_src = grid.Z.ravel()

    # Triple loop over evaluation points
    for k in range(grid.Nz):
        for j in range(grid.Ny):
            for i in range(grid.Nx):
                x_eval = grid.X[k, j, i]
                y_eval = grid.Y[k, j, i]
                z_eval = grid.Z[k, j, i]

                dx = x_eval - x_src
                dy = y_eval - y_src
                dz = z_eval - z_src

                if cutoff_radius is None:
                    mask = np.ones_like(source_flat, dtype=bool)
                else:
                    mask = dx * dx + dy * dy + dz * dz <= cutoff_radius * cutoff_radius

                if exclude_self:
                    self_mask = ~(
                        (np.abs(dx) < 0.5 * grid.dx)
                        & (np.abs(dy) < 0.5 * grid.dy)
                        & (np.abs(dz) < 0.5 * grid.dz)
                    )
                    mask = mask & self_mask

                if not np.any(mask):
                    continue

                K = pressure_kernel_gradient_3d(
                    dx[mask],
                    dy[mask],
                    dz[mask],
                    softening=softening,
                )

                weighted_source = source_flat[mask] * cell_volume

                grad_p[k, j, i, 0] = np.sum(K[..., 0] * weighted_source)
                grad_p[k, j, i, 1] = np.sum(K[..., 1] * weighted_source)
                grad_p[k, j, i, 2] = np.sum(K[..., 2] * weighted_source)

    return grad_p


def compute_pressure_gradient(
    grid: CartesianGrid3D,
    U: np.ndarray,
    F: np.ndarray,
    cutoff_radius: float | None = None,
    softening: float = 1.0e-10,
) -> np.ndarray:
    """
    Compute grad p from U and F in 3D.

    Computes the pressure source, then the pressure gradient via direct
    quadrature.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    U : ndarray, shape (Nz, Ny, Nx, 3)
        Velocity field.
    F : ndarray, shape (Nz, Ny, Nx, 3)
        Forcing field.
    cutoff_radius : float, optional
        Cutoff for quadrature.
    softening : float, optional
        Softening parameter.

    Returns
    -------
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        Pressure gradient.
    """
    source = pressure_source(U, F, grid.dx, grid.dy, grid.dz)
    return pressure_gradient_from_source(
        grid=grid,
        source=source,
        cutoff_radius=cutoff_radius,
        softening=softening,
        exclude_self=True,
    )


def compute_pressure_gradient_fft(
    grid: CartesianGrid3D,
    U: np.ndarray,
    F: np.ndarray,
    source: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Compute ∇p via the spectral K-kernel convolution (Qian et al. 2025).

    Implements the Fourier representation of the Biot-Savart / Green's
    function kernel K_d (Qian Lemma 3, eq. 28-29):

        ∇p(x) = ∫ K_d(x, y) Δp(y) dy       (physical space)
        (∇p)_i(k) = -i k_i ŝ(k) / |k|²     (Fourier space)

    where ŝ = FFT(source) and source = Δp = -∂ᵢuⱼ ∂ⱼuᵢ + ∇·F.

    This is the **direct spectral K-kernel** approach: the pressure gradient
    is computed in a single spectral step using TRUE wavenumbers k_i (not
    modified wavenumbers).  No intermediate scalar pressure field p is
    formed and no finite-difference gradient is taken.  This ensures the
    mapping source → ∇p is exactly the Green's function convolution that
    the Qian integral representation theory is built upon.

    The previous implementation solved Δp = source with modified
    wavenumbers (sin(k·h)/h)² and then differentiated p via central
    differences.  While self-consistent in the finite-difference world,
    this is NOT the true K-kernel and introduces transfer-function errors
    of order sin(kh)/(kh) ≈ 0.64 at Nyquist, breaking the theoretical
    guarantee that g = -∇p + F is nearly divergence-free.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    U : ndarray, shape (Nz, Ny, Nx, 3)
        Velocity field.
    F : ndarray, shape (Nz, Ny, Nx, 3)
        Forcing field.
    source : ndarray, shape (Nz, Ny, Nx), optional
        Pre-computed pressure Poisson source.  If None, it is computed
        from U and F via ``pressure_source``.

    Returns
    -------
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        Pressure gradient [∂p/∂x, ∂p/∂y, ∂p/∂z].
    """
    _validate_vector_field(U, "U")
    _validate_vector_field(F, "F")

    # Step 1: pressure Poisson source  Δp = source
    if source is None:
        source = pressure_source(U, F, grid.dx, grid.dy, grid.dz)

    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx

    # Step 2: TRUE spectral wavenumbers (NOT modified wavenumbers)
    # k = 2π n / L  for n in fftfreq ordering
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=grid.dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=grid.dz)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

    # True spectral Laplacian: |k|² = kx² + ky² + kz²
    k2 = KX**2 + KY**2 + KZ**2

    # Step 3: FFT of source
    source_hat = np.fft.fftn(source)

    # Step 4: Direct K-kernel in Fourier space (Qian eq. 44 / Lemma 3)
    #   Poisson:  -|k|² p̂ = ŝ   →   p̂ = -ŝ / |k|²
    #   Gradient:  (∇p)_i = ik_i p̂ = -ik_i ŝ / |k|²
    # Avoid division by zero at k = 0 (mean mode); set to zero (arbitrary
    # constant in p does not affect ∇p).
    mask = k2 > 0

    grad_p = np.zeros(grid.vector_shape, dtype=np.float64)
    for c, Kc in enumerate([KX, KY, KZ]):
        grad_hat = np.zeros_like(source_hat)
        grad_hat[mask] = -1j * Kc[mask] * source_hat[mask] / k2[mask]
        grad_p[..., c] = np.fft.ifftn(grad_hat).real

    return grad_p


def compute_g_field_fft(
    grid: CartesianGrid3D,
    U: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Return source, grad_p, and g = -grad_p + F via spectral K-kernel.

    This is the recommended fast version of compute_g_field for use in
    the Qian et al. (2025) accumulated-history LES loop.  Uses the direct
    spectral K-kernel convolution (Qian eq. 44) to compute ∇p, ensuring
    the pressure gradient is the true Green's function result.

    Parameters
    ----------
    grid : CartesianGrid3D
    U : ndarray, shape (Nz, Ny, Nx, 3) — velocity field
    F : ndarray, shape (Nz, Ny, Nx, 3) — forcing field

    Returns
    -------
    source : ndarray, shape (Nz, Ny, Nx)
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
    g      : ndarray, shape (Nz, Ny, Nx, 3)   g = -grad_p + F
    """
    # Compute source once and pass it through to avoid redundant work
    source = pressure_source(U, F, grid.dx, grid.dy, grid.dz)
    grad_p = compute_pressure_gradient_fft(grid, U, F, source=source)
    g = -grad_p + F
    return source, grad_p, g


def compute_g_field(
    grid: CartesianGrid3D,
    U: np.ndarray,
    F: np.ndarray,
    cutoff_radius: float | None = None,
    softening: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return source, grad_p, and g = -grad_p + F in 3D.

    The drift term in the stochastic representation is:
        G = -∇p + F

    where ∇p is the pressure gradient from the nonlinear Poisson source
    and F is the external forcing.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    U : ndarray, shape (Nz, Ny, Nx, 3)
        Velocity field.
    F : ndarray, shape (Nz, Ny, Nx, 3)
        Forcing field.
    cutoff_radius : float, optional
        Cutoff for quadrature.
    softening : float, optional
        Softening parameter.

    Returns
    -------
    source : ndarray, shape (Nz, Ny, Nx)
        Pressure Poisson source.
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        Pressure gradient.
    g : ndarray, shape (Nz, Ny, Nx, 3)
        The field g = -grad_p + F used as drift in particle evolution.
    """
    source = pressure_source(U, F, grid.dx, grid.dy, grid.dz)

    grad_p = pressure_gradient_from_source(
        grid=grid,
        source=source,
        cutoff_radius=cutoff_radius,
        softening=softening,
        exclude_self=True,
    )

    g = -grad_p + F
    return source, grad_p, g


def trusted_pressure_gradient_norms(
    grid: CartesianGrid3D,
    grad_p: np.ndarray,
) -> tuple[float, float]:
    """
    Max and mean |grad p| on the trusted region.

    Computes the magnitude of the pressure gradient and evaluates
    statistics on the trusted interior.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid.
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        Pressure gradient field.

    Returns
    -------
    max_mag : float
        Maximum magnitude of grad p on trusted region.
    mean_mag : float
        Mean magnitude of grad p on trusted region.
    """
    _validate_vector_field(grad_p, "grad_p")

    mag = np.sqrt(
        grad_p[..., 0] ** 2 + grad_p[..., 1] ** 2 + grad_p[..., 2] ** 2
    )
    mag_trusted = grid.restrict_to_trusted(mag)

    max_mag = float(np.max(mag_trusted))
    mean_mag = float(np.mean(mag_trusted))
    return max_mag, mean_mag


# ------------------------------------------------------------------ #
#  Validation helpers
# ------------------------------------------------------------------ #

def _validate_scalar_field(
    grid: CartesianGrid3D, field: np.ndarray, name: str
) -> None:
    """Validate that field has shape (Nz, Ny, Nx)."""
    if field.shape != grid.shape:
        raise ValueError(f"Expected {name} to have shape {grid.shape}.")


def _validate_vector_field(field: np.ndarray, name: str) -> None:
    """Validate that field has shape (Nz, Ny, Nx, 3)."""
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(
            f"Expected {name} to have shape (Nz, Ny, Nx, 3), got {field.shape}."
        )
