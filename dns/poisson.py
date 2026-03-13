import numpy as np


def fft_wave_numbers(n: int, d: float) -> np.ndarray:
    """
    Fourier wave numbers for an evenly spaced grid.
    """
    return 2.0 * np.pi * np.fft.fftfreq(n, d=d)


def solve_poisson_periodic(rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve
        Delta phi = rhs
    on a periodic box using FFTs.

    The zero mode is set to zero.
    """
    if rhs.ndim != 2:
        raise ValueError("Expected rhs with shape (Ny, Nx).")

    ny, nx = rhs.shape

    kx = fft_wave_numbers(nx, dx)
    ky = fft_wave_numbers(ny, dy)

    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k2 = KX * KX + KY * KY

    rhs_hat = np.fft.fft2(rhs)
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    mask = k2 > 0.0
    phi_hat[mask] = -rhs_hat[mask] / k2[mask]
    phi_hat[~mask] = 0.0

    phi = np.fft.ifft2(phi_hat).real
    return phi


def project_velocity(U: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a velocity field onto divergence-free fields.

    Returns:
        U_proj
        p_corr

    where p_corr solves
        Delta p_corr = div(U)
    and
        U_proj = U - grad(p_corr)
    """
    _validate_vector_field(U)

    ux = U[..., 0]
    uy = U[..., 1]

    dux_dx = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dx)
    duy_dy = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dy)
    divU = dux_dx + duy_dy

    p_corr = solve_poisson_periodic(divU, dx=dx, dy=dy)

    dp_dx = (np.roll(p_corr, -1, axis=1) - np.roll(p_corr, 1, axis=1)) / (2.0 * dx)
    dp_dy = (np.roll(p_corr, -1, axis=0) - np.roll(p_corr, 1, axis=0)) / (2.0 * dy)

    U_proj = np.zeros_like(U)
    U_proj[..., 0] = ux - dp_dx
    U_proj[..., 1] = uy - dp_dy

    return U_proj, p_corr


def _validate_vector_field(U: np.ndarray) -> None:
    if U.ndim != 3 or U.shape[-1] != 2:
        raise ValueError("Expected vector field with shape (Ny, Nx, 2).")