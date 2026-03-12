import numpy as np

from les.grid import Grid2D
from les.filter import gaussian_kernel_2d, stencil_half_width_xy


def reconstruct_velocity_from_particles(
    grid: Grid2D,
    positions: np.ndarray,
    amplitudes: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    cutoff_sigma: float = 3.0,
) -> np.ndarray:
    """
    Reconstruct Eulerian velocity from particle data via truncated Gaussian deposition.

    For each particle p:
        U(x_m) += chi(x_m - X_p) * A_p * w_p

    Notes
    -----
    - This is a local truncated reconstruction for efficiency.
    - It is designed for the padded-box setting.
    - Contributions outside the computational box are ignored.
    - The trusted interior should be used for diagnostics and assessment.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"Expected positions.shape = (Np, 2), got {positions.shape}."
        )
    if amplitudes.shape != positions.shape:
        raise ValueError(
            f"Expected amplitudes.shape == {positions.shape}, got {amplitudes.shape}."
        )
    if weights.shape != (positions.shape[0],):
        raise ValueError(
            f"Expected weights.shape == ({positions.shape[0]},), got {weights.shape}."
        )

    U_rec = np.zeros((grid.Ny, grid.Nx, 2), dtype=np.float64)

    hx = grid.dx
    hy = grid.dy
    half_width_x, half_width_y = stencil_half_width_xy(
        dx=hx,
        dy=hy,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    x0 = grid.x[0]
    y0 = grid.y[0]

    for p in range(positions.shape[0]):
        xp = positions[p, 0]
        yp = positions[p, 1]
        ap0 = amplitudes[p, 0]
        ap1 = amplitudes[p, 1]
        wp = weights[p]

        # Nearest grid index in the full padded box
        i_center = int(np.round((xp - x0) / hx))
        j_center = int(np.round((yp - y0) / hy))

        i_min = max(0, i_center - half_width_x)
        i_max = min(grid.Nx - 1, i_center + half_width_x)
        j_min = max(0, j_center - half_width_y)
        j_max = min(grid.Ny - 1, j_center + half_width_y)

        xs = grid.x[i_min:i_max + 1]
        ys = grid.y[j_min:j_max + 1]

        Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
        K = gaussian_kernel_2d(Xs - xp, Ys - yp, sigma=sigma)

        radius2 = (cutoff_sigma * sigma) ** 2
        mask = (Xs - xp) ** 2 + (Ys - yp) ** 2 <= radius2
        K = np.where(mask, K, 0.0)

        U_rec[j_min:j_max + 1, i_min:i_max + 1, 0] += wp * K * ap0
        U_rec[j_min:j_max + 1, i_min:i_max + 1, 1] += wp * K * ap1

    return U_rec


def reconstruct_velocity_from_particles_trusted(
    grid: Grid2D,
    positions: np.ndarray,
    amplitudes: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    cutoff_sigma: float = 3.0,
) -> np.ndarray:
    """
    Reconstruct on the full padded box, then restrict to the trusted interior.
    """
    U_rec = reconstruct_velocity_from_particles(
        grid=grid,
        positions=positions,
        amplitudes=amplitudes,
        weights=weights,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )
    return grid.restrict_to_trusted(U_rec)


def reconstruction_error(U_ref: np.ndarray, U_rec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    err : ndarray, shape (..., 2)
        Vector error field.
    err_mag : ndarray, shape (...)
        Pointwise magnitude of the error.
    """
    if U_ref.shape != U_rec.shape:
        raise ValueError(
            f"Shape mismatch: U_ref.shape = {U_ref.shape}, U_rec.shape = {U_rec.shape}."
        )

    err = U_rec - U_ref
    err_mag = np.sqrt(err[..., 0] ** 2 + err[..., 1] ** 2)
    return err, err_mag


def relative_l2_error(U_ref: np.ndarray, U_rec: np.ndarray) -> float:
    """
    Relative discrete L2 error over the full array.
    """
    if U_ref.shape != U_rec.shape:
        raise ValueError(
            f"Shape mismatch: U_ref.shape = {U_ref.shape}, U_rec.shape = {U_rec.shape}."
        )

    num = np.sqrt(np.sum((U_rec - U_ref) ** 2))
    den = np.sqrt(np.sum(U_ref ** 2))
    if den == 0.0:
        return 0.0
    return float(num / den)


def trusted_relative_l2_error(grid: Grid2D, U_ref: np.ndarray, U_rec: np.ndarray) -> float:
    """
    Relative discrete L2 error restricted to the trusted interior.
    """
    U_ref_trusted = grid.restrict_to_trusted(U_ref)
    U_rec_trusted = grid.restrict_to_trusted(U_rec)
    return relative_l2_error(U_ref_trusted, U_rec_trusted)


def trusted_reconstruction_error(
    grid: Grid2D,
    U_ref: np.ndarray,
    U_rec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruction error restricted to the trusted interior.
    """
    U_ref_trusted = grid.restrict_to_trusted(U_ref)
    U_rec_trusted = grid.restrict_to_trusted(U_rec)
    return reconstruction_error(U_ref_trusted, U_rec_trusted)