import numpy as np

from les.grid import Grid2D
from les.filter import gaussian_kernel_2d, stencil_half_width_xy


def apply_spatial_filter(
    grid: Grid2D,
    U: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply a truncated Gaussian spatial filter to a vector field.
    """
    _validate_vector_field(grid, U, "U")

    U_bar = grid.zeros_vector()

    hx, hy = stencil_half_width_xy(
        dx=grid.dx,
        dy=grid.dy,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    radius2 = (cutoff_sigma * sigma) ** 2

    for j in range(grid.Ny):
        for i in range(grid.Nx):
            i_min = max(0, i - hx)
            i_max = min(grid.Nx - 1, i + hx)
            j_min = max(0, j - hy)
            j_max = min(grid.Ny - 1, j + hy)

            xs = grid.x[i_min:i_max + 1]
            ys = grid.y[j_min:j_max + 1]

            Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
            DX = Xs - grid.X[j, i]
            DY = Ys - grid.Y[j, i]

            K = gaussian_kernel_2d(DX, DY, sigma=sigma)
            mask = DX * DX + DY * DY <= radius2
            K = np.where(mask, K, 0.0)

            if normalize:
                mass = np.sum(K) * grid.cell_area
                if mass > 0.0:
                    K = K / mass

            patch = U[j_min:j_max + 1, i_min:i_max + 1, :]
            U_bar[j, i, 0] = np.sum(K * patch[..., 0]) * grid.cell_area
            U_bar[j, i, 1] = np.sum(K * patch[..., 1]) * grid.cell_area

    return U_bar


def apply_spatial_filter_scalar(
    grid: Grid2D,
    field: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply a truncated Gaussian spatial filter to a scalar field.
    """
    _validate_scalar_field(grid, field, "field")

    field_bar = grid.zeros_scalar()

    hx, hy = stencil_half_width_xy(
        dx=grid.dx,
        dy=grid.dy,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
    )

    radius2 = (cutoff_sigma * sigma) ** 2

    for j in range(grid.Ny):
        for i in range(grid.Nx):
            i_min = max(0, i - hx)
            i_max = min(grid.Nx - 1, i + hx)
            j_min = max(0, j - hy)
            j_max = min(grid.Ny - 1, j + hy)

            xs = grid.x[i_min:i_max + 1]
            ys = grid.y[j_min:j_max + 1]

            Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
            DX = Xs - grid.X[j, i]
            DY = Ys - grid.Y[j, i]

            K = gaussian_kernel_2d(DX, DY, sigma=sigma)
            mask = DX * DX + DY * DY <= radius2
            K = np.where(mask, K, 0.0)

            if normalize:
                mass = np.sum(K) * grid.cell_area
                if mass > 0.0:
                    K = K / mass

            patch = field[j_min:j_max + 1, i_min:i_max + 1]
            field_bar[j, i] = np.sum(K * patch) * grid.cell_area

    return field_bar


def filtered_velocity(
    grid: Grid2D,
    U: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
) -> np.ndarray:
    """
    Convenience wrapper for filtered velocity.
    """
    return apply_spatial_filter(
        grid=grid,
        U=U,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
        normalize=True,
    )


def filtered_force(
    grid: Grid2D,
    F: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
) -> np.ndarray:
    """
    Convenience wrapper for filtered forcing.
    """
    return apply_spatial_filter(
        grid=grid,
        U=F,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
        normalize=True,
    )


def filtered_pressure_gradient(
    grid: Grid2D,
    grad_p: np.ndarray,
    sigma: float,
    cutoff_sigma: float,
) -> np.ndarray:
    """
    Convenience wrapper for filtered pressure gradient.
    """
    return apply_spatial_filter(
        grid=grid,
        U=grad_p,
        sigma=sigma,
        cutoff_sigma=cutoff_sigma,
        normalize=True,
    )


def _validate_scalar_field(grid: Grid2D, field: np.ndarray, name: str) -> None:
    if field.shape != grid.shape:
        raise ValueError(f"Expected {name} to have shape {grid.shape}.")


def _validate_vector_field(grid: Grid2D, field: np.ndarray, name: str) -> None:
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")