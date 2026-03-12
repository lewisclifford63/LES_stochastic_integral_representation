import numpy as np

from les.grid import Grid2D
from les.differential_operators import divergence, double_contraction_gradU


def pressure_source(
    U: np.ndarray,
    F: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Compute the pressure Poisson source:
        source = -sum_{i,j} (du_j/dx_i)(du_i/dx_j) + div(F)
    """
    _validate_vector_field(U, "U")
    _validate_vector_field(F, "F")

    nonlinear_term = double_contraction_gradU(U, dx, dy)
    forcing_div = divergence(F, dx, dy)
    return -nonlinear_term + forcing_div


def pressure_kernel_gradient_2d(
    dx: np.ndarray,
    dy: np.ndarray,
    softening: float = 1.0e-10,
) -> np.ndarray:
    """
    2D kernel for grad p:
        K(dx, dy) = (1 / 2pi) * (dx, dy) / (dx^2 + dy^2)
    """
    r2 = dx * dx + dy * dy + softening
    prefactor = 1.0 / (2.0 * np.pi)

    K = np.zeros(dx.shape + (2,), dtype=np.float64)
    K[..., 0] = prefactor * dx / r2
    K[..., 1] = prefactor * dy / r2
    return K


def pressure_gradient_from_source(
    grid: Grid2D,
    source: np.ndarray,
    cutoff_radius: float | None = None,
    softening: float = 1.0e-10,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Compute grad p on the grid from the source field by direct quadrature.
    """
    _validate_scalar_field(grid, source, "source")

    grad_p = grid.zeros_vector()
    cell_area = grid.cell_area

    source_flat = source.ravel()
    x_src = grid.X.ravel()
    y_src = grid.Y.ravel()

    for j in range(grid.Ny):
        for i in range(grid.Nx):
            x_eval = grid.X[j, i]
            y_eval = grid.Y[j, i]

            dx = x_eval - x_src
            dy = y_eval - y_src

            if cutoff_radius is None:
                mask = np.ones_like(source_flat, dtype=bool)
            else:
                mask = dx * dx + dy * dy <= cutoff_radius * cutoff_radius

            if exclude_self:
                self_mask = ~(
                    (np.abs(dx) < 0.5 * grid.dx) & (np.abs(dy) < 0.5 * grid.dy)
                )
                mask = mask & self_mask

            if not np.any(mask):
                continue

            K = pressure_kernel_gradient_2d(
                dx[mask],
                dy[mask],
                softening=softening,
            )

            weighted_source = source_flat[mask] * cell_area

            grad_p[j, i, 0] = np.sum(K[..., 0] * weighted_source)
            grad_p[j, i, 1] = np.sum(K[..., 1] * weighted_source)

    return grad_p


def compute_pressure_gradient(
    grid: Grid2D,
    U: np.ndarray,
    F: np.ndarray,
    cutoff_radius: float | None = None,
    softening: float = 1.0e-10,
) -> np.ndarray:
    """
    Compute grad p from U and F.
    """
    source = pressure_source(U, F, grid.dx, grid.dy)
    return pressure_gradient_from_source(
        grid=grid,
        source=source,
        cutoff_radius=cutoff_radius,
        softening=softening,
        exclude_self=True,
    )


def compute_g_field(
    grid: Grid2D,
    U: np.ndarray,
    F: np.ndarray,
    cutoff_radius: float | None = None,
    softening: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return source, grad_p, and g = -grad_p + F.
    """
    source = pressure_source(U, F, grid.dx, grid.dy)

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
    grid: Grid2D,
    grad_p: np.ndarray,
) -> tuple[float, float]:
    """
    Max and mean |grad p| on the trusted region.
    """
    _validate_vector_field(grad_p, "grad_p")

    mag = np.sqrt(grad_p[..., 0] ** 2 + grad_p[..., 1] ** 2)
    mag_trusted = grid.restrict_to_trusted(mag)

    max_mag = float(np.max(mag_trusted))
    mean_mag = float(np.mean(mag_trusted))
    return max_mag, mean_mag


def _validate_scalar_field(grid: Grid2D, field: np.ndarray, name: str) -> None:
    if field.shape != grid.shape:
        raise ValueError(f"Expected {name} to have shape {grid.shape}.")


def _validate_vector_field(field: np.ndarray, name: str) -> None:
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(f"Expected {name} to have shape (Ny, Nx, 2).")