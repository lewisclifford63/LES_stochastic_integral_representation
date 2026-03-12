import numpy as np

from les.grid import Grid2D


def ddx_scalar(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Central-difference x-derivative of a scalar field.

    Uses periodic-style differences via np.roll. This is acceptable for the
    padded-box approximation provided diagnostics are interpreted primarily on
    the trusted interior.
    """
    _validate_scalar_field(field)
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx)


def ddy_scalar(field: np.ndarray, dy: float) -> np.ndarray:
    """
    Central-difference y-derivative of a scalar field.
    """
    _validate_scalar_field(field)
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dy)


def gradient_scalar(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Gradient of a scalar field.

    Returns
    -------
    grad : ndarray, shape (Ny, Nx, 2)
        grad[..., 0] = d(field)/dx
        grad[..., 1] = d(field)/dy
    """
    _validate_scalar_field(field)

    grad = np.zeros(field.shape + (2,), dtype=np.float64)
    grad[..., 0] = ddx_scalar(field, dx)
    grad[..., 1] = ddy_scalar(field, dy)
    return grad


def divergence(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Divergence of a vector field U[..., 0], U[..., 1].

    Returns
    -------
    divU : ndarray, shape (Ny, Nx)
    """
    _validate_vector_field(U)

    ux = U[..., 0]
    uy = U[..., 1]

    dux_dx = ddx_scalar(ux, dx)
    duy_dy = ddy_scalar(uy, dy)
    return dux_dx + duy_dy


def grad_velocity(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Velocity gradient tensor field.

    For velocity U = (u_1, u_2), returns gradU with components
        gradU[..., i, j] = d u_i / d x_j

    where:
        j = 0 means x-derivative
        j = 1 means y-derivative

    Returns
    -------
    gradU : ndarray, shape (Ny, Nx, 2, 2)
    """
    _validate_vector_field(U)

    gradU = np.zeros(U.shape[:2] + (2, 2), dtype=np.float64)

    gradU[..., 0, 0] = ddx_scalar(U[..., 0], dx)  # du1/dx
    gradU[..., 0, 1] = ddy_scalar(U[..., 0], dy)  # du1/dy
    gradU[..., 1, 0] = ddx_scalar(U[..., 1], dx)  # du2/dx
    gradU[..., 1, 1] = ddy_scalar(U[..., 1], dy)  # du2/dy

    return gradU


def symmetric_grad_velocity(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Symmetric part of the velocity gradient:
        S = 0.5 * (gradU + gradU^T)

    Returns
    -------
    S : ndarray, shape (Ny, Nx, 2, 2)
    """
    gradU = grad_velocity(U, dx, dy)
    return 0.5 * (gradU + np.swapaxes(gradU, -1, -2))


def convective_term(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the convective term (U · ∇)U.

    Returns
    -------
    C : ndarray, shape (Ny, Nx, 2)
        C[..., i] = U_j * dU_i/dx_j
    """
    _validate_vector_field(U)

    gradU = grad_velocity(U, dx, dy)
    C = np.zeros_like(U)

    # C_i = sum_j U_j * dU_i/dx_j
    C[..., 0] = U[..., 0] * gradU[..., 0, 0] + U[..., 1] * gradU[..., 0, 1]
    C[..., 1] = U[..., 0] * gradU[..., 1, 0] + U[..., 1] * gradU[..., 1, 1]

    return C


def laplacian_scalar(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    2D Laplacian of a scalar field.
    """
    _validate_scalar_field(field)

    d2x = (
        np.roll(field, -1, axis=1)
        - 2.0 * field
        + np.roll(field, 1, axis=1)
    ) / (dx * dx)

    d2y = (
        np.roll(field, -1, axis=0)
        - 2.0 * field
        + np.roll(field, 1, axis=0)
    ) / (dy * dy)

    return d2x + d2y


def laplacian_vector(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Componentwise Laplacian of a vector field.

    Returns
    -------
    LU : ndarray, shape (Ny, Nx, 2)
    """
    _validate_vector_field(U)

    LU = np.zeros_like(U)
    LU[..., 0] = laplacian_scalar(U[..., 0], dx, dy)
    LU[..., 1] = laplacian_scalar(U[..., 1], dx, dy)
    return LU


def frobenius_inner(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pointwise Frobenius inner product of two tensor fields.

    Parameters
    ----------
    A, B : ndarray, shape (Ny, Nx, 2, 2)

    Returns
    -------
    inner : ndarray, shape (Ny, Nx)
        inner = sum_{i,j} A_ij * B_ij
    """
    _validate_tensor_field(A)
    _validate_tensor_field(B)

    return np.sum(A * B, axis=(-2, -1))


def matrix_trace(A: np.ndarray) -> np.ndarray:
    """
    Pointwise trace of a 2x2 tensor field.

    Parameters
    ----------
    A : ndarray, shape (Ny, Nx, 2, 2)

    Returns
    -------
    trA : ndarray, shape (Ny, Nx)
    """
    _validate_tensor_field(A)
    return A[..., 0, 0] + A[..., 1, 1]


def double_contraction_gradU(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the scalar field
        sum_{i,j} (du_j/dx_i) (du_i/dx_j)

    This is exactly the term appearing in the whole-space pressure Poisson
    source for incompressible Navier-Stokes:
        Δp = -∂_i u_j ∂_j u_i + div(F)

    Returns
    -------
    contraction : ndarray, shape (Ny, Nx)
    """
    _validate_vector_field(U)

    gradU = grad_velocity(U, dx, dy)

    du1_dx = gradU[..., 0, 0]
    du1_dy = gradU[..., 0, 1]
    du2_dx = gradU[..., 1, 0]
    du2_dy = gradU[..., 1, 1]

    return (
        du1_dx * du1_dx
        + du2_dy * du2_dy
        + 2.0 * du2_dx * du1_dy
    )


def trusted_divergence_stats(U: np.ndarray, grid: Grid2D) -> tuple[float, float]:
    """
    Max and mean absolute divergence on the trusted interior.
    """
    divU = divergence(U, grid.dx, grid.dy)
    div_trusted = grid.restrict_to_trusted(divU)
    return (
        float(np.max(np.abs(div_trusted))),
        float(np.mean(np.abs(div_trusted))),
    )


def _validate_scalar_field(field: np.ndarray) -> None:
    if field.ndim != 2:
        raise ValueError(f"Expected scalar field with shape (Ny, Nx), got {field.shape}.")


def _validate_vector_field(field: np.ndarray) -> None:
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(
            f"Expected vector field with shape (Ny, Nx, 2), got {field.shape}."
        )


def _validate_tensor_field(field: np.ndarray) -> None:
    if field.ndim != 4 or field.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected tensor field with shape (Ny, Nx, 2, 2), got {field.shape}."
        )