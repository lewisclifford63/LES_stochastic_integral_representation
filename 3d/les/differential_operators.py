"""
3D finite difference differential operators for LES calculations.

Provides central-difference approximations to spatial derivatives on uniform
Cartesian grids using periodic-style differences via np.roll. Operators act
componentwise on vector and tensor fields.

Axis conventions:
  - axis 0: z-direction
  - axis 1: y-direction
  - axis 2: x-direction

Field conventions:
  - Scalar fields: shape (Nz, Ny, Nx)
  - Vector fields: shape (Nz, Ny, Nx, 3) with components [u_x, u_y, u_z]
  - Tensor fields: shape (Nz, Ny, Nx, 3, 3) with gradU[..., i, j] = du_i/dx_j
"""

import numpy as np

from les.grid import CartesianGrid3D


def ddx_scalar(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Central-difference x-derivative of a scalar field.

    Uses periodic-style differences via np.roll on axis=2 (x-direction).
    This approximation is valid for the padded-box LES setup provided
    diagnostics are interpreted primarily on the trusted interior.

    Parameters
    ----------
    field
        Scalar field of shape (Nz, Ny, Nx).
    dx
        Grid spacing in x-direction.

    Returns
    -------
    dfdx : ndarray, shape (Nz, Ny, Nx)
        Central-difference approximation to df/dx.
    """
    _validate_scalar_field(field)
    return (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2.0 * dx)


def ddy_scalar(field: np.ndarray, dy: float) -> np.ndarray:
    """
    Central-difference y-derivative of a scalar field.

    Uses periodic-style differences via np.roll on axis=1 (y-direction).

    Parameters
    ----------
    field
        Scalar field of shape (Nz, Ny, Nx).
    dy
        Grid spacing in y-direction.

    Returns
    -------
    dfdy : ndarray, shape (Nz, Ny, Nx)
        Central-difference approximation to df/dy.
    """
    _validate_scalar_field(field)
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dy)


def ddz_scalar(field: np.ndarray, dz: float) -> np.ndarray:
    """
    Central-difference z-derivative of a scalar field.

    Uses periodic-style differences via np.roll on axis=0 (z-direction).

    Parameters
    ----------
    field
        Scalar field of shape (Nz, Ny, Nx).
    dz
        Grid spacing in z-direction.

    Returns
    -------
    dfdz : ndarray, shape (Nz, Ny, Nx)
        Central-difference approximation to df/dz.
    """
    _validate_scalar_field(field)
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dz)


def gradient_scalar(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Gradient of a scalar field in 3D.

    Computes the gradient vector ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z) using central
    differences on all three coordinate directions.

    Parameters
    ----------
    field
        Scalar field of shape (Nz, Ny, Nx).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    grad : ndarray, shape (Nz, Ny, Nx, 3)
        Gradient field where:
        grad[..., 0] = ∂f/∂x
        grad[..., 1] = ∂f/∂y
        grad[..., 2] = ∂f/∂z
    """
    _validate_scalar_field(field)

    grad = np.zeros(field.shape + (3,), dtype=np.float64)
    grad[..., 0] = ddx_scalar(field, dx)
    grad[..., 1] = ddy_scalar(field, dy)
    grad[..., 2] = ddz_scalar(field, dz)
    return grad


def divergence(U: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Divergence of a 3D vector field.

    Computes ∇·U = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z where U has components
    [u_x, u_y, u_z].

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    divU : ndarray, shape (Nz, Ny, Nx)
        Scalar divergence field.
    """
    _validate_vector_field(U)

    ux = U[..., 0]
    uy = U[..., 1]
    uz = U[..., 2]

    dux_dx = ddx_scalar(ux, dx)
    duy_dy = ddy_scalar(uy, dy)
    duz_dz = ddz_scalar(uz, dz)

    return dux_dx + duy_dy + duz_dz


def grad_velocity(U: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Velocity gradient tensor field in 3D.

    For velocity U = [u_x, u_y, u_z], computes all nine components of the
    gradient tensor ∇U with indices:
        gradU[..., i, j] = ∂u_i/∂x_j

    where:
        j=0 means ∂/∂x
        j=1 means ∂/∂y
        j=2 means ∂/∂z
        i=0 is u_x component
        i=1 is u_y component
        i=2 is u_z component

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    gradU : ndarray, shape (Nz, Ny, Nx, 3, 3)
        Velocity gradient tensor.
    """
    _validate_vector_field(U)

    gradU = np.zeros(U.shape[:3] + (3, 3), dtype=np.float64)

    # First row: derivatives of u_x
    gradU[..., 0, 0] = ddx_scalar(U[..., 0], dx)  # ∂u_x/∂x
    gradU[..., 0, 1] = ddy_scalar(U[..., 0], dy)  # ∂u_x/∂y
    gradU[..., 0, 2] = ddz_scalar(U[..., 0], dz)  # ∂u_x/∂z

    # Second row: derivatives of u_y
    gradU[..., 1, 0] = ddx_scalar(U[..., 1], dx)  # ∂u_y/∂x
    gradU[..., 1, 1] = ddy_scalar(U[..., 1], dy)  # ∂u_y/∂y
    gradU[..., 1, 2] = ddz_scalar(U[..., 1], dz)  # ∂u_y/∂z

    # Third row: derivatives of u_z
    gradU[..., 2, 0] = ddx_scalar(U[..., 2], dx)  # ∂u_z/∂x
    gradU[..., 2, 1] = ddy_scalar(U[..., 2], dy)  # ∂u_z/∂y
    gradU[..., 2, 2] = ddz_scalar(U[..., 2], dz)  # ∂u_z/∂z

    return gradU


def symmetric_grad_velocity(
    U: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    """
    Symmetric part of the velocity gradient tensor.

    Computes the strain-rate tensor:
        S = 0.5 * (∇U + (∇U)^T) = 0.5 * (gradU + gradU^T)

    This is the fundamental quantity in the stochastic integral
    representation and viscous stress tensor formulations.

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    S : ndarray, shape (Nz, Ny, Nx, 3, 3)
        Symmetric strain-rate tensor.
    """
    gradU = grad_velocity(U, dx, dy, dz)
    return 0.5 * (gradU + np.swapaxes(gradU, -1, -2))


def convective_term(U: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Compute the convective term (U · ∇)U in 3D.

    For each component i, computes:
        C_i = sum_j U_j * ∂U_i/∂x_j

    This is the advective acceleration term in the Navier-Stokes equations.

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    C : ndarray, shape (Nz, Ny, Nx, 3)
        Convective acceleration: C[..., i] = U_j * ∂U_i/∂x_j (summed over j).
    """
    _validate_vector_field(U)

    gradU = grad_velocity(U, dx, dy, dz)
    C = np.zeros_like(U)

    # C_i = sum_j U_j * dU_i/dx_j for i = 0, 1, 2
    # Component 0: u_x
    C[..., 0] = (
        U[..., 0] * gradU[..., 0, 0]
        + U[..., 1] * gradU[..., 0, 1]
        + U[..., 2] * gradU[..., 0, 2]
    )
    # Component 1: u_y
    C[..., 1] = (
        U[..., 0] * gradU[..., 1, 0]
        + U[..., 1] * gradU[..., 1, 1]
        + U[..., 2] * gradU[..., 1, 2]
    )
    # Component 2: u_z
    C[..., 2] = (
        U[..., 0] * gradU[..., 2, 0]
        + U[..., 1] * gradU[..., 2, 1]
        + U[..., 2] * gradU[..., 2, 2]
    )

    return C


def laplacian_scalar(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    3D Laplacian of a scalar field.

    Computes ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² using central differences
    with periodic-style boundary conditions.

    Parameters
    ----------
    field
        Scalar field of shape (Nz, Ny, Nx).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    lap : ndarray, shape (Nz, Ny, Nx)
        Laplacian of field.
    """
    _validate_scalar_field(field)

    # Second derivatives in each direction
    d2x = (
        np.roll(field, -1, axis=2) - 2.0 * field + np.roll(field, 1, axis=2)
    ) / (dx * dx)

    d2y = (
        np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)
    ) / (dy * dy)

    d2z = (
        np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)
    ) / (dz * dz)

    return d2x + d2y + d2z


def laplacian_vector(U: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Componentwise Laplacian of a 3D vector field.

    Applies the scalar Laplacian independently to each of the three components
    of the velocity field.

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    LU : ndarray, shape (Nz, Ny, Nx, 3)
        Laplacian of each velocity component.
    """
    _validate_vector_field(U)

    LU = np.zeros_like(U)
    LU[..., 0] = laplacian_scalar(U[..., 0], dx, dy, dz)
    LU[..., 1] = laplacian_scalar(U[..., 1], dx, dy, dz)
    LU[..., 2] = laplacian_scalar(U[..., 2], dx, dy, dz)
    return LU


def frobenius_inner(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pointwise Frobenius inner product of two 3x3 tensor fields.

    Computes the element-wise product and sums over the last two indices
    (the tensor components).

    Parameters
    ----------
    A, B
        Tensor fields of shape (Nz, Ny, Nx, 3, 3).

    Returns
    -------
    inner : ndarray, shape (Nz, Ny, Nx)
        Frobenius product: inner[k,j,i] = sum_{m,n} A[k,j,i,m,n] * B[k,j,i,m,n]
    """
    _validate_tensor_field(A)
    _validate_tensor_field(B)

    return np.sum(A * B, axis=(-2, -1))


def matrix_trace(A: np.ndarray) -> np.ndarray:
    """
    Pointwise trace of a 3x3 tensor field.

    Computes the trace (sum of diagonal elements) at each grid point.

    Parameters
    ----------
    A
        Tensor field of shape (Nz, Ny, Nx, 3, 3).

    Returns
    -------
    trA : ndarray, shape (Nz, Ny, Nx)
        Trace field: trA[k,j,i] = A[k,j,i,0,0] + A[k,j,i,1,1] + A[k,j,i,2,2]
    """
    _validate_tensor_field(A)
    return A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]


def double_contraction_gradU(
    U: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    """
    Double contraction of velocity gradient tensor with itself.

    Computes the scalar field:
        sum_{i,j} (∂u_j/∂x_i) (∂u_i/∂x_j)

    In 3D, this expands to:
        (∂u_x/∂x)² + (∂u_y/∂y)² + (∂u_z/∂z)²
        + 2[(∂u_y/∂x)(∂u_x/∂y) + (∂u_z/∂x)(∂u_x/∂z) + (∂u_z/∂y)(∂u_y/∂z)]

    This is the term appearing in the pressure Poisson source for
    incompressible Navier-Stokes:
        Δp = -∂_i u_j ∂_j u_i + divergence(F)

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    dx
        Grid spacing in x-direction.
    dy
        Grid spacing in y-direction.
    dz
        Grid spacing in z-direction.

    Returns
    -------
    contraction : ndarray, shape (Nz, Ny, Nx)
        Double contraction scalar field.
    """
    _validate_vector_field(U)

    gradU = grad_velocity(U, dx, dy, dz)

    # Extract individual derivatives
    du_x_dx = gradU[..., 0, 0]
    du_x_dy = gradU[..., 0, 1]
    du_x_dz = gradU[..., 0, 2]

    du_y_dx = gradU[..., 1, 0]
    du_y_dy = gradU[..., 1, 1]
    du_y_dz = gradU[..., 1, 2]

    du_z_dx = gradU[..., 2, 0]
    du_z_dy = gradU[..., 2, 1]
    du_z_dz = gradU[..., 2, 2]

    # Compute double contraction with explicit formula
    return (
        du_x_dx * du_x_dx
        + du_y_dy * du_y_dy
        + du_z_dz * du_z_dz
        + 2.0 * du_y_dx * du_x_dy
        + 2.0 * du_z_dx * du_x_dz
        + 2.0 * du_z_dy * du_y_dz
    )


def trusted_divergence_stats(U: np.ndarray, grid: CartesianGrid3D) -> tuple[float, float]:
    """
    Maximum and mean absolute divergence on the trusted interior.

    Evaluates divergence throughout the domain and then restricts to the
    trusted region for analysis. Useful for monitoring numerical
    incompressibility errors.

    Parameters
    ----------
    U
        Vector field of shape (Nz, Ny, Nx, 3).
    grid
        3D Cartesian grid object (CartesianGrid3D).

    Returns
    -------
    max_div : float
        Maximum absolute divergence on trusted interior.
    mean_div : float
        Mean absolute divergence on trusted interior.
    """
    divU = divergence(U, grid.dx, grid.dy, grid.dz)
    div_trusted = grid.restrict_to_trusted(divU)
    return (
        float(np.max(np.abs(div_trusted))),
        float(np.mean(np.abs(div_trusted))),
    )


def _validate_scalar_field(field: np.ndarray) -> None:
    """
    Validate scalar field shape.

    Parameters
    ----------
    field
        Field to validate.

    Raises
    ------
    ValueError
        If field is not 3D (Nz, Ny, Nx).
    """
    if field.ndim != 3:
        raise ValueError(
            f"Expected scalar field with shape (Nz, Ny, Nx), got {field.shape}."
        )


def _validate_vector_field(field: np.ndarray) -> None:
    """
    Validate vector field shape.

    Parameters
    ----------
    field
        Field to validate.

    Raises
    ------
    ValueError
        If field is not (Nz, Ny, Nx, 3).
    """
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(
            f"Expected vector field with shape (Nz, Ny, Nx, 3), got {field.shape}."
        )


def _validate_tensor_field(field: np.ndarray) -> None:
    """
    Validate tensor field shape.

    Parameters
    ----------
    field
        Field to validate.

    Raises
    ------
    ValueError
        If field is not (Nz, Ny, Nx, 3, 3).
    """
    if field.ndim != 5 or field.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected tensor field with shape (Nz, Ny, Nx, 3, 3), got {field.shape}."
        )
