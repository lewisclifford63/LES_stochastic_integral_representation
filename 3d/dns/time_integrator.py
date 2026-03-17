"""
3D DNS time integration using 4th-order Runge-Kutta scheme.

References:
    Qian et al. (2025) - LES stochastic integral representation

Implements incompressible Navier-Stokes with explicit RK4 timestepping
and pressure projection for divergence-free constraint.
"""

import numpy as np
from dns.poisson import project_velocity_3d
from les.differential_operators import convective_term, laplacian_vector


def explicit_navier_stokes_rhs(U, F, dx, dy, dz, nu):
    r"""
    Compute RHS of 3D incompressible Navier-Stokes equation.

    ∂U/∂t = -(U·∇)U + ν∆U + F

    Parameters
    ----------
    U : ndarray
        Velocity field, shape (Nz, Ny, Nx, 3)
    F : ndarray
        External forcing, shape (Nz, Ny, Nx, 3)
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    dz : float
        Grid spacing in z-direction
    nu : float
        Kinematic viscosity

    Returns
    -------
    dU_dt : ndarray
        Time derivative of velocity, shape (Nz, Ny, Nx, 3)
    """
    # Nonlinear convective term: -(U·∇)U
    convection = -convective_term(U, dx, dy, dz)

    # Linear viscous term: ν∆U
    viscosity = nu * laplacian_vector(U, dx, dy, dz)

    # Total RHS
    dU_dt = convection + viscosity + F

    return dU_dt


def advance_velocity_rk4(grid, U_n, forcing_function, t_n, dt, nu):
    r"""
    Advance velocity field by one timestep using 4th-order Runge-Kutta.

    Standard RK4 scheme with 4 stages:
        k1 = f(t_n, U_n)
        k2 = f(t_n + dt/2, U_n + dt*k1/2)
        k3 = f(t_n + dt/2, U_n + dt*k2/2)
        k4 = f(t_n + dt, U_n + dt*k3)
        U_{n+1} = U_n + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    Velocity is projected to divergence-free space after each RK stage.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    U_n : ndarray
        Velocity at time t_n, shape (Nz, Ny, Nx, 3)
    forcing_function : callable
        Forcing function with signature (grid, t) -> (Nz, Ny, Nx, 3)
    t_n : float
        Current time
    dt : float
        Timestep size
    nu : float
        Kinematic viscosity

    Returns
    -------
    U_np1 : ndarray
        Velocity at time t_{n+1}, shape (Nz, Ny, Nx, 3)
    """
    # Stage 1
    f_n = forcing_function(grid, t_n)
    k1 = explicit_navier_stokes_rhs(U_n, f_n, grid.dx, grid.dy, grid.dz, nu)
    U_rk1 = U_n + 0.5 * dt * k1
    U_rk1, _ = project_velocity_3d(U_rk1, grid.dx, grid.dy, grid.dz)

    # Stage 2
    f_rk1 = forcing_function(grid, t_n + 0.5 * dt)
    k2 = explicit_navier_stokes_rhs(U_rk1, f_rk1, grid.dx, grid.dy, grid.dz, nu)
    U_rk2 = U_n + 0.5 * dt * k2
    U_rk2, _ = project_velocity_3d(U_rk2, grid.dx, grid.dy, grid.dz)

    # Stage 3
    f_rk2 = forcing_function(grid, t_n + 0.5 * dt)
    k3 = explicit_navier_stokes_rhs(U_rk2, f_rk2, grid.dx, grid.dy, grid.dz, nu)
    U_rk3 = U_n + dt * k3
    U_rk3, _ = project_velocity_3d(U_rk3, grid.dx, grid.dy, grid.dz)

    # Stage 4
    f_rk3 = forcing_function(grid, t_n + dt)
    k4 = explicit_navier_stokes_rhs(U_rk3, f_rk3, grid.dx, grid.dy, grid.dz, nu)

    # Combine stages
    U_star = U_n + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    # Final projection to ensure divergence-free
    U_np1, p_corr = project_velocity_3d(U_star, grid.dx, grid.dy, grid.dz)

    return U_np1


def advance_velocity_one_step(grid, U_n, forcing_function, t_n, dt, nu):
    """
    Legacy Euler step for reference (not recommended for production).

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    U_n : ndarray
        Velocity at time t_n, shape (Nz, Ny, Nx, 3)
    forcing_function : callable
        Forcing function with signature (grid, t) -> (Nz, Ny, Nx, 3)
    t_n : float
        Current time
    dt : float
        Timestep size
    nu : float
        Kinematic viscosity

    Returns
    -------
    U_np1 : ndarray
        Velocity at time t_{n+1}, shape (Nz, Ny, Nx, 3)
    """
    f_n = forcing_function(grid, t_n)

    dU_dt = explicit_navier_stokes_rhs(U_n, f_n, grid.dx, grid.dy, grid.dz, nu)
    U_pred = U_n + dt * dU_dt
    U_np1, p_corr = project_velocity_3d(U_pred, grid.dx, grid.dy, grid.dz)

    return U_np1


def run_time_loop(grid, U_0, forcing_function, num_steps, dt, nu, t0=0.0, save_every=1):
    """
    Run DNS time integration loop from t=0 for num_steps.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    U_0 : ndarray
        Initial velocity, shape (Nz, Ny, Nx, 3)
    forcing_function : callable
        Forcing function with signature (grid, t) -> (Nz, Ny, Nx, 3)
    num_steps : int
        Number of timesteps to integrate
    dt : float
        Timestep size
    nu : float
        Kinematic viscosity
    t0 : float, default 0.0
        Initial time
    save_every : int, default 1
        Save a snapshot every this many steps

    Returns
    -------
    history : dict[str, list]
        'times' : list of float
        'U'     : list of (Nz, Ny, Nx, 3) velocity snapshots
    """
    U_n = U_0.copy()
    t_n = float(t0)

    history = {
        "times": [t_n],
        "U": [U_n.copy()],
    }

    for step in range(1, num_steps + 1):
        U_n = advance_velocity_rk4(grid, U_n, forcing_function, t_n, dt, nu)
        t_n = t0 + step * dt

        if step % save_every == 0 or step == num_steps:
            history["times"].append(t_n)
            history["U"].append(U_n.copy())

    return history
