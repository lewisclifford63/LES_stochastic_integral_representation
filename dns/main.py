"""
DNS reference simulation using explicit Euler + FFT projection.

Reads grid, dt, nu, T, save_every from ``les.config.SimulationConfig``
so that the DNS and LES runs use identical physical parameters.

Initial condition and forcing are set to match the LES ``main.py``
(Qian-aligned Gaussian vortex + localised Gaussian forcing).
"""

import numpy as np

from les.config import SimulationConfig
from les.grid import Grid2D
from les.initial_conditions import (
    taylor_green_velocity,
    gaussian_vortex_velocity,
)
from les.forcing import zero_forcing, constant_forcing, gaussian_forcing
from les.diagnostics import (
    kinetic_energy,
    trusted_kinetic_energy,
    max_speed,
    mean_speed,
    trusted_speed_stats,
    divergence_stats,
    trusted_divergence_stats,
    incompressibility_test,
    trusted_incompressibility_test,
)
from les.plotting import plot_velocity_snapshot, plot_step_diagnostics

from dns.time_integrator import run_time_loop


def main() -> None:
    cfg = SimulationConfig()

    if cfg.dim != 2:
        raise ValueError("DNS prototype only supports dim = 2.")

    grid = Grid2D(
        L_box=cfg.L_box,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        L_trust=cfg.L_trust,
    )

    U0 = build_initial_velocity(grid, kind="gaussian_vortex")

    if cfg.verbose:
        print_setup(cfg, grid)
        print_velocity_diagnostics("Initial DNS diagnostics", U0, grid)

    if cfg.plot_initial_field:
        plot_velocity_snapshot(
            grid=grid,
            U=U0,
            title="DNS initial velocity",
            trusted_only=True,
        )

    forcing_function = build_forcing_function()

    history = run_time_loop(
        grid=grid,
        U_0=U0,
        forcing_function=forcing_function,
        num_steps=cfg.num_steps,
        dt=cfg.dt,
        nu=cfg.nu,
        t0=cfg.t0,
        save_every=cfg.save_every,
    )

    if cfg.verbose:
        print_dns_history(history, grid)

    if cfg.plot_velocity_each_save:
        times = history["times"]
        states = history["U"]

        for idx in range(1, len(times)):
            t_val = times[idx]
            U_val = states[idx]

            plot_velocity_snapshot(
                grid=grid,
                U=U_val,
                title=f"DNS velocity at t = {t_val:.3f}",
                trusted_only=True,
            )

    plot_step_diagnostics(history, grid)


def build_initial_velocity(grid: Grid2D, kind: str = "gaussian_vortex") -> np.ndarray:
    if kind == "taylor_green":
        return taylor_green_velocity(grid)

    if kind == "gaussian_vortex":
        # Qian-aligned scales: max speed ~ U0 ~ 32 for Re = 1000
        # Must match the LES main.py exactly for a fair comparison
        return gaussian_vortex_velocity(
            grid=grid,
            strength=52.5,
            sigma=1.57,
            center=(0.0, 0.0),
        )

    raise ValueError(f"Unknown initial condition kind: {kind}")


def build_forcing_function():
    """
    Qian-aligned forcing — must match the LES main.py exactly.

    Localised Gaussian forcing in x with amplitude 10, sigma 1.57.
    """
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return gaussian_forcing(
            grid=grid,
            t=t,
            amplitude=(10.0, 0.0),
            sigma=1.57,
            center=(0.0, 0.0),
        )

    return forcing


def print_setup(cfg: SimulationConfig, grid: Grid2D) -> None:
    print("=== DNS setup ===")
    print(f"dim                       = {cfg.dim}")
    print(f"trusted domain            = [-{cfg.L_trust}, {cfg.L_trust}]^2")
    print(f"padded computational box  = [-{cfg.L_box}, {cfg.L_box}]^2")
    print(f"padding width             = {cfg.pad}")
    print(f"grid                      = {cfg.Nx} x {cfg.Ny}")
    print(
        f"trusted grid shape        = "
        f"{grid.trusted_shape[1]} x {grid.trusted_shape[0]}"
    )
    print(f"dx, dy                    = {grid.dx:.6f}, {grid.dy:.6f}")
    print(f"dt                        = {cfg.dt}")
    print(f"T                         = {cfg.T}")
    print(f"num_steps                 = {cfg.num_steps}")
    print(f"nu                        = {cfg.nu}")
    print(f"save_every                = {cfg.save_every}")


def print_velocity_diagnostics(label: str, U: np.ndarray, grid: Grid2D) -> None:
    ke_full = kinetic_energy(U, grid)
    ke_trust = trusted_kinetic_energy(U, grid)

    max_u = max_speed(U)
    mean_u = mean_speed(U)

    max_u_t, mean_u_t = trusted_speed_stats(U, grid)

    div_max, div_mean = divergence_stats(U, grid)
    div_t_max, div_t_mean = trusted_divergence_stats(U, grid)

    qian_full = incompressibility_test(U, grid)
    qian_trust = trusted_incompressibility_test(U, grid)

    print(f"=== {label} ===")
    print(f"kinetic energy (full)     = {ke_full:.6e}")
    print(f"kinetic energy (trusted)  = {ke_trust:.6e}")
    print(f"max|u| (full)             = {max_u:.6e}")
    print(f"mean|u| (full)            = {mean_u:.6e}")
    print(f"max|u| (trusted)          = {max_u_t:.6e}")
    print(f"mean|u| (trusted)         = {mean_u_t:.6e}")
    print(f"max|div u| (full)         = {div_max:.6e}")
    print(f"mean|div u| (full)        = {div_mean:.6e}")
    print(f"max|div u| (trusted)      = {div_t_max:.6e}")
    print(f"mean|div u| (trusted)     = {div_t_mean:.6e}")
    print(f"Qian test (full)          = {qian_full:.6e}")
    print(f"Qian test (trusted)       = {qian_trust:.6e}")


def print_dns_history(history: dict[str, list], grid: Grid2D) -> None:
    times = history["times"]
    states = history["U"]

    for idx, t_val in enumerate(times):
        print()
        print(f"=== DNS saved state {idx} ===")
        print(f"time = {t_val:.6f}")
        print_velocity_diagnostics("DNS velocity diagnostics", states[idx], grid)


if __name__ == "__main__":
    main()