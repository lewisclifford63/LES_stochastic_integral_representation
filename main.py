"""
Main driver for the random LES simulation using Qian's full
accumulated-history scheme (eqs. 41-44 / 54-56).

This script lives *outside* the ``les/`` package directory, so all
imports reach into ``les.*`` explicitly.

The iterative procedure per time-step is:

    U_k, Y_k  -->  Y_{k+1}            (Milstein, eq. 55)
               +   G_k, u_0  -->  U_{k+1}   (direct particle sum, eq. 56)
                                  +  dU_{k+1}/dx, F  -->  G_{k+1}  (eq. 44)
"""

import numpy as np

from les.config import SimulationConfig
from les.grid import Grid2D
from les.initial_conditions import (
    taylor_green_velocity,
    gaussian_vortex_velocity,
)
from les.particles import initialize_particle_history
from les.forcing import zero_forcing
from les.forcing import constant_forcing
from les.forcing import swirling_gaussian_forcing
from les.time_integrator import (
    advance_velocity_one_step_qian,
    run_time_loop_qian,
)
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
    particle_box_escape_fraction,
    particle_trusted_fraction,
)
from les.plotting import (
    plot_velocity_snapshot,
    plot_pressure_gradient_snapshot,
    plot_mass_field,
    plot_particles,
    plot_step_diagnostics,
)


def main() -> None:
    cfg = SimulationConfig()

    if cfg.dim != 2:
        raise ValueError("This prototype only supports dim = 2.")

    rng = np.random.default_rng(cfg.seed)

    grid = Grid2D(
        L_box=cfg.L_box,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        L_trust=cfg.L_trust,
    )

    # ---------------------------------------------------------- #
    #  Initial velocity
    # ---------------------------------------------------------- #
    U0 = build_initial_velocity(grid, kind="gaussian_vortex")

    # ---------------------------------------------------------- #
    #  Particle state  (Qian's accumulated-history structure)
    # ---------------------------------------------------------- #
    particle_state = initialize_particle_history(
        grid=grid,
        U0=U0,
        trusted_only=False,
    )

    if cfg.verbose:
        print_setup(cfg, grid, particle_state)
        print_velocity_diagnostics("Initial diagnostics", U0, grid)
        print_particle_diagnostics(
            "Initial particle diagnostics",
            particle_state["positions"],
            grid,
        )

    if cfg.plot_initial_field:
        plot_velocity_snapshot(
            grid=grid,
            U=U0,
            title="Initial velocity",
            trusted_only=True,
        )
        plot_particles(
            grid=grid,
            positions=particle_state["positions"],
            title="Initial particles",
            trusted_only=False,
        )

    # ---------------------------------------------------------- #
    #  Time loop  (Qian's full scheme)
    # ---------------------------------------------------------- #
    forcing_function = build_forcing_function()

    history = run_time_loop_qian(
        grid=grid,
        particle_state=particle_state,
        U_0=U0,
        forcing_function=forcing_function,
        num_steps=cfg.num_steps,
        dt=cfg.dt,
        nu=cfg.nu,
        filter_width=cfg.filter_width,
        filter_cutoff_sigma=cfg.filter_cutoff_sigma,
        rng=rng,
        t0=cfg.t0,
        pressure_cutoff_radius=cfg.pressure_cutoff_radius,
        pressure_softening=cfg.pressure_softening,
        clip_to_box=False,
        use_milstein=True,
        save_every=cfg.save_every,
    )

    # ---------------------------------------------------------- #
    #  Post-processing output
    # ---------------------------------------------------------- #
    if cfg.verbose:
        print_history_diagnostics(history, grid)

    if cfg.plot_velocity_each_save:
        times = history["times"]
        states = history["U"]

        for idx in range(1, len(times)):
            t_val = times[idx]
            U_val = states[idx]

            plot_velocity_snapshot(
                grid=grid,
                U=U_val,
                title=f"Velocity at t = {t_val:.3f}",
                trusted_only=True,
            )

    if cfg.plot_pressure_each_save:
        times = history["times"]
        grad_ps = history["grad_p"]

        for idx in range(len(grad_ps)):
            t_val = times[idx + 1]
            plot_pressure_gradient_snapshot(
                grid=grid,
                grad_p=grad_ps[idx],
                title=f"Pressure gradient at t = {t_val:.3f}",
                trusted_only=True,
            )

    plot_step_diagnostics(history, grid)

    if cfg.verbose:
        print()
        print("=== Final diagnostics ===")
        U_final = history["U"][-1]
        positions_final = history["positions"][-1]
        print_velocity_diagnostics("Final velocity diagnostics", U_final, grid)
        print_particle_diagnostics(
            "Final particle diagnostics", positions_final, grid,
        )


# ================================================================== #
#  Helpers
# ================================================================== #

def build_initial_velocity(grid: Grid2D, kind: str = "taylor_green") -> np.ndarray:
    if kind == "taylor_green":
        return taylor_green_velocity(grid)

    if kind == "gaussian_vortex":
        return gaussian_vortex_velocity(
            grid=grid,
            strength=1.0,
            sigma=0.4,
            center=(0.0, 0.0),
        )

    raise ValueError(f"Unknown initial condition kind: {kind}")


def build_forcing_function():
    def forcing(grid: Grid2D, t: float) -> np.ndarray:
        return constant_forcing(grid=grid, t=t, fx=0.5, fy=0.0)

    return forcing


def print_setup(
    cfg: SimulationConfig,
    grid: Grid2D,
    particle_state: dict[str, np.ndarray],
) -> None:
    positions = particle_state["positions"]
    weights = particle_state["weights"]

    print("=== Simulation setup (Qian scheme) ===")
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
    print(f"filter_width              = {cfg.filter_width}")
    print(f"filter_cutoff_sigma       = {cfg.filter_cutoff_sigma}")
    print(f"pressure_cutoff_radius    = {cfg.pressure_cutoff_radius}")
    print(f"pressure_softening        = {cfg.pressure_softening}")
    print(f"seed                      = {cfg.seed}")
    print(f"SDE integrator            = Milstein (Qian eq. 55)")
    print(f"deposition                = direct particle sum (no normalisation)")
    print(f"history                   = full accumulated (Qian eq. 56)")
    print(f"number of particles       = {positions.shape[0]}")
    if weights.shape[0] > 0:
        print(f"particle weight (s^d)     = {weights[0]:.6e}")


def print_velocity_diagnostics(
    label: str,
    U: np.ndarray,
    grid: Grid2D,
) -> None:
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


def print_particle_diagnostics(
    label: str,
    positions: np.ndarray,
    grid: Grid2D,
) -> None:
    frac_out = particle_box_escape_fraction(positions, grid)
    frac_trust = particle_trusted_fraction(positions, grid)

    print(f"=== {label} ===")
    print(f"particle count            = {positions.shape[0]}")
    print(f"fraction outside box      = {frac_out:.6e}")
    print(f"fraction in trusted region = {frac_trust:.6e}")


def print_history_diagnostics(
    history: dict[str, list],
    grid: Grid2D,
) -> None:
    times = history["times"]
    states = history["U"]
    positions_list = history["positions"]

    for idx in range(1, len(times)):
        t_val = times[idx]
        U_val = states[idx]
        pos_val = positions_list[idx]

        print()
        print(f"=== Saved state {idx}  (t = {t_val:.6f}) ===")
        print_velocity_diagnostics("Velocity diagnostics", U_val, grid)
        print_particle_diagnostics("Particle diagnostics", pos_val, grid)


if __name__ == "__main__":
    main()