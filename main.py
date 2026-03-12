import numpy as np

from les.config import SimulationConfig
from les.grid import Grid2D
from les.initial_conditions import (
    taylor_green_velocity,
    gaussian_vortex_velocity,
)
from les.particles import initialize_particles_from_grid
from les.forcing import zero_forcing
from les.forcing import constant_forcing
from les.forcing import swirling_gaussian_forcing
from les.time_integrator import advance_velocity_one_step
from les.diagnostics import (
    kinetic_energy,
    trusted_kinetic_energy,
    max_speed,
    mean_speed,
    trusted_speed_stats,
    divergence_stats,
    trusted_divergence_stats,
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

    U0 = build_initial_velocity(grid, kind="gaussian_vortex")
    positions, weights = initialize_particles_from_grid(
        grid=grid,
        trusted_only=False,
    )

    if cfg.verbose:
        print_setup(cfg, grid, positions, weights)
        print_velocity_diagnostics("Initial diagnostics", U0, grid)
        print_particle_diagnostics("Initial particle diagnostics", positions, grid)

    if cfg.plot_initial_field:
        plot_velocity_snapshot(
            grid=grid,
            U=U0,
            title="Initial velocity",
            trusted_only=True,
        )
        plot_particles(
            grid=grid,
            positions=positions,
            title="Initial particles",
            trusted_only=False,
        )

    history = {
        "times": [cfg.t0],
        "U": [U0.copy()],
        "positions": [positions.copy()],
        "weights": [weights.copy()],
        "source": [],
        "grad_p": [],
        "g": [],
        "mass_field": [],
    }

    U_n = U0.copy()
    positions_n = positions.copy()
    weights_n = weights.copy()
    t_n = cfg.t0

    for step in range(1, cfg.num_steps + 1):
        F_n = constant_forcing(grid=grid, t=t_n, fx=0.5, fy=0.0)

        result = advance_velocity_one_step(
            grid=grid,
            positions_n=positions_n,
            weights=weights_n,
            U_n=U_n,
            F_n=F_n,
            dt=cfg.dt,
            nu=cfg.nu,
            filter_width=cfg.filter_width,
            filter_cutoff_sigma=cfg.filter_cutoff_sigma,
            rng=rng,
            pressure_cutoff_radius=cfg.pressure_cutoff_radius,
            pressure_softening=cfg.pressure_softening,
            apply_les_filter=False,
            clip_to_box=False,
            mass_tol=1.0e-14,
        )

        U_n = result["U_np1"]
        positions_n = result["positions_np1"]
        weights_n = result["weights_np1"]
        t_n = cfg.t0 + step * cfg.dt

        should_save = (step % cfg.save_every == 0) or (step == cfg.num_steps)

        if should_save:
            history["times"].append(t_n)
            history["U"].append(U_n.copy())
            history["positions"].append(positions_n.copy())
            history["weights"].append(weights_n.copy())
            history["source"].append(result["source_n"].copy())
            history["grad_p"].append(result["grad_p_n"].copy())
            history["g"].append(result["g_n"].copy())
            history["mass_field"].append(result["mass_field"].copy())

            if cfg.verbose:
                print()
                print(f"=== Step {step} / {cfg.num_steps} ===")
                print(f"time = {t_n:.6f}")
                print_velocity_diagnostics("Velocity diagnostics", U_n, grid)
                print_particle_diagnostics(
                    "Particle diagnostics",
                    positions_n,
                    grid,
                )

            if cfg.plot_velocity_each_save:
                plot_velocity_snapshot(
                    grid=grid,
                    U=U_n,
                    title=f"Velocity at t = {t_n:.3f}",
                    trusted_only=True,
                )

            if cfg.plot_pressure_each_save:
                plot_pressure_gradient_snapshot(
                    grid=grid,
                    grad_p=result["grad_p_n"],
                    title=f"Pressure gradient at t = {t_n:.3f}",
                    trusted_only=True,
                )

            plot_mass_field(
                grid=grid,
                mass_field=result["mass_field"],
                title=f"Mass field at t = {t_n:.3f}",
                trusted_only=True,
            )

    if cfg.verbose:
        print()
        print("=== Final diagnostics ===")
        print_velocity_diagnostics("Final velocity diagnostics", U_n, grid)
        print_particle_diagnostics("Final particle diagnostics", positions_n, grid)

    plot_step_diagnostics(history, grid)


def build_initial_velocity(grid: Grid2D, kind: str = "taylor_green") -> np.ndarray:
    if kind == "taylor_green":
        return taylor_green_velocity(grid)

    if kind == "gaussian_vortex":
        return gaussian_vortex_velocity(
            grid=grid,
            strength=3.0,
            sigma=0.3,
            center=(0.0, 0.0),
        )

    raise ValueError(f"Unknown initial condition kind: {kind}")


def print_setup(
    cfg: SimulationConfig,
    grid: Grid2D,
    positions: np.ndarray,
    weights: np.ndarray,
) -> None:
    print("=== Simulation setup ===")
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
    print(f"number of particles       = {positions.shape[0]}")
    if weights.shape[0] > 0:
        print(f"particle weight           = {weights[0]:.6e}")


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


if __name__ == "__main__":
    main()