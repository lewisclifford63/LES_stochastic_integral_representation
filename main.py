import numpy as np

from les.config import SimulationConfig
from les.grid import Grid2D
from les.initial_conditions import (
    taylor_green_velocity,
    compute_divergence,
    compute_speed,
    trusted_max_abs_divergence,
    trusted_mean_abs_divergence,
)
from les.particles import initialize_particles_from_grid, particle_count
from les.reconstruction import (
    reconstruct_velocity_from_particles,
    reconstruction_error,
    relative_l2_error,
    trusted_relative_l2_error,
    trusted_reconstruction_error,
)


def main() -> None:
    cfg = SimulationConfig()

    if cfg.dim != 2:
        raise ValueError("This prototype is only implemented for the 2D case.")

    grid = Grid2D(
        L_box=cfg.L_box,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        L_trust=cfg.L_trust,
    )

    # Initial velocity field on the full padded computational box
    U0 = taylor_green_velocity(grid)

    # Initial diagnostics on full box
    divU0 = compute_divergence(U0, grid.dx, grid.dy)
    speed0 = compute_speed(U0)

    # Initial diagnostics on trusted interior only
    speed0_trusted = grid.restrict_to_trusted(speed0)
    divU0_trusted = grid.restrict_to_trusted(divU0)

    print("=== Simulation setup ===")
    print(f"dim                       = {cfg.dim}")
    print(f"trusted domain            = [-{cfg.L_trust}, {cfg.L_trust}]^2")
    print(f"padded computational box  = [-{cfg.L_box}, {cfg.L_box}]^2")
    print(f"padding width             = {cfg.pad}")
    print(f"grid                      = {cfg.Nx} x {cfg.Ny}")
    print(f"trusted grid shape        = {grid.trusted_shape[1]} x {grid.trusted_shape[0]}")
    print(f"dx, dy                    = {grid.dx:.6f}, {grid.dy:.6f}")
    print(f"dt                        = {cfg.dt}")
    print(f"T                         = {cfg.T}")
    print(f"num_steps                 = {cfg.num_steps}")
    print(f"nu                        = {cfg.nu}")
    print(f"filter_width              = {cfg.filter_width}")
    print(f"filter_cutoff_sigma       = {cfg.filter_cutoff_sigma}")
    print(f"seed                      = {cfg.seed}")
    print()

    print("=== Initial field diagnostics (full padded box) ===")
    print(f"max|u|                    = {np.max(speed0):.6e}")
    print(f"mean|u|                   = {np.mean(speed0):.6e}")
    print(f"max|div u|                = {np.max(np.abs(divU0)):.6e}")
    print(f"mean|div u|               = {np.mean(np.abs(divU0)):.6e}")
    print()

    print("=== Initial field diagnostics (trusted interior only) ===")
    print(f"max|u|                    = {np.max(speed0_trusted):.6e}")
    print(f"mean|u|                   = {np.mean(speed0_trusted):.6e}")
    print(f"max|div u|                = {trusted_max_abs_divergence(U0, grid):.6e}")
    print(f"mean|div u|               = {trusted_mean_abs_divergence(U0, grid):.6e}")
    print()

    if cfg.plot_initial_field:
        plot_initial_field(grid, U0, speed0, divU0)

    # ------------------------------------------------------------------
    # Particle initialisation and time-zero reconstruction test
    # ------------------------------------------------------------------
    positions, amplitudes, weights = initialize_particles_from_grid(
        grid=grid,
        U0=U0,
        trusted_only=False,
    )

    print("=== Particle initialisation ===")
    print(f"number of particles       = {particle_count(positions)}")
    print(f"particle weight           = {weights[0]:.6e}")
    print(f"particles on padded box   = True")
    print()

    U_rec = reconstruct_velocity_from_particles(
        grid=grid,
        positions=positions,
        amplitudes=amplitudes,
        weights=weights,
        sigma=cfg.filter_width,
        cutoff_sigma=cfg.filter_cutoff_sigma,
    )

    err_vec_full, err_mag_full = reconstruction_error(U0, U_rec)
    rel_err_full = relative_l2_error(U0, U_rec)

    err_vec_trusted, err_mag_trusted = trusted_reconstruction_error(grid, U0, U_rec)
    rel_err_trusted = trusted_relative_l2_error(grid, U0, U_rec)

    divU_rec = compute_divergence(U_rec, grid.dx, grid.dy)
    speed_rec = compute_speed(U_rec)

    speed_rec_trusted = grid.restrict_to_trusted(speed_rec)
    divU_rec_trusted = grid.restrict_to_trusted(divU_rec)

    print("=== Time-zero reconstruction diagnostics (full padded box) ===")
    print(f"max|u_rec|                = {np.max(speed_rec):.6e}")
    print(f"mean|u_rec|               = {np.mean(speed_rec):.6e}")
    print(f"max|div u_rec|            = {np.max(np.abs(divU_rec)):.6e}")
    print(f"mean|div u_rec|           = {np.mean(np.abs(divU_rec)):.6e}")
    print(f"max reconstruction error  = {np.max(err_mag_full):.6e}")
    print(f"mean reconstruction error = {np.mean(err_mag_full):.6e}")
    print(f"relative L2 error         = {rel_err_full:.6e}")
    print()

    print("=== Time-zero reconstruction diagnostics (trusted interior only) ===")
    print(f"max|u_rec|                = {np.max(speed_rec_trusted):.6e}")
    print(f"mean|u_rec|               = {np.mean(speed_rec_trusted):.6e}")
    print(f"max|div u_rec|            = {np.max(np.abs(divU_rec_trusted)):.6e}")
    print(f"mean|div u_rec|           = {np.mean(np.abs(divU_rec_trusted)):.6e}")
    print(f"max reconstruction error  = {np.max(err_mag_trusted):.6e}")
    print(f"mean reconstruction error = {np.mean(err_mag_trusted):.6e}")
    print(f"relative L2 error         = {rel_err_trusted:.6e}")

    if cfg.plot_reconstruction_test:
        plot_reconstruction_test(grid, U0, U_rec, err_mag_full)


def _full_extent(grid: Grid2D) -> list[float]:
    return [
        grid.x[0],
        grid.x[-1] + grid.dx,
        grid.y[0],
        grid.y[-1] + grid.dy,
    ]


def _draw_trusted_box(ax, grid: Grid2D) -> None:
    if grid.L_trust is None:
        return

    import matplotlib.patches as patches

    L = grid.L_trust
    rect = patches.Rectangle(
        (-L, -L),
        2.0 * L,
        2.0 * L,
        fill=False,
        linewidth=2.0,
        linestyle="--",
    )
    ax.add_patch(rect)


def plot_initial_field(
    grid: Grid2D,
    U: np.ndarray,
    speed: np.ndarray,
    divU: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    speed_masked = grid.apply_trust_mask(speed, fill_value=np.nan)
    div_masked = grid.apply_trust_mask(divU, fill_value=np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = _full_extent(grid)

    stride_x = max(1, grid.Nx // 24)
    stride_y = max(1, grid.Ny // 24)

    im0 = axes[0].imshow(
        speed_masked,
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title("Initial speed |u| (trusted region)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    _draw_trusted_box(axes[0], grid)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        div_masked,
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    axes[1].set_title("Initial divergence ∇·u (trusted region)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    _draw_trusted_box(axes[1], grid)
    fig.colorbar(im1, ax=axes[1])

    axes[2].quiver(
        grid.X[::stride_y, ::stride_x],
        grid.Y[::stride_y, ::stride_x],
        U[::stride_y, ::stride_x, 0],
        U[::stride_y, ::stride_x, 1],
    )
    axes[2].set_title("Initial velocity field on padded box")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    _draw_trusted_box(axes[2], grid)

    plt.tight_layout()
    plt.show()


def plot_reconstruction_test(
    grid: Grid2D,
    U0: np.ndarray,
    U_rec: np.ndarray,
    err_mag: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    speed0 = compute_speed(U0)
    speed_rec = compute_speed(U_rec)

    speed0_masked = grid.apply_trust_mask(speed0, fill_value=np.nan)
    speed_rec_masked = grid.apply_trust_mask(speed_rec, fill_value=np.nan)
    err_mag_masked = grid.apply_trust_mask(err_mag, fill_value=np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = _full_extent(grid)

    im0 = axes[0].imshow(
        speed0_masked,
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title("Original speed |u₀| (trusted region)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    _draw_trusted_box(axes[0], grid)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        speed_rec_masked,
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    axes[1].set_title("Reconstructed speed |u_rec| (trusted region)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    _draw_trusted_box(axes[1], grid)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_mag_masked,
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    axes[2].set_title("Reconstruction error magnitude (trusted region)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    _draw_trusted_box(axes[2], grid)
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()