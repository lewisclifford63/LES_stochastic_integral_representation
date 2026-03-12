import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from les.grid import Grid2D
from les.diagnostics import divergence_stats, trusted_divergence_stats


def plot_velocity_snapshot(
    grid: Grid2D,
    U: np.ndarray,
    title: str = "Velocity snapshot",
    trusted_only: bool = True,
    show_quiver: bool = True,
    quiver_stride: int | None = None,
) -> None:
    """
    Plot speed and velocity field.
    """
    _validate_vector_field(grid, U, "U")

    speed = np.sqrt(U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1])

    if trusted_only:
        speed_plot = grid.apply_trust_mask(speed, fill_value=np.nan)
    else:
        speed_plot = speed

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(
        speed_plot,
        extent=grid.extent,
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title(f"{title}: speed")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    _draw_trusted_box(axes[0], grid)
    fig.colorbar(im, ax=axes[0])

    if quiver_stride is None:
        quiver_stride = max(1, min(grid.Nx, grid.Ny) // 24)

    axes[1].quiver(
        grid.X[::quiver_stride, ::quiver_stride],
        grid.Y[::quiver_stride, ::quiver_stride],
        U[::quiver_stride, ::quiver_stride, 0],
        U[::quiver_stride, ::quiver_stride, 1],
    )
    axes[1].set_title(f"{title}: quiver")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    _draw_trusted_box(axes[1], grid)

    plt.tight_layout()
    plt.show()


def plot_scalar_field(
    grid: Grid2D,
    field: np.ndarray,
    title: str = "Scalar field",
    trusted_only: bool = True,
) -> None:
    """
    Plot a scalar field.
    """
    _validate_scalar_field(grid, field, "field")

    if trusted_only:
        field_plot = grid.apply_trust_mask(field, fill_value=np.nan)
    else:
        field_plot = field

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        field_plot,
        extent=grid.extent,
        origin="lower",
        aspect="equal",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    _draw_trusted_box(plt.gca(), grid)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def plot_pressure_gradient_snapshot(
    grid: Grid2D,
    grad_p: np.ndarray,
    title: str = "Pressure gradient",
    trusted_only: bool = True,
) -> None:
    """
    Plot pressure-gradient magnitude.
    """
    _validate_vector_field(grid, grad_p, "grad_p")

    mag = np.sqrt(grad_p[..., 0] * grad_p[..., 0] + grad_p[..., 1] * grad_p[..., 1])

    if trusted_only:
        mag_plot = grid.apply_trust_mask(mag, fill_value=np.nan)
    else:
        mag_plot = mag

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        mag_plot,
        extent=grid.extent,
        origin="lower",
        aspect="equal",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    _draw_trusted_box(plt.gca(), grid)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def plot_particles(
    grid: Grid2D,
    positions: np.ndarray,
    title: str = "Particle positions",
    trusted_only: bool = False,
    max_points: int | None = 5000,
    s: float = 4.0,
) -> None:
    """
    Scatter plot of particle positions.
    """
    _validate_positions(positions)

    pts = positions
    if trusted_only:
        mask = grid.points_in_trusted_region(positions)
        pts = positions[mask]

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=s)
    plt.xlim(-grid.L_box, grid.L_box)
    plt.ylim(-grid.L_box, grid.L_box)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    _draw_trusted_box(plt.gca(), grid)
    plt.tight_layout()
    plt.show()


def plot_step_diagnostics(history: dict[str, list], grid: Grid2D) -> None:
    """
    Plot basic diagnostics from a saved history dictionary.
    """
    times = history["times"]
    U_list = history["U"]

    kinetic = []
    max_speed_vals = []
    div_max = []
    div_mean = []
    div_t_max = []
    div_t_mean = []

    for U in U_list:
        speed2 = U[..., 0] * U[..., 0] + U[..., 1] * U[..., 1]
        kinetic.append(0.5 * float(np.sum(speed2) * grid.cell_area))

        speed = np.sqrt(speed2)
        max_speed_vals.append(float(np.max(speed)))

        a, b = divergence_stats(U, grid)
        div_max.append(a)
        div_mean.append(b)

        c, d = trusted_divergence_stats(U, grid)
        div_t_max.append(c)
        div_t_mean.append(d)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(times, kinetic)
    axes[0, 0].set_title("Kinetic energy")
    axes[0, 0].set_xlabel("t")

    axes[0, 1].plot(times, max_speed_vals)
    axes[0, 1].set_title("Max speed")
    axes[0, 1].set_xlabel("t")

    axes[1, 0].plot(times, div_max, label="full max")
    axes[1, 0].plot(times, div_mean, label="full mean")
    axes[1, 0].set_title("Divergence stats")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].legend()

    axes[1, 1].plot(times, div_t_max, label="trusted max")
    axes[1, 1].plot(times, div_t_mean, label="trusted mean")
    axes[1, 1].set_title("Trusted divergence stats")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_mass_field(
    grid: Grid2D,
    mass_field: np.ndarray,
    title: str = "Mass field",
    trusted_only: bool = True,
) -> None:
    """
    Plot the deposition mass field.
    """
    _validate_scalar_field(grid, mass_field, "mass_field")

    if trusted_only:
        mass_plot = grid.apply_trust_mask(mass_field, fill_value=np.nan)
    else:
        mass_plot = mass_field

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        mass_plot,
        extent=grid.extent,
        origin="lower",
        aspect="equal",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    _draw_trusted_box(plt.gca(), grid)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def _draw_trusted_box(ax, grid: Grid2D) -> None:
    if grid.L_trust is None:
        return

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


def _validate_scalar_field(grid: Grid2D, field: np.ndarray, name: str) -> None:
    if field.shape != grid.shape:
        raise ValueError(f"Expected {name} to have shape {grid.shape}.")


def _validate_vector_field(grid: Grid2D, field: np.ndarray, name: str) -> None:
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")


def _validate_positions(positions: np.ndarray) -> None:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected positions to have shape (Np, 2).")