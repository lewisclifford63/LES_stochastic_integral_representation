"""
3D visualization utilities for the random LES stochastic integral representation.

Includes 2D slice extraction from 3D velocity/scalar fields and Qian's normalised
incompressibility test. Oxford Mathematics Masters dissertation implementing
Qian et al. (2025).

Key concept: To visualize 3D fields, we extract 2D slices (xy-plane at a given z).
For vector fields, we show only the horizontal components (u, v) on the slice.
For particles, we plot their x-y positions and optionally filter by z range.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from les.grid import CartesianGrid3D
from les.diagnostics import (
    divergence_stats,
    trusted_divergence_stats,
    incompressibility_test,
    trusted_incompressibility_test,
)


# ------------------------------------------------------------------ #
#  Velocity snapshots (3D with z-slice extraction)
# ------------------------------------------------------------------ #

def plot_velocity_snapshot(
    grid: CartesianGrid3D,
    U: np.ndarray,
    z_slice: int | None = None,
    title: str = "Velocity snapshot",
    trusted_only: bool = True,
    show_quiver: bool = True,
    quiver_stride: int | None = None,
) -> None:
    """
    Plot speed and velocity field from a 2D horizontal slice of 3D velocity.

    The speed is computed from horizontal velocity components (u, v):
        speed = sqrt(u^2 + v^2)

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    U : ndarray, shape (Nz, Ny, Nx, 3)
        3D velocity field with components [u, v, w]
    z_slice : int, optional
        Index of z-slice to extract. Default is Nz//2 (middle of domain).
    title : str
        Plot title
    trusted_only : bool, default True
        If True, mask values outside trusted region with NaN
    show_quiver : bool, default True
        If True, show quiver plot in second subplot
    quiver_stride : int, optional
        Stride for quiver plot. Default is auto-scaled.

    Returns
    -------
    None (displays matplotlib figure)
    """
    _validate_vector_field(grid, U, "U")

    if z_slice is None:
        z_slice = grid.Nz // 2
    if z_slice < 0 or z_slice >= grid.Nz:
        raise ValueError(f"z_slice={z_slice} out of bounds [0, {grid.Nz})")

    # Extract 2D slice: shape (Ny, Nx, 3)
    U_slice = U[z_slice, :, :]

    # Compute horizontal speed from u, v components
    speed = np.sqrt(U_slice[..., 0] ** 2 + U_slice[..., 1] ** 2)

    if trusted_only:
        # Apply 2D trust mask to the slice
        trust_y = grid.trust_y_mask
        trust_x = grid.trust_x_mask
        trust_2d = trust_y[:, np.newaxis] & trust_x[np.newaxis, :]
        speed_plot = speed.copy()
        speed_plot[~trust_2d] = np.nan
    else:
        speed_plot = speed

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: speed magnitude
    im = axes[0].imshow(
        speed_plot,
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title(f"{title}: speed (z-slice {z_slice}/{grid.Nz-1})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    _draw_trusted_box_2d(axes[0], grid)
    fig.colorbar(im, ax=axes[0])

    # Right plot: quiver (vector field)
    if quiver_stride is None:
        quiver_stride = max(1, min(grid.Nx, grid.Ny) // 24)

    axes[1].quiver(
        grid.X[z_slice, ::quiver_stride, ::quiver_stride],
        grid.Y[z_slice, ::quiver_stride, ::quiver_stride],
        U_slice[::quiver_stride, ::quiver_stride, 0],
        U_slice[::quiver_stride, ::quiver_stride, 1],
    )
    axes[1].set_title(f"{title}: quiver (z-slice {z_slice}/{grid.Nz-1})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    _draw_trusted_box_2d(axes[1], grid)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Scalar field (3D with z-slice extraction)
# ------------------------------------------------------------------ #

def plot_scalar_field(
    grid: CartesianGrid3D,
    field: np.ndarray,
    z_slice: int | None = None,
    title: str = "Scalar field",
    trusted_only: bool = True,
) -> None:
    """
    Plot a 2D horizontal slice of a 3D scalar field.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    field : ndarray, shape (Nz, Ny, Nx)
        3D scalar field
    z_slice : int, optional
        Index of z-slice to extract. Default is Nz//2.
    title : str
        Plot title
    trusted_only : bool, default True
        Mask values outside trusted region with NaN

    Returns
    -------
    None (displays matplotlib figure)
    """
    _validate_scalar_field(grid, field, "field")

    if z_slice is None:
        z_slice = grid.Nz // 2
    if z_slice < 0 or z_slice >= grid.Nz:
        raise ValueError(f"z_slice={z_slice} out of bounds [0, {grid.Nz})")

    # Extract 2D slice
    field_slice = field[z_slice, :, :]

    if trusted_only:
        trust_y = grid.trust_y_mask
        trust_x = grid.trust_x_mask
        trust_2d = trust_y[:, np.newaxis] & trust_x[np.newaxis, :]
        field_plot = field_slice.copy()
        field_plot[~trust_2d] = np.nan
    else:
        field_plot = field_slice

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        field_plot,
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.title(f"{title} (z-slice {z_slice}/{grid.Nz-1})")
    plt.xlabel("x")
    plt.ylabel("y")
    _draw_trusted_box_2d(plt.gca(), grid)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Pressure gradient (3D with z-slice extraction)
# ------------------------------------------------------------------ #

def plot_pressure_gradient_snapshot(
    grid: CartesianGrid3D,
    grad_p: np.ndarray,
    z_slice: int | None = None,
    title: str = "Pressure gradient",
    trusted_only: bool = True,
) -> None:
    """
    Plot the magnitude of pressure gradient on a 2D horizontal slice.

    Magnitude is computed as sqrt(grad_p_x^2 + grad_p_y^2 + grad_p_z^2).

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    grad_p : ndarray, shape (Nz, Ny, Nx, 3)
        3D pressure gradient field
    z_slice : int, optional
        Index of z-slice. Default is Nz//2.
    title : str
        Plot title
    trusted_only : bool, default True
        Mask outside trusted region

    Returns
    -------
    None (displays matplotlib figure)
    """
    _validate_vector_field(grid, grad_p, "grad_p")

    if z_slice is None:
        z_slice = grid.Nz // 2
    if z_slice < 0 or z_slice >= grid.Nz:
        raise ValueError(f"z_slice={z_slice} out of bounds [0, {grid.Nz})")

    # Extract 2D slice and compute magnitude
    grad_p_slice = grad_p[z_slice, :, :]
    mag = np.sqrt(
        grad_p_slice[..., 0] ** 2 + grad_p_slice[..., 1] ** 2 + grad_p_slice[..., 2] ** 2
    )

    if trusted_only:
        trust_y = grid.trust_y_mask
        trust_x = grid.trust_x_mask
        trust_2d = trust_y[:, np.newaxis] & trust_x[np.newaxis, :]
        mag_plot = mag.copy()
        mag_plot[~trust_2d] = np.nan
    else:
        mag_plot = mag

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        mag_plot,
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.title(f"{title} (z-slice {z_slice}/{grid.Nz-1})")
    plt.xlabel("x")
    plt.ylabel("y")
    _draw_trusted_box_2d(plt.gca(), grid)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Particle positions (3D)
# ------------------------------------------------------------------ #

def plot_particles(
    grid: CartesianGrid3D,
    positions: np.ndarray,
    title: str = "Particle positions",
    trusted_only: bool = False,
    z_min: float | None = None,
    z_max: float | None = None,
    max_points: int | None = 5000,
    s: float = 4.0,
) -> None:
    """
    Scatter plot of particle positions projected onto the x-y plane.

    Optionally filter particles by z-coordinate range.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    positions : ndarray, shape (Np, 3)
        Particle positions [x_i, y_i, z_i]
    title : str
        Plot title
    trusted_only : bool, default False
        If True, only plot particles in the trusted region
    z_min, z_max : float, optional
        Filter particles: include only those with z_min <= z <= z_max.
        If None, no filtering by z.
    max_points : int, optional
        Maximum number of points to plot (for performance). Default 5000.
    s : float, default 4.0
        Marker size for scatter plot

    Returns
    -------
    None (displays matplotlib figure)
    """
    _validate_positions(positions)

    pts = positions
    if trusted_only:
        mask = grid.points_in_trusted_region(positions)
        pts = positions[mask]

    # Filter by z range if specified
    if z_min is not None or z_max is not None:
        z_min_val = z_min if z_min is not None else -np.inf
        z_max_val = z_max if z_max is not None else np.inf
        mask = (pts[:, 2] >= z_min_val) & (pts[:, 2] <= z_max_val)
        pts = pts[mask]

    # Subsample if too many points
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
    title_suffix = ""
    if z_min is not None or z_max is not None:
        title_suffix = f" (z in [{z_min}, {z_max}])"
    plt.title(title + title_suffix)
    _draw_trusted_box_2d(plt.gca(), grid)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Mass field (3D with z-slice extraction)
# ------------------------------------------------------------------ #

def plot_mass_field(
    grid: CartesianGrid3D,
    mass_field: np.ndarray,
    z_slice: int | None = None,
    title: str = "Mass field",
    trusted_only: bool = True,
) -> None:
    """
    Plot the deposition mass field on a 2D horizontal slice.

    Parameters
    ----------
    grid : CartesianGrid3D
        3D Cartesian grid
    mass_field : ndarray, shape (Nz, Ny, Nx)
        3D mass field
    z_slice : int, optional
        Index of z-slice. Default is Nz//2.
    title : str
        Plot title
    trusted_only : bool, default True
        Mask outside trusted region

    Returns
    -------
    None (displays matplotlib figure)
    """
    _validate_scalar_field(grid, mass_field, "mass_field")

    if z_slice is None:
        z_slice = grid.Nz // 2
    if z_slice < 0 or z_slice >= grid.Nz:
        raise ValueError(f"z_slice={z_slice} out of bounds [0, {grid.Nz})")

    # Extract 2D slice
    mass_slice = mass_field[z_slice, :, :]

    if trusted_only:
        trust_y = grid.trust_y_mask
        trust_x = grid.trust_x_mask
        trust_2d = trust_y[:, np.newaxis] & trust_x[np.newaxis, :]
        mass_plot = mass_slice.copy()
        mass_plot[~trust_2d] = np.nan
    else:
        mass_plot = mass_slice

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        mass_plot,
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.title(f"{title} (z-slice {z_slice}/{grid.Nz-1})")
    plt.xlabel("x")
    plt.ylabel("y")
    _draw_trusted_box_2d(plt.gca(), grid)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Step diagnostics (3x2 layout with Qian's incompressibility test)
# ------------------------------------------------------------------ #

def plot_step_diagnostics(history: dict[str, list], grid: CartesianGrid3D) -> None:
    """
    Plot basic diagnostics from a saved history dictionary (3D).

    Layout (3 rows x 2 cols):
        [0,0] Kinetic energy          [0,1] Max speed
        [1,0] Divergence stats        [1,1] Trusted divergence stats
        [2,0] Qian test (full)        [2,1] Qian test (trusted)

    Diagnostics computed from 3D velocity field U (Nz, Ny, Nx, 3).

    Parameters
    ----------
    history : dict[str, list]
        'times'     list of float
        'U'         list of (Nz, Ny, Nx, 3) velocity snapshots
    grid : CartesianGrid3D
        3D Cartesian grid for domain/volume computation

    Returns
    -------
    None (displays matplotlib figure)
    """
    times = history["times"]
    U_list = history["U"]

    kinetic = []
    max_speed_vals = []
    div_max = []
    div_mean = []
    div_t_max = []
    div_t_mean = []
    qian_full = []
    qian_trusted = []

    for U in U_list:
        # Kinetic energy: 0.5 * integral(u^2 + v^2 + w^2) dV
        speed2 = U[..., 0] ** 2 + U[..., 1] ** 2 + U[..., 2] ** 2
        kinetic.append(0.5 * float(np.sum(speed2) * grid.cell_volume))

        # Max speed
        speed = np.sqrt(speed2)
        max_speed_vals.append(float(np.max(speed)))

        # Divergence statistics
        a, b = divergence_stats(U, grid)
        div_max.append(a)
        div_mean.append(b)

        # Trusted region divergence
        c, d = trusted_divergence_stats(U, grid)
        div_t_max.append(c)
        div_t_mean.append(d)

        # Qian's normalised incompressibility test (Qian et al. 2025, Section 4.3.1)
        qian_full.append(incompressibility_test(U, grid))
        qian_trusted.append(trusted_incompressibility_test(U, grid))

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Row 0: Kinetic energy and max speed
    axes[0, 0].plot(times, kinetic, "o-", linewidth=2)
    axes[0, 0].set_title("Kinetic energy")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, max_speed_vals, "o-", linewidth=2)
    axes[0, 1].set_title("Max speed")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].grid(True, alpha=0.3)

    # Row 1: Divergence statistics (full and trusted)
    axes[1, 0].plot(times, div_max, "o-", label="full max", linewidth=2)
    axes[1, 0].plot(times, div_mean, "s-", label="full mean", linewidth=2)
    axes[1, 0].set_title("Divergence stats (full domain)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, div_t_max, "o-", label="trusted max", linewidth=2)
    axes[1, 1].plot(times, div_t_mean, "s-", label="trusted mean", linewidth=2)
    axes[1, 1].set_title("Divergence stats (trusted region)")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Row 2: Qian's normalised incompressibility test
    # Test = <|div u| / ||grad u||>  (should be ~ 0 for incompressible flow)
    axes[2, 0].plot(times, qian_full, "o-", linewidth=2)
    axes[2, 0].set_title(
        r"Qian test $\langle |\nabla \cdot \mathbf{u}| / \|\nabla \mathbf{u}\| \rangle$ (full)"
    )
    axes[2, 0].set_xlabel("t")
    axes[2, 0].axhline(y=1.0, color="r", linestyle="--", alpha=0.4, label="threshold = 1")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(times, qian_trusted, "o-", linewidth=2)
    axes[2, 1].set_title(
        r"Qian test $\langle |\nabla \cdot \mathbf{u}| / \|\nabla \mathbf{u}\| \rangle$ (trusted)"
    )
    axes[2, 1].set_xlabel("t")
    axes[2, 1].axhline(y=1.0, color="r", linestyle="--", alpha=0.4, label="threshold = 1")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _draw_trusted_box_2d(ax, grid: CartesianGrid3D) -> None:
    """
    Draw the boundary of the 2D trusted region on a 2D subplot.

    The trusted region in (x, y) is [-L_trust, L_trust]^2.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    grid : CartesianGrid3D
        3D grid containing L_trust
    """
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
        edgecolor="red",
        alpha=0.6,
    )
    ax.add_patch(rect)


def _validate_scalar_field(grid: CartesianGrid3D, field: np.ndarray, name: str) -> None:
    """Validate that field has shape (Nz, Ny, Nx)."""
    if field.shape != grid.shape:
        raise ValueError(f"Expected {name} to have shape {grid.shape}.")


def _validate_vector_field(grid: CartesianGrid3D, field: np.ndarray, name: str) -> None:
    """Validate that field has shape (Nz, Ny, Nx, 3)."""
    if field.shape != grid.vector_shape:
        raise ValueError(f"Expected {name} to have shape {grid.vector_shape}.")


def _validate_positions(positions: np.ndarray) -> None:
    """Validate that positions have shape (Np, 3)."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected positions to have shape (Np, 3).")
