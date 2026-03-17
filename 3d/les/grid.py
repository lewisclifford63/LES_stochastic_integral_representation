"""
3D uniform Cartesian grid utilities for LES on [-L_box, L_box]^3.

Scalar fields: shape (Nz, Ny, Nx)
Vector fields: shape (Nz, Ny, Nx, 3) with components [u_x, u_y, u_z]
Tensor fields: shape (Nz, Ny, Nx, 3, 3)

Indexing convention:
  - axis 0: z-direction
  - axis 1: y-direction
  - axis 2: x-direction
  - vector component indices: 0 = u_x, 1 = u_y, 2 = u_z
"""

import numpy as np


class CartesianGrid3D:
    """
    Uniform Cartesian grid on the padded computational box [-L_box, L_box]^3,
    together with trusted-interior bookkeeping for [-L_trust, L_trust]^3.

    Notes
    -----
    - endpoint=False is used so the grid stays uniform and FFT-friendly.
    - the trusted region is tracked by boolean masks and contiguous slices.
    - scalar fields are stored with shape (Nz, Ny, Nx)
    - vector fields are stored with shape (Nz, Ny, Nx, 3)
    - 3x3 tensor fields are stored with shape (Nz, Ny, Nx, 3, 3)
    """

    def __init__(
        self,
        L_box: float,
        Nx: int,
        Ny: int,
        Nz: int,
        L_trust: float | None = None,
        L_box_z: float | None = None,
        L_trust_z: float | None = None,
    ):
        self.L_box = float(L_box)
        # L_box_z: separate vertical half-width; defaults to L_box (cubic grid).
        self.L_box_z = float(L_box_z) if L_box_z is not None else self.L_box
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.L_trust = None if L_trust is None else float(L_trust)
        # L_trust_z: vertical trust half-width; defaults to L_trust.
        self.L_trust_z = (float(L_trust_z) if L_trust_z is not None
                          else self.L_trust)

        if self.Nx < 2 or self.Ny < 2 or self.Nz < 2:
            raise ValueError("Nx, Ny, and Nz must all be at least 2.")

        self.x = np.linspace(-self.L_box,   self.L_box,   self.Nx, endpoint=False)
        self.y = np.linspace(-self.L_box,   self.L_box,   self.Ny, endpoint=False)
        self.z = np.linspace(-self.L_box_z, self.L_box_z, self.Nz, endpoint=False)

        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.dz = float(self.z[1] - self.z[0])

        # 3D meshgrid: Z, Y, X all of shape (Nz, Ny, Nx)
        self.Z, self.Y, self.X = np.meshgrid(
            self.z, self.y, self.x, indexing="ij",
        )

        # Trust region masks
        if self.L_trust is None:
            self.trust_x_mask = np.ones(self.Nx, dtype=bool)
            self.trust_y_mask = np.ones(self.Ny, dtype=bool)
            self.trust_z_mask = np.ones(self.Nz, dtype=bool)
        else:
            L_tz = self.L_trust_z if self.L_trust_z is not None else self.L_trust
            self.trust_x_mask = np.abs(self.x) <= self.L_trust
            self.trust_y_mask = np.abs(self.y) <= self.L_trust
            self.trust_z_mask = np.abs(self.z) <= L_tz

        # 3D trust mask: outer product of 1D masks, shape (Nz, Ny, Nx)
        self.trust_mask = (
            self.trust_z_mask[:, np.newaxis, np.newaxis]
            & self.trust_y_mask[np.newaxis, :, np.newaxis]
            & self.trust_x_mask[np.newaxis, np.newaxis, :]
        )

        if not (np.any(self.trust_x_mask)
                and np.any(self.trust_y_mask)
                and np.any(self.trust_z_mask)):
            raise ValueError("Trusted region is empty; check L_trust versus grid.")

    # ------------------------------------------------------------------
    # Basic geometric properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.Nz, self.Ny, self.Nx)

    @property
    def vector_shape(self) -> tuple[int, int, int, int]:
        return (self.Nz, self.Ny, self.Nx, 3)

    @property
    def tensor_shape(self) -> tuple[int, int, int, int, int]:
        return (self.Nz, self.Ny, self.Nx, 3, 3)

    @property
    def cell_volume(self) -> float:
        """Cell volume s^d = dx * dy * dz."""
        return self.dx * self.dy * self.dz

    # Keep cell_area as an alias for cell_volume for interface compat
    @property
    def cell_area(self) -> float:
        return self.cell_volume

    @property
    def padded_bounds(self) -> tuple[float, float]:
        return (-self.L_box, self.L_box)

    @property
    def trusted_bounds(self) -> tuple[float, float] | None:
        if self.L_trust is None:
            return None
        return (-self.L_trust, self.L_trust)

    @property
    def pad_width(self) -> float:
        if self.L_trust is None:
            return 0.0
        return self.L_box - self.L_trust

    @property
    def trusted_shape(self) -> tuple[int, int, int]:
        return (
            int(np.sum(self.trust_z_mask)),
            int(np.sum(self.trust_y_mask)),
            int(np.sum(self.trust_x_mask)),
        )

    @property
    def trusted_x_indices(self) -> np.ndarray:
        return np.where(self.trust_x_mask)[0]

    @property
    def trusted_y_indices(self) -> np.ndarray:
        return np.where(self.trust_y_mask)[0]

    @property
    def trusted_z_indices(self) -> np.ndarray:
        return np.where(self.trust_z_mask)[0]

    @property
    def trusted_slices(self) -> tuple[slice, slice, slice]:
        """
        Contiguous slices covering the trusted interior region.
        """
        iz = self.trusted_z_indices
        iy = self.trusted_y_indices
        ix = self.trusted_x_indices

        if len(iz) == 0 or len(iy) == 0 or len(ix) == 0:
            raise ValueError("Trusted region is empty; check L_trust versus grid.")

        return (
            slice(iz[0], iz[-1] + 1),
            slice(iy[0], iy[-1] + 1),
            slice(ix[0], ix[-1] + 1),
        )

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """
        Matplotlib-friendly plotting extent for the full padded box (xy plane).
        """
        return (
            self.x[0],
            self.x[-1] + self.dx,
            self.y[0],
            self.y[-1] + self.dy,
        )

    @property
    def trusted_extent(self) -> list[float] | None:
        if self.L_trust is None:
            return None
        return [-self.L_trust, self.L_trust, -self.L_trust, self.L_trust]

    # ------------------------------------------------------------------
    # Field allocation helpers
    # ------------------------------------------------------------------
    def zeros_scalar(self) -> np.ndarray:
        return np.zeros((self.Nz, self.Ny, self.Nx), dtype=np.float64)

    def zeros_vector(self) -> np.ndarray:
        return np.zeros((self.Nz, self.Ny, self.Nx, 3), dtype=np.float64)

    def zeros_tensor(self) -> np.ndarray:
        return np.zeros((self.Nz, self.Ny, self.Nx, 3, 3), dtype=np.float64)

    def ones_scalar(self) -> np.ndarray:
        return np.ones((self.Nz, self.Ny, self.Nx), dtype=np.float64)

    def ones_vector(self) -> np.ndarray:
        return np.ones((self.Nz, self.Ny, self.Nx, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def coordinates_as_particles(self) -> np.ndarray:
        """
        Return all grid nodes flattened as particle positions of shape (Np, 3).
        """
        return np.stack(
            [self.X.ravel(), self.Y.ravel(), self.Z.ravel()],
            axis=1,
        )

    def trusted_coordinates_as_particles(self) -> np.ndarray:
        """
        Return trusted interior grid nodes as particle positions of shape (Np_trusted, 3).
        """
        return np.stack(
            [
                self.X[self.trust_mask],
                self.Y[self.trust_mask],
                self.Z[self.trust_mask],
            ],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Region restriction helpers
    # ------------------------------------------------------------------
    def restrict_to_trusted(self, field: np.ndarray) -> np.ndarray:
        """
        Restrict a scalar, vector, or tensor field on the full padded box to the
        trusted interior.

        Supported shapes:
            scalar field: (Nz, Ny, Nx)
            vector field: (Nz, Ny, Nx, 3)
            tensor field: (Nz, Ny, Nx, 3, 3)
        """
        sz, sy, sx = self.trusted_slices

        if field.ndim == 3:
            return field[sz, sy, sx]
        if field.ndim == 4:
            return field[sz, sy, sx, :]
        if field.ndim == 5:
            return field[sz, sy, sx, :, :]

        raise ValueError(
            f"Unsupported field shape {field.shape}; expected 3D, 4D, or 5D array."
        )

    def apply_trust_mask(
        self,
        field: np.ndarray,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Return a copy of the field where points outside the trusted region are
        replaced by fill_value.

        Supported shapes:
            scalar field: (Nz, Ny, Nx)
            vector field: (Nz, Ny, Nx, 3)
            tensor field: (Nz, Ny, Nx, 3, 3)
        """
        out = np.array(field, copy=True)

        if field.ndim == 3:
            out[~self.trust_mask] = fill_value
            return out
        if field.ndim == 4:
            out[~self.trust_mask, :] = fill_value
            return out
        if field.ndim == 5:
            out[~self.trust_mask, :, :] = fill_value
            return out

        raise ValueError(
            f"Unsupported field shape {field.shape}; expected 3D, 4D, or 5D array."
        )

    # ------------------------------------------------------------------
    # Point-location helpers
    # ------------------------------------------------------------------
    def points_in_box(self, positions: np.ndarray) -> np.ndarray:
        """
        Boolean mask for points lying in the full padded box.
        Uses L_box for x/y and L_box_z for z (supports non-cubic grids).
        """
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return (
            (x >= -self.L_box)
            & (x < self.L_box)
            & (y >= -self.L_box)
            & (y < self.L_box)
            & (z >= -self.L_box_z)
            & (z < self.L_box_z)
        )

    def points_in_trusted_region(self, positions: np.ndarray) -> np.ndarray:
        """
        Boolean mask for points lying in the trusted interior.
        Uses L_trust for x/y and L_trust_z for z (supports non-cubic grids).
        """
        if self.L_trust is None:
            return np.ones(positions.shape[0], dtype=bool)

        L_tz = self.L_trust_z if self.L_trust_z is not None else self.L_trust
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return (
            (x >= -self.L_trust)
            & (x <= self.L_trust)
            & (y >= -self.L_trust)
            & (y <= self.L_trust)
            & (z >= -L_tz)
            & (z <= L_tz)
        )

    def nearest_grid_indices(
        self, positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return nearest gridpoint indices (k, j, i) for a set of positions.

        Indices are clipped to the padded-box grid.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        i = np.rint((x - self.x[0]) / self.dx).astype(int)
        j = np.rint((y - self.y[0]) / self.dy).astype(int)
        k = np.rint((z - self.z[0]) / self.dz).astype(int)

        i = np.clip(i, 0, self.Nx - 1)
        j = np.clip(j, 0, self.Ny - 1)
        k = np.clip(k, 0, self.Nz - 1)

        return k, j, i

    def cell_indices(
        self, positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return lower-left-back cell indices (k, j, i) for trilinear interpolation
        with periodic wrapping.

        Indices are wrapped modulo N so that the last cell [N-1] wraps to [0].
        The caller should use (index + 1) % N for the upper corner.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        i = np.floor((x - self.x[0]) / self.dx).astype(int) % self.Nx
        j = np.floor((y - self.y[0]) / self.dy).astype(int) % self.Ny
        k = np.floor((z - self.z[0]) / self.dz).astype(int) % self.Nz

        return k, j, i


# Alias for convenience — many modules use the shorter name
Grid3D = CartesianGrid3D
