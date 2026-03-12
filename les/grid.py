import numpy as np


class Grid2D:
    """
    Uniform Cartesian grid on the padded computational box [-L_box, L_box]^2,
    together with trusted-interior bookkeeping for [-L_trust, L_trust]^2.

    Notes
    -----
    - endpoint=False is used so the grid stays uniform and FFT-friendly.
    - the trusted region is tracked by boolean masks and contiguous slices.
    - scalar fields are stored with shape (Ny, Nx)
    - vector fields are stored with shape (Ny, Nx, 2)
    - 2x2 tensor fields are stored with shape (Ny, Nx, 2, 2)
    """

    def __init__(self, L_box: float, Nx: int, Ny: int, L_trust: float | None = None):
        self.L_box = float(L_box)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.L_trust = None if L_trust is None else float(L_trust)

        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Nx and Ny must both be at least 2.")

        self.x = np.linspace(-self.L_box, self.L_box, self.Nx, endpoint=False)
        self.y = np.linspace(-self.L_box, self.L_box, self.Ny, endpoint=False)

        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        if self.L_trust is None:
            self.trust_x_mask = np.ones(self.Nx, dtype=bool)
            self.trust_y_mask = np.ones(self.Ny, dtype=bool)
        else:
            self.trust_x_mask = np.abs(self.x) <= self.L_trust
            self.trust_y_mask = np.abs(self.y) <= self.L_trust

        self.trust_mask = np.outer(self.trust_y_mask, self.trust_x_mask)

        if not np.any(self.trust_x_mask) or not np.any(self.trust_y_mask):
            raise ValueError("Trusted region is empty; check L_trust versus grid.")

    # ------------------------------------------------------------------
    # Basic geometric properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return (self.Ny, self.Nx)

    @property
    def vector_shape(self) -> tuple[int, int, int]:
        return (self.Ny, self.Nx, 2)

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (self.Ny, self.Nx, 2, 2)

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

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
    def trusted_shape(self) -> tuple[int, int]:
        return (int(np.sum(self.trust_y_mask)), int(np.sum(self.trust_x_mask)))

    @property
    def trusted_x_indices(self) -> np.ndarray:
        return np.where(self.trust_x_mask)[0]

    @property
    def trusted_y_indices(self) -> np.ndarray:
        return np.where(self.trust_y_mask)[0]

    @property
    def trusted_slices(self) -> tuple[slice, slice]:
        """
        Contiguous slices covering the trusted interior region.
        """
        ix = self.trusted_x_indices
        iy = self.trusted_y_indices

        if len(ix) == 0 or len(iy) == 0:
            raise ValueError("Trusted region is empty; check L_trust versus grid.")

        return (slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1))

    @property
    def extent(self) -> list[float]:
        """
        Matplotlib-friendly plotting extent for the full padded box.
        """
        return [
            self.x[0],
            self.x[-1] + self.dx,
            self.y[0],
            self.y[-1] + self.dy,
        ]

    @property
    def trusted_extent(self) -> list[float] | None:
        """
        Matplotlib-friendly plotting extent for the trusted interior.
        """
        if self.L_trust is None:
            return None
        return [-self.L_trust, self.L_trust, -self.L_trust, self.L_trust]

    # ------------------------------------------------------------------
    # Field allocation helpers
    # ------------------------------------------------------------------
    def zeros_scalar(self) -> np.ndarray:
        return np.zeros((self.Ny, self.Nx), dtype=np.float64)

    def zeros_vector(self) -> np.ndarray:
        return np.zeros((self.Ny, self.Nx, 2), dtype=np.float64)

    def zeros_tensor(self) -> np.ndarray:
        return np.zeros((self.Ny, self.Nx, 2, 2), dtype=np.float64)

    def ones_scalar(self) -> np.ndarray:
        return np.ones((self.Ny, self.Nx), dtype=np.float64)

    def ones_vector(self) -> np.ndarray:
        return np.ones((self.Ny, self.Nx, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def coordinates_as_particles(self) -> np.ndarray:
        """
        Return all grid nodes flattened as particle positions of shape (Np, 2).
        """
        return np.stack([self.X.ravel(), self.Y.ravel()], axis=1)

    def trusted_coordinates_as_particles(self) -> np.ndarray:
        """
        Return trusted interior grid nodes flattened as particle positions
        of shape (Np_trusted, 2).
        """
        return np.stack(
            [self.X[self.trust_mask], self.Y[self.trust_mask]],
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
            scalar field: (Ny, Nx)
            vector field: (Ny, Nx, 2)
            tensor field: (Ny, Nx, 2, 2)
        """
        sy, sx = self.trusted_slices

        if field.ndim == 2:
            return field[sy, sx]
        if field.ndim == 3:
            return field[sy, sx, :]
        if field.ndim == 4:
            return field[sy, sx, :, :]

        raise ValueError(
            f"Unsupported field shape {field.shape}; expected 2D, 3D, or 4D array."
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
            scalar field: (Ny, Nx)
            vector field: (Ny, Nx, 2)
            tensor field: (Ny, Nx, 2, 2)
        """
        out = np.array(field, copy=True)

        if field.ndim == 2:
            out[~self.trust_mask] = fill_value
            return out

        if field.ndim == 3:
            out[~self.trust_mask, :] = fill_value
            return out

        if field.ndim == 4:
            out[~self.trust_mask, :, :] = fill_value
            return out

        raise ValueError(
            f"Unsupported field shape {field.shape}; expected 2D, 3D, or 4D array."
        )

    # ------------------------------------------------------------------
    # Point-location helpers
    # ------------------------------------------------------------------
    def points_in_box(self, positions: np.ndarray) -> np.ndarray:
        """
        Boolean mask for points lying in the full padded box.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        return (
            (x >= -self.L_box)
            & (x < self.L_box)
            & (y >= -self.L_box)
            & (y < self.L_box)
        )

    def points_in_trusted_region(self, positions: np.ndarray) -> np.ndarray:
        """
        Boolean mask for points lying in the trusted interior.
        """
        if self.L_trust is None:
            return np.ones(positions.shape[0], dtype=bool)

        x = positions[:, 0]
        y = positions[:, 1]
        return (
            (x >= -self.L_trust)
            & (x <= self.L_trust)
            & (y >= -self.L_trust)
            & (y <= self.L_trust)
        )

    def nearest_grid_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return nearest gridpoint indices (j, i) for a set of positions.

        Indices are clipped to the padded-box grid.
        """
        x = positions[:, 0]
        y = positions[:, 1]

        i = np.rint((x - self.x[0]) / self.dx).astype(int)
        j = np.rint((y - self.y[0]) / self.dy).astype(int)

        i = np.clip(i, 0, self.Nx - 1)
        j = np.clip(j, 0, self.Ny - 1)

        return j, i

    def cell_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return lower-left cell indices (j, i) for bilinear interpolation.

        Indices are clipped so that (i+1, j+1) stays inside the array bounds.
        """
        x = positions[:, 0]
        y = positions[:, 1]

        i = np.floor((x - self.x[0]) / self.dx).astype(int)
        j = np.floor((y - self.y[0]) / self.dy).astype(int)

        i = np.clip(i, 0, self.Nx - 2)
        j = np.clip(j, 0, self.Ny - 2)

        return j, i