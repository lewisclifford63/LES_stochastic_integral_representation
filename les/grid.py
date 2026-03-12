import numpy as np


class Grid2D:
    """
    Uniform Cartesian grid on a padded computational box [-L_box, L_box]^2,
    together with masks and index ranges for the trusted interior region
    [-L_trust, L_trust]^2.

    We use endpoint=False so the grid remains uniform and FFT-friendly later
    if needed.
    """

    def __init__(self, L_box: float, Nx: int, Ny: int, L_trust: float | None = None):
        self.L_box = float(L_box)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.L_trust = None if L_trust is None else float(L_trust)

        self.x = np.linspace(-self.L_box, self.L_box, self.Nx, endpoint=False)
        self.y = np.linspace(-self.L_box, self.L_box, self.Ny, endpoint=False)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        # Interior/trusted region bookkeeping
        if self.L_trust is None:
            self.trust_x_mask = np.ones(self.Nx, dtype=bool)
            self.trust_y_mask = np.ones(self.Ny, dtype=bool)
        else:
            self.trust_x_mask = np.abs(self.x) <= self.L_trust
            self.trust_y_mask = np.abs(self.y) <= self.L_trust

        self.trust_mask = np.outer(self.trust_y_mask, self.trust_x_mask)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Ny, self.Nx)

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
        Returns contiguous slices covering the trusted interior region.

        Assumes the trusted region corresponds to one contiguous block in x and y,
        which is true for the current symmetric box setup.
        """
        ix = self.trusted_x_indices
        iy = self.trusted_y_indices

        if len(ix) == 0 or len(iy) == 0:
            raise ValueError("Trusted region is empty; check L_trust versus grid.")

        return (slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1))

    def zeros_scalar(self) -> np.ndarray:
        return np.zeros((self.Ny, self.Nx), dtype=np.float64)

    def zeros_vector(self) -> np.ndarray:
        return np.zeros((self.Ny, self.Nx, 2), dtype=np.float64)

    def coordinates_as_particles(self) -> np.ndarray:
        """
        Returns all grid nodes flattened as particle positions of shape (Np, 2).
        """
        return np.stack([self.X.ravel(), self.Y.ravel()], axis=1)

    def trusted_coordinates_as_particles(self) -> np.ndarray:
        """
        Returns only trusted interior grid nodes flattened as particle positions
        of shape (Np_trusted, 2).
        """
        return np.stack(
            [self.X[self.trust_mask], self.Y[self.trust_mask]],
            axis=1,
        )

    def restrict_to_trusted(self, field: np.ndarray) -> np.ndarray:
        """
        Restrict a scalar or vector field on the full padded box to the trusted box.

        Supported shapes:
            scalar field: (Ny, Nx)
            vector field: (Ny, Nx, 2)
        """
        sy, sx = self.trusted_slices

        if field.ndim == 2:
            return field[sy, sx]
        if field.ndim == 3:
            return field[sy, sx, :]
        raise ValueError(
            f"Unsupported field shape {field.shape}; expected 2D or 3D array."
        )

    def apply_trust_mask(self, field: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """
        Return a copy of the field where points outside the trusted region are replaced
        by fill_value.

        Useful for plotting only the reliable interior while retaining the full array shape.
        """
        out = np.array(field, copy=True)

        if field.ndim == 2:
            out[~self.trust_mask] = fill_value
            return out

        if field.ndim == 3:
            out[~self.trust_mask, :] = fill_value
            return out

        raise ValueError(
            f"Unsupported field shape {field.shape}; expected 2D or 3D array."
        )