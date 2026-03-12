from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Configuration for the 2D padded-box LES prototype on R^2.

    Idea:
        We approximate the unbounded domain R^2 by solving on a larger square
        [-L_box, L_box]^2, but only regard the smaller interior square
        [-L_trust, L_trust]^2 as physically trustworthy.

    This helps reduce boundary contamination near the edges of the numerical box.
    """

    # Physical parameters
    dim: int = 2
    nu: float = 0.01
    t0: float = 0.0
    T: float = 1.0
    dt: float = 0.01

    # Trusted interior region: [-L_trust, L_trust]^2
    L_trust: float = 2.0

    # Padding width added on each side of the trusted box
    pad: float = 1.0

    # Grid resolution on the full computational box
    Nx: int = 192
    Ny: int = 192

    # LES / filter parameters
    filter_width: float = 0.15
    filter_cutoff_sigma: float = 3.0

    # Randomness
    seed: int = 12345

    # Output / plotting
    save_every: int = 10
    plot_initial_field: bool = True
    plot_reconstruction_test: bool = True

    @property
    def L_box(self) -> float:
        """
        Half-width of the full computational box.
        """
        return self.L_trust + self.pad

    @property
    def trusted_bounds(self) -> tuple[float, float]:
        """
        Returns (-L_trust, L_trust).
        """
        return (-self.L_trust, self.L_trust)

    @property
    def box_bounds(self) -> tuple[float, float]:
        """
        Returns (-L_box, L_box).
        """
        L = self.L_box
        return (-L, L)

    @property
    def num_steps(self) -> int:
        return int(round((self.T - self.t0) / self.dt))