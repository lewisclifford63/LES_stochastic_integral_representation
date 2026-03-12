from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Configuration for the padded-box R^2 Navier-Stokes / LES prototype.

    The physical idea is:
        - approximate the whole space R^2 by a larger computational box
          [-L_box, L_box]^2
        - only trust results in the smaller interior
          [-L_trust, L_trust]^2

    The stochastic representation will later use one Brownian diffusion sample
    per particle per time step.
    """

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    dim: int = 2
    nu: float = 0.01
    t0: float = 0.0
    T: float = 0.5
    dt: float = 0.02

    # ------------------------------------------------------------------
    # Geometry: trusted region + padding
    # ------------------------------------------------------------------
    L_trust: float = 1.5
    pad: float = 0.5

    # ------------------------------------------------------------------
    # Grid resolution on the full padded computational box
    # ------------------------------------------------------------------
    Nx: int = 64
    Ny: int = 64

    # ------------------------------------------------------------------
    # Particle / stochastic step parameters
    # ------------------------------------------------------------------
    one_particle_per_grid_node: bool = True
    reseed_rng_each_run: bool = True
    seed: int = 12345

    # ------------------------------------------------------------------
    # Gaussian kernel / reconstruction / filtering parameters
    # ------------------------------------------------------------------
    filter_width: float = 0.12
    filter_cutoff_sigma: float = 3.0

    # ------------------------------------------------------------------
    # Pressure representation parameters
    # ------------------------------------------------------------------
    pressure_cutoff_radius: float | None = None
    pressure_softening: float = 1.0e-10

    # ------------------------------------------------------------------
    # Diagnostics / output
    # ------------------------------------------------------------------
    save_every: int = 5
    plot_initial_field: bool = True
    plot_pressure_each_save: bool = False
    plot_velocity_each_save: bool = True
    trusted_only_diagnostics: bool = True
    verbose: bool = True

    @property
    def L_box(self) -> float:
        """
        Half-width of the full padded computational box.
        """
        return self.L_trust + self.pad

    @property
    def trusted_bounds(self) -> tuple[float, float]:
        """
        Bounds of the trusted interior interval.
        """
        return (-self.L_trust, self.L_trust)

    @property
    def box_bounds(self) -> tuple[float, float]:
        """
        Bounds of the full computational interval.
        """
        L = self.L_box
        return (-L, L)

    @property
    def num_steps(self) -> int:
        """
        Number of time steps.
        """
        return int(round((self.T - self.t0) / self.dt))

    @property
    def diffusion_scale(self) -> float:
        """
        Brownian increment scale sqrt(2 * nu * dt).
        """
        return (2.0 * self.nu * self.dt) ** 0.5