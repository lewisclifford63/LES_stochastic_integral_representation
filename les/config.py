from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Configuration for the padded-box R^2 Navier-Stokes / LES prototype.

    Default parameters are aligned with Qian et al. (2025) Experiment 1
    scales for a 2D flow in R^2 (no walls):

        nu = 0.3,  Re = 1000,  U0 ~ 32,  L ~ 3*pi
        mesh size  s ~ L / 50 ~ 0.19
        time step  dt = 0.001
        T = 0.09  (90 steps)

    The physical idea is:
        - approximate the whole space R^2 by a larger computational box
          [-L_box, L_box]^2
        - only trust results in the smaller interior
          [-L_trust, L_trust]^2

    The stochastic representation uses one Brownian diffusion sample
    per particle per time step (single trajectory, no Monte Carlo average).
    """

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    dim: int = 2
    nu: float = 0.3
    t0: float = 0.0
    T: float = 0.09
    dt: float = 0.001

    # ------------------------------------------------------------------
    # Geometry: trusted region + padding
    #
    # L_trust ~ 3*pi / 2 ~ 4.71   (half the observation scale)
    # pad = 4.0  (accommodates drift U0*T ~ 2.9 plus Brownian spread)
    # ------------------------------------------------------------------
    L_trust: float = 4.712
    pad: float = 4.0

    # ------------------------------------------------------------------
    # Grid resolution on the full padded computational box
    #
    # N = 90 gives dx ~ 0.194, close to Qian's s = 3*pi/50 ~ 0.189
    # ------------------------------------------------------------------
    Nx: int = 90
    Ny: int = 90

    # ------------------------------------------------------------------
    # Particle / stochastic step parameters
    # ------------------------------------------------------------------
    one_particle_per_grid_node: bool = True
    reseed_rng_each_run: bool = True
    seed: int = 12345

    # ------------------------------------------------------------------
    # Gaussian kernel / reconstruction / filtering parameters
    #
    # filter_width ~ 1.5 * dx ~ 0.29
    # ------------------------------------------------------------------
    filter_width: float = 0.29
    filter_cutoff_sigma: float = 3.0

    # ------------------------------------------------------------------
    # Pressure representation parameters
    # ------------------------------------------------------------------
    pressure_cutoff_radius: float | None = None
    pressure_softening: float = 1.0e-10

    # ------------------------------------------------------------------
    # Diagnostics / output
    # ------------------------------------------------------------------
    save_every: int = 30
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