import numpy as np
import math

class MonteCarlo:
    def __init__(self, S, K, r, T, sigma):
        """
        Monte Carlo model for European option pricing under geometric Brownian motion (GBM).

        Parameters
        ----------
        S : float
            Initial underlying asset price.
        K : float
            Strike price.
        r : float
            Risk-free interest rate.
        T : float
            Time to maturity (in years).
        sigma : float
            Annualized volatility of the underlying asset.
        """
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma

    def simulate_paths(self, n_paths, n_steps):
        """
        Simulate GBM price paths using the exact analytical solution.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths to simulate.
        n_steps : int
            Number of time steps per path.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps + 1) containing simulated price paths.
        """
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S

        Z = np.random.normal(size=(n_paths, n_steps))

        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        log_returns = np.cumsum(drift + diffusion, axis=1)
        paths[:, 1:] = self.S * np.exp(log_returns)

        return paths

    def price_european(self, option_type="call", n_paths=10000, n_steps=100):
        """
        Compute the Monte Carlo price of a European option.

        Parameters
        ----------
        option_type : str, optional
            "call" or "put". Default is "call".
        n_paths : int, optional
            Number of Monte Carlo paths. Default is 10,000.
        n_steps : int, optional
            Number of time steps per path. Default is 100.

        Returns
        -------
        float
            Estimated Monte Carlo price of the option.
        """
        paths = self.simulate_paths(n_paths, n_steps)
        ST = paths[:, -1]

        if option_type == "call":
            payoff = np.maximum(ST - self.K, 0)
        elif option_type == "put":
            payoff = np.maximum(self.K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return math.exp(-self.r * self.T) * payoff.mean()
