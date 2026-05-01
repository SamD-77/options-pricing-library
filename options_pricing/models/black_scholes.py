import math
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, r, t, sigma):
        """
        Black-Scholes model for European option pricing.

        Parameters
        ----------
        S : float
            Current underlying price
        K : float
            Strike price
        r : float
            Risk-free interest rate
        t : float
            Time to maturity
        sigma : float
            Annualized volatility
        """
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.sigma = sigma

        self.d1 = (math.log(S / K ) + (r + (sigma**2 / 2)) * t) / (sigma * math.sqrt(t))
        self.d2 = self.d1 - sigma * math.sqrt(t)

    def call_price(self):
        """
        Compute the price of a European call option under the Black-Scholes model.
        
        Returns
        -------
        float
            The Black-Scholes price of the European call option.
        """
        return self.S * norm.cdf(self.d1) - self.K * math.exp(-self.r * self.t) * norm.cdf(self.d2)
    
    def put_price(self):
        """
        Compute the price of a European put option under the Black-Scholes model.

        Returns
        -------
        float
            The Black-Scholes price of the European put option.
        """
        return self.K * math.exp(-self.r * self.t) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)