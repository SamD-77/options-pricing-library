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
    
    def delta(self, option_type="call"):
        """
        Compute the Delta of a European option under the Black-Scholes model.

        Parameters
        ----------
        option_type : str
            Type of option: "call" or "put". Default is "call".

        Returns
        -------
        float
            The Delta of the option.
        """
        option_type = option_type.lower()
        
        if option_type == "call":
            return norm.cdf(self.d1)
        
        elif option_type == "put":
            return norm.cdf(self.d1) - 1
        
        else: 
            raise ValueError("option_type must be 'call' or 'put'")

    def gamma(self):
        """
        Compute the Gamma of a European option under the Black-Scholes model.

        Returns
        -------
        float
            The Gamma of the option.
        """
        return (norm.pdf(self.d1)) / (self.S * self.sigma * math.sqrt(self.t))
    
    def vega(self):
        """
        Compute the Vega of a European option under the Black-Scholes model.

        Returns
        -------
        float
            The Vega of the option.
        """
        return self.S * norm.pdf(self.d1) * math.sqrt(self.t)
    
    def theta(self, option_type="call"):
        """
        Compute the Theta of a European option under the Black-Scholes model.

        Parameters
        ----------
        option_type : str
            Type of option: "call" or "put". Default is "call".

        Returns
        -------
        float
            The Theta of the option.
        """
        option_type = option_type.lower()

        first_term = -((self.S * norm.pdf(self.d1) * self.sigma) / (2 * math.sqrt(self.t)))
        
        if option_type == "call":
            return first_term - self.r * self.K * math.exp(-self.r * self.t) * norm.cdf(self.d2)
        
        elif option_type == "put":
            return first_term + self.r * self.K * math.exp(-self.r * self.t) * norm.cdf(-self.d2)
        
        else: 
            raise ValueError("option_type must be 'call' or 'put'")
        
    def rho(self, option_type="call"):
        """
        Compute the Rho of a European option under the Black-Scholes model.

        Parameters
        ----------
        option_type : str
            Type of option: "call" or "put". Default is "call".

        Returns
        -------
        float
            The Rho of the option.
        """
        option_type = option_type.lower()

        if option_type == "call":
            return self.K * self.t * math.exp(-self.r * self.t) * norm.cdf(self.d2)
        
        elif option_type == "put":
            return -self.K * self.t * math.exp(-self.r * self.t) * norm.cdf(-self.d2)
        
        else: 
            raise ValueError("option_type must be 'call' or 'put'")