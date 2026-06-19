import math
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, r, t, sigma):
        """
        Black-Scholes model for European option pricing.

        Parameters
        ----------
        S : float
            Current underlying asset price.
        K : float
            Strike price.
        r : float
            Risk-free interest rate.
        t : float
            Time to maturity (in years).
        sigma : float
            Annualized volatility of the underlying asset.
        """
        self.S = S
        self.K = K
        self.r = r
        self.t = t
        self.sigma = sigma

    @property    
    def d1(self):
        return (math.log(self.S / self.K ) + (self.r + (self.sigma**2 / 2)) * self.t) / (self.sigma * math.sqrt(self.t))
    
    @property
    def d2(self):
        return self.d1 - self.sigma * math.sqrt(self.t)

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
    
    def price_with_sigma(self, sigma, option_type="call"):
        """
        Compute the option price using a temporary volatility value without permanently modifying self.sigma.

        Parameters
        ----------
        sigma : float
            Temporary volatility value to use for pricing.
        option_type : str
            Type of option: "call" or "put". Default is "call".

        Returns
        -------
        float
            The option price.
        """
        original_sigma = self.sigma
        self.sigma = sigma

        option_type = option_type.lower()
        if option_type == "call":
            price = self.call_price()
        
        elif option_type == "put":
            price = self.put_price()
        
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        self.sigma = original_sigma
        return price

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
        
    def implied_vol(self, market_price, option_type="call", sigma_est=0.2, tol=1e-7, max_iter=200):
        """
        Compute implied volatility using a hybrid approach combining Newton-Raphson with Bisection.

        Parameters
        ----------
        market_price : float
            Market price of the option.
        option_type : str, optional
            "call" or "put". Default is "call".
        sigma_est : float, optional
            Initial volatility guess. Default is 0.2.
        tol : float, optional
            Tolerance for price difference. Default is 1e-7.
        max_iter : int, optional
            Maximum number of iterations. Default is 100.

        Returns
        -------
        float
            Implied volatility.

        Note
        ----
        This method updates self.sigma to the solved implied volatility.    
        """
        option_type = option_type.lower()

        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        
        sigma = sigma_est
        low, high = 1e-5, 5.0

        for _ in range(max_iter):
            self.sigma = sigma

            if option_type == "call":
                price = self.call_price()
            else:
                price = self.put_price()
            
            diff = price - market_price

            vega = self.vega()

            if abs(vega) < 1e-6:
                sigma = (low + high) / 2
            else:
                sigma = sigma - diff / vega

            sigma = max(low, min(high, sigma))

            if abs(diff) < tol:
                break

            if diff > 0:
                high = sigma
            else:
                low = sigma

        return sigma