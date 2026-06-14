import math
import numpy as np

class BinomialTree():
    """
    Cox-Ross-Rubinstein (CRR) binomial tree model for option pricing.

    Supports European and American calls and puts.
    """
    def __init__(self, S, K, T, r, sigma, steps):
        """
        Parameters
        ----------
        S : float
            Spot price.
        K : float
            Strike price.
        r : float
            Risk-free rate (continuously compounded).
        T : float
            Time to maturity (in years).
        sigma : float
            Volatility.
        steps : int
            Number of time intervals in the tree.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        
        self.dt = self.T / self.steps
        self.discount = math.exp(-self.r * self.dt)

        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp(self.r * self.dt) - self.d) / (self.u - self.d)

        if not (0.0 < self.p < 1.0):
            raise ValueError("Risk-neutral probability p is not in (0,1). Check parameters.")
        
    def price(self, option_type="call", exercise="european"):
        """
        Price an option using the binomial tree.

        Parameters
        ----------
        option_type : {"call", "put"}, default "call"
            Type of option to price.
        exercise : {"european", "american"}, default "european"
            Exercise style of the option.

        Returns
        -------
        float
        Option price.
        """
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")
    
        if exercise not in {"european", "american"}:
            raise ValueError("exercise must be 'european' or 'american'.")

        return self._price_tree(option_type=option_type, exercise=exercise)
    
    def _terminal_payoffs(self, option_type):
        """
        Compute terminal payoffs at maturity for all nodes.
        """
        j = np.arange(self.steps + 1)
        S_T = self.S * (self.u ** j) * (self.d ** (self.steps - j))

        if option_type == "call":
            return np.maximum(S_T - self.K, 0.0)
        
        else:
            return np.maximum(self.K - S_T, 0.0)
        
    def _price_tree(self, option_type, exercise):
        """
        Backward induction on the binomial tree.
        """
        values = self._terminal_payoffs(option_type)

        for n in range(self.steps - 1, -1, -1):
            values = self.discount * (self.p * values[1:] + (1.0 - self.p) * values[:-1])

            if exercise == "american":
                j = np.arange(n + 1)
                S_n = self.S * (self.u ** j) * (self.d ** (n - j))

                if option_type == "call":
                    exercise_value = np.maximum(S_n - self.K, 0.0)
                
                else:
                    exercise_value = np.maximum(self.K - S_n, 0.0)

                values = np.maximum(values, exercise_value)

        return float(values[0])