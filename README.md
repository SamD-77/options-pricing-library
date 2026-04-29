# Options Pricing Library
A quantitative finance project implementing core derivatives pricing models, starting with the Black–Scholes model for European options. This repository will expand to include Greeks, implied volatility, binomial trees, Monte Carlo simulation, and volatility surface visualization.

---

## Black–Scholes Model

The model prices European options under lognormal asset dynamics.

### **Key formulas**
$$
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)t}{\sigma\sqrt{t}}
$$

$$
d_2 = d_1 - \sigma\sqrt{t}
$$


**Call price**

$$
C = S N(d_1) - K e^{-rt} N(d_2)
$$

**Put price**

$$
P = K e^{-rt} N(-d_2) - S N(-d_1)
$$


---
## Current Features
- Black–Scholes pricing  
  - European call  
  - European put  

---
## Upcoming Features
- Greeks (Delta, Gamma, Vega, Theta, Rho)  
- Implied volatility solver (Newton–Raphson)  
- Binomial tree pricing (CRR model)  
- Monte Carlo simulation  
- Volatility smile and surface  
- Jupyter notebooks with visualizations  

---
