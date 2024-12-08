"""
Merton Jump Diffusion model for  option pricing.

Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility of the underlying
    lam: Intensity of jump arrivals (lambda)
    m: Mean jump size (log-normal jumps)
    v: Volatility of jump sizes
    k: Number of jumps

"""
import numpy as np
import scipy.stats as si
import scipy.special as ss

def d1(S, T, r, sigma, K):
    '''
    Calculates the d1 parameter used in the Black-Scholes formula.
    '''
    return (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

def d2(S, T, r, sigma, K):
    '''
    Calculates the d2 parameter used in the Black-Scholes formula.
    '''
    return d1(S, T, r, sigma, K) - sigma * np.sqrt(T)

def call_option_price(S, T, r, sigma, K) -> float:
    '''
    Calculates the price of a call option using the Black-Scholes formula.
    Returns: call_option_price (float): The price of the call option.
    '''
    return (S * si.norm.cdf(d1(S, T, r, sigma, K)) - K * np.exp(-r * T) * si.norm.cdf(d2(S, T, r, sigma, K))).item()

def put_option_price(S, T, r, sigma, K) -> float:
    '''
    Calculates the price of a put option using the Black-Scholes formula.
    Returns: put_option_price (float): The price of the put option.
    '''
    return (K * np.exp(-r * T) * si.norm.cdf(-d2(S, T, r, sigma, K)) - S * si.norm.cdf(-d1(S, T, r, sigma, K))).item()

def jump_call(S, T, r, m, v, lam, sigma, K, k: int) -> float:
    '''
    Calculates the price of a call option using the Merton Jump Diffusion model.
    Returns: call_option_price (float): The price of the call option.
    '''
    sigma_k = np.sqrt(sigma**2 + k * v**2 / T)
    r_k = r - lam * (m - 1) + k * np.log(m) / T
    return (np.exp(-lam * T) * (lam * T)**k / ss.factorial(k) * call_option_price(S, T, r_k, sigma_k, K)).item()

def jump_put(S, T, r, m, v, lam, sigma, K, k: int) -> float:
    '''
    Calculates the price of a put option using the Merton Jump Diffusion model.
    Returns: put_option_price (float): The price of the put option.
    '''
    sigma_k = np.sqrt(sigma**2 + k * v**2 / T)
    r_k = r - lam * (m - 1) + k * np.log(m) / T
    return (np.exp(-lam * T) * (lam * T)**k / ss.factorial(k) * put_option_price(S, T, r_k, sigma_k, K)).item()

def viz_jump_paths(S, T, r, m, v, lam, steps, Npaths, sigma) -> np.ndarray:
    '''
    Simulates jump diffusion paths.
    Args:
    - S (float): Current stock price
    - T (float): Time to maturity (in years)
    - r (float): Risk-free rate
    - m (float): Mean jump size (log-normal jumps)
    - v (float): Volatility of jump sizes
    - lam (float): Intensity of jump arrivals (lambda)
    - steps (int): Number of time steps
    - Npaths (int): Number of paths to simulate
    - sigma (float): Volatility of the underlying
    Returns: jump_diffusion_paths (np.ndarray): The jump diffusion paths.
    '''
    size = (steps, Npaths)
    dt = T / steps
    Z = np.random.normal(size=size)

    poi_rv = np.multiply(np.random.poisson(lam * dt, size=size), np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma**2 / 2 - lam * (m + v**2 * 0.5)) * dt + sigma * np.sqrt(dt) * Z), axis=0)
    return np.exp(geo + poi_rv) * S
    