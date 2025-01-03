import numpy as np
import scipy.stats as si
from functools import lru_cache

class BlackScholesModel:
    '''
    Class to calculate the price of a call or put option using the Black-Scholes model.
    '''

    def __init__(self, S, K, T, r, sigma, q):
        '''
        Args:
        - S (float): The spot price of the underlying stock.
        - K (float): The strike price of the option.
        - T (float): The time to maturity of the option.
        - r (float): The risk-free rate.
        - sigma (float): The volatility of the underlying stock.
        - q (float): The dividend yield of the underlying stock.
        '''
        self.S = S 
        self.K = K 
        self.T = T 
        self.r = r 
        self.sigma = sigma
        self.q = q

    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}

    @lru_cache(maxsize=128)
    def d1(self):
        '''
        Calculates the d1 parameter used in the Black-Scholes formula.
        '''
        return (np.log(self.S / self.K) + (self.r -self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        '''
        Calculates the d2 parameter used in the Black-Scholes formula.
        '''
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_option_price(self) -> float:
        '''
        Calculates the price of a call option using the Black-Scholes formula.
        Returns: call_option_price (float): The price of the call option.
        '''
        return (self.S*np.exp(-self.q*self.T) * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0)).item()
    
    def put_option_price(self) -> float:
        '''
        Calculates the price of a put option using the Black-Scholes formula.
        Returns: put_option_price (float): The price of the put option.
        '''
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S*np.exp(-self.q*self.T) * si.norm.cdf(-self.d1(), 0.0, 1.0)).item()

    # Greeks Calculations 
    def delta(self):
        '''
        First derivative of option price with respect to underlying price
       '''
        if self.is_call:
            return np.exp(-self.q * self.T) * si.norm.cdf(self.d1(), 0.0, 1.0)
        return -np.exp(-self.q * self.T) * si.norm.cdf(-self.d1(), 0.0, 1.0)

    def gamma(self):
        '''
        Second derivative of option price with respect to underlying price
        '''
        return (np.exp(-self.q * self.T) * si.norm.pdf(self.d1(), 0.0, 1.0)) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self):
        '''
        First derivative of option price with respect to time
        '''
        term1 = -(self.S * self.sigma * np.exp(-self.q * self.T) * si.norm.pdf(self.d1(), 0.0, 1.0)) / (2 * np.sqrt(self.T))
        if self.is_call:
            return term1 - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0) + self.q * self.S * np.exp(-self.q * self.T) * si.norm.cdf(self.d1(), 0.0, 1.0)
        return term1 + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.q * self.S * np.exp(-self.q * self.T) * si.norm.cdf(-self.d1(), 0.0, 1.0)

    def vega(self):
        '''
        First derivative of option price with respect to volatility
        '''
        return self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * si.norm.pdf(self.d1(), 0.0, 1.0)

    # Implied Volatility Calculator
    def implied_volatility(self, market_price, is_call=True, precision=0.00001):
        '''
        Calculate implied volatility using Newton-Raphson method
        Args:
        - market_price (float): The market price of the option.
        - is_call (bool): True if the option is a call option, False if it is a put option.
        - precision (float): The precision of the calculation.
        Returns: implied_volatility (float): The implied volatility of the option.
        '''
        max_iterations = 100
        vol = 0.5  # Initial guess
        
        for i in range(max_iterations):
            if is_call:
                price = self.call_option_price()
            else:
                price = self.put_option_price()
                
            diff = market_price - int(price)
            
            if abs(diff) < precision:
                return vol
                
            vega = self.vega()
            if vega == 0:
                return None
                
            vol = vol + diff/vega
            
            if vol <= 0:
                return None
                
        return None  # Failed to converge

    def update_sigma(self, sigma):
        '''
        Update the volatility parameter.
        Args:
        - sigma (float): The new volatility parameter.
        '''
        self.sigma = sigma
    
    def check_put_call_parity(self):
        '''
        Check if put-call parity holds
        '''
        call_price = self.call_option_price()
        put_price = self.put_option_price()
        
        left_side = call_price - put_price
        right_side = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        
        return abs(left_side - right_side) < 1e-10