import numpy as np
import scipy.stats as si

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