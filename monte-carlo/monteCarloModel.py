import numpy as np
import scipy.stats as si

class MonteCarloModel:
    '''
    Class to calculate 
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
    
    def monte_carlo_model(self, Npaths, steps, random_generator=None):
        '''
        Simulates asset paths using the Monte Carlo model.
        Args:
        - Npaths (int): Number of paths to simulate.
        - steps (int): Number of time steps.
        - random_generator (str): The random number generator to use. Options: "quasi" or None.
        Returns:
        - [steps, N] Matrix of asset paths.
        '''
        size = (steps, Npaths)
        dt = self.T/steps

        # Generate random numbers
        if random_generator == "quasi":
            sampler = si.qmc.Sobol(d=steps)
            Z = si.norm.ppf(sampler.random(Npaths).T)
        else:
            Z = np.random.normal(size=size)

        geo = np.cumsum(((self.r - self.q - self.sigma**2/2)*dt + self.sigma*np.sqrt(dt) * Z),axis=0)
        st = np.log(self.S) +  geo
        return np.exp(st)
    
    def monte_carlo_call_option(self, Npaths, steps, use_antithetic=False, use_control_variate=False, random_generator=None):
        '''
        Calculates the price of a call option using the Monte Carlo model.
        Args:
        - Npaths (int): Number of paths to simulate.
        - steps (int): Number of time steps.
        - use_antithetic (bool): Whether to use antithetic variates.
        - use_control_variate (bool): Whether to use control variates.
        - random_generator (str): The random number generator to use. Options: "quasi" or None.
        Returns: call_option_price (float): The price of the call option.
        '''
        if use_antithetic:
            Z = np.random.normal(size=(steps, Npaths // 2))
            Z = np.concatenate((Z, -Z), axis=1)
            st = self.monte_carlo_model(Npaths, steps)
        else:
            st = self.monte_carlo_model(Npaths, steps, random_generator=random_generator)

        payoff = np.maximum(0, st[-1] - self.K)

        if use_control_variate:
            bs_price = self.call_option_price()
            mc_mean = np.mean(payoff)
            alpha = 0.2
            payoff = payoff + alpha * (bs_price - mc_mean)

        expectedpayoffT = 1/Npaths * np.sum(payoff)
        discountedpayoffT = np.exp(-self.r*self.T)
        return discountedpayoffT * expectedpayoffT
    
    def monte_carlo_put_option(self, Npaths, steps, use_antithetic=False, use_control_variate=False, random_generator=None):
        '''
        Calculates the price of a put option using the Monte Carlo model.
        Args:
        - Npaths (int): Number of paths to simulate.
        - steps (int): Number of time steps.
        - use_antithetic (bool): Whether to use antithetic variates.
        - use_control_variate (bool): Whether to use control variates.
        - random_generator (str): The random number generator to use. Options: "quasi" or None.
        Returns: put_option_price (float): The price of the put option.
        '''
        if use_antithetic:
            Z = np.random.normal(size=(steps, Npaths // 2))
            Z = np.concatenate((Z, -Z), axis=1)
            st = self.monte_carlo_model(Npaths, steps)
        else:
            st = self.monte_carlo_model(Npaths, steps, random_generator=random_generator)
        
        payoff = np.maximum(0, self.K - st[-1])

        if use_control_variate:
            bs_price = self.put_option_price()
            mc_mean = np.mean(payoff)
            alpha = 0.2
            payoff = payoff + alpha * (bs_price - mc_mean)

        expectedpayoffT = 1/Npaths * np.sum(payoff)
        discountedpayoffT = np.exp(-self.r*self.T)
        return discountedpayoffT * expectedpayoffT
    