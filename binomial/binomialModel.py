import numpy as np
from functools import lru_cache

class BinomialModel:
    '''
    Class to calculate the price of an American call or put option using the Binomial Tree model.
    '''

    def __init__(self, S, K, T, r, sigma, q, N=100):
        '''
        Args:
        - S (float): The spot price of the underlying stock.
        - K (float): The strike price of the option.
        - T (float): The time to maturity of the option.
        - r (float): The risk-free rate.
        - sigma (float): The volatility of the underlying stock.
        - q (float): The dividend yield of the underlying stock.
        - N (int): The number of time steps in the binomial tree.
        '''
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = N
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)

    @property
    def params(self):
        return {'S': self.S,
                'K': self.K,
                'T': self.T,
                'r': self.r,
                'q': self.q,
                'sigma': self.sigma,
                'N': self.N}

    def option_price(self, call=True):
        '''
        Calculates the price of an American option using the Binomial Tree model.
        Args:
        - call (bool): True for call option, False for put option.
        
        Returns: option_price (float): The price of the American option.
        '''
        # Initialize the binomial tree
        prices = np.zeros((self.N + 1, self.N + 1))
        option_values = np.zeros((self.N + 1, self.N + 1))
        
        # Calculate the stock price at maturity
        for i in range(self.N + 1):
            for j in range(i + 1):
                prices[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)

        # Calculate option values at maturity
        if call:
            option_values[:, self.N] = np.maximum(0, prices[:, self.N] - self.K)
        else:
            option_values[:, self.N] = np.maximum(0, self.K - prices[:, self.N])

        # Backtrack the option values to the present value
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_value_no_exercise = np.exp(-self.r * self.dt) * (self.p * option_values[j, i + 1] + (1 - self.p) * option_values[j + 1, i + 1])
                if call:
                    option_values[j, i] = np.maximum(option_value_no_exercise, prices[j, i] - self.K)
                else:
                    option_values[j, i] = np.maximum(option_value_no_exercise, self.K - prices[j, i])

        return option_values[0, 0]
