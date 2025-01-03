import numpy as np
import yfinance as yf

def fetch_stock_data(ticker_symbol: str):
    '''
    Fetches stock data for a given ticker symbol.
    Args:   ticker_symbol (str): The ticker symbol of the stock.
    Returns: stock_data (pd.DataFrame): The stock data.
    '''
    stock_data = yf.download(ticker_symbol, period='1y', interval='1d')
    return stock_data

def fetch_options_data(ticker_symbol: str, expiry_date: int = 5) -> tuple:
    '''
    Fetches options data for a given ticker symbol.
    Args:   ticker_symbol (str): The ticker symbol of the stock.
    Returns: options_data (tuple): A tuple containing the expiry date, call options data, and put options data.
    '''
    ticker = yf.Ticker(ticker_symbol)
    options_dates = ticker.options
    options_data = ticker.option_chain(options_dates[expiry_date])  # Closest expiry date [Expand to include more]
    return options_dates[expiry_date], options_data.calls, options_data.puts

def fetch_risk_free_rate():
    '''
    Fetches the risk-free rate.
    Returns: risk_free_rate (float): The risk-free rate based on the 13-week Treasury Bill.
    '''
    rfr = yf.download("^IRX", period='1d')
    return rfr['Close'].iloc[-1]

def fetch_dividend_yield(ticker_symbol: str) -> float:
    '''
    Fetches the dividend yield for a given stock.
    Args:   ticker_symbol (str): The ticker symbol of the stock.
    Returns: dividend_yield (float): The dividend yield.
    '''
    dividend_yield = float(yf.Ticker(ticker_symbol).info['dividendRate']/ yf.Ticker(ticker_symbol).info['previousClose'])
    return dividend_yield

def calculate_historical_volatility(stock_data, window=252) -> float:
    '''
    Calculates the historical volatility of a stock.
    Args:
    - stock_data (pd.DataFrame): The stock data.
    - window (int): The window size for calculating the historical volatility.
    Returns: volatility (float): The historical volatility of the stock.
    '''
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
    volatility = np.sqrt(window) * log_returns.std()
    return volatility