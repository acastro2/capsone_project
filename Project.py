"""
Capstone Project: Evaluating the performance of GARCH models in times of high price fluctuation
Date Created | Author                 |
30-Apr-2020  | Alexandre Castro       |
30-Apr-2020  | Bruno Jordan Orfei Abe |
30-Apr-2020  | Francisco Costa        |

Code uses Python 3.0

==================================================================================================
The main goal of this research is to understand if GARCH models maintain superior performance in
forecasting volatility, despite the high price fluctuations in the financial markets due to extreme
shocks such as the one that happened with the COVID-19 phenom- ena. In this study, we will focus on
three different markets: the USA, Germany, and China. To achieve that, we use data composed of
daily quotations of the S&P 500, DAX 30, and SSE, and spanning over a period of 180 days until the
31st of March 2020. The GARCH models’ performance will be assessed using loss functions such as
Mean Squared Error (MSE) and Mean Absolute Error (MAE).
===================================================================================================
"""

# Import necesary modules
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import scipy as sp
import arch
from random import gauss
from random import seed
from matplotlib import pyplot
from arch import arch_model


def get_prices(ticker, start, end):
    """
    Get the Adj Close prices from the provided ticker using Yahoo Finance

    Parameters
    ==========
    ticker : string
        object to identify the stock
    start : datetime
        start date of the data collection
    end : datetime
        end date of the data collection
    """
    prices = pdr.get_data_yahoo(ticker, start=start, end=end)['Adj Close']

    prices = prices.dropna()

    return prices


def calculate_var(number_shares, stock_price, confidence_level, volatility):
    """
    Quantify the Maximum Expected Loss for the next day using a Value-at-Risk (VaR) model

    Parameters
    ==========
    number_shares : int
        Number of shares in portfolio
    stock_price : number
        Last stock price
    confidence_level : number
        Confidence level of the model
    volatility : number
        Last volatility
    number_days : int
        Number of days for the model
    """
    position = number_shares * stock_price

    z_value = sp.stats.norm.ppf(confidence_level)

    value_at_risk = position * z_value * volatility

    print("VaR Results:")
    print("=" * 25)
    print("%10s = %.0f" % ("Shares", number_shares))
    print("%10s = %.2f" % ("Price", stock_price))
    print("%10s = %.2f" % ("Holding", position))
    print("%10s = %.2f" % ("Conf Level", confidence_level))
    print("%10s = %.2f" % ("z value", z_value))
    print("%10s = %.4f" % ("Volatility", volatility))
    print("%10s = %.4f" % ("VaR", value_at_risk))
    print("=" * 25)

    return value_at_risk


def forecast_volatility(prices):
    """
    Forecast the volatility using a GARCH(1,1)

    Parameters
    ==========
    prices : series
        Asset price for forecasting volatility
    """
    price_dataframe = pd.DataFrame(prices)

    price_dataframe['log_price'] = np.log(price_dataframe['Adj Close'])
    price_dataframe['pct_change'] = price_dataframe['log_price'].diff()

    price_dataframe['stdev21'] = price_dataframe['pct_change'].rolling(
        window=21, center=False).std()

    price_dataframe['hvol21'] = price_dataframe['stdev21'] * (252**0.5)
    price_dataframe = price_dataframe.dropna()

    returns = price_dataframe['pct_change'] * 100

    model = arch.arch_model(returns)

    res = model.fit(disp='off')

    print(res.summary())

    price_dataframe['forecast_vol'] = 0.1 * np.sqrt(res.params['omega'] +
                                                    res.params['alpha[1]'] *
                                                    res.resid**2 +
                                                    res.conditional_volatility**2 *
                                                    res.params['beta[1]'])

    return price_dataframe.iloc[-1]['forecast_vol'].T

def run(ticker, init, end):
    print("=" * 25)
    print(ticker)
    print("=" * 25)
    prices = get_prices(ticker, init, end)
    last_price = prices.iloc[-1].T

    volatility = forecast_volatility(prices)

    calculate_var(100000, last_price, 0.95, volatility)

def main():
    """
    Program's main entrypoint
    """
    days_to_subtract = 180
    end = datetime(2020, 3, 31)
    init = end - timedelta(days=days_to_subtract)

    run("^GSPC", init, end)
    run("^GDAXI", init, end)
    run("SSE.L", init, end)


if __name__ == '__main__':
    main()
