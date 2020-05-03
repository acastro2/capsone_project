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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats import diagnostic


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


def get_daily_returns(prices):
    """
    Calculate and return Daily Returns based on provided prices

    Parameters
    ==========
    prices : array
        Prices of stock
    """

    data_ret = np.log(prices).diff().dropna()

    return data_ret


def main():
    """
    Program's main entrypoint
    """
    # Define start data and end date
    start = "2019-01-01"
    end = "31-03-31"

    # Tickers
    sp500_ticker = "^GSPC"
    dax30_ticker = "^GDAXI"
    sse_ticker = "SSE.L"

    # S&P500
    sp500 = get_prices(sp500_ticker, start, end)
    # DAX30
    dax30 = get_prices(dax30_ticker, start, end)
    # SSE
    sse = get_prices(sse_ticker, start, end)

    # Ploting the data
    fig, axs = pyplot.subplots(
        1, 3, sharex=True, figsize=(14, 5), constrained_layout=True)
    fig.suptitle('Prices', fontsize=16)

    # Setting the different subplots
    axs[0].plot(sp500, color='red')
    axs[0].set_title('S&P 500')
    axs[1].plot(dax30, color='green')
    axs[1].set_title('DAX 30')
    axs[2].plot(sse, color='blue')
    axs[2].set_title('SSE')

    # S&P500 daily returns
    sp500_ret = get_daily_returns(sp500)
    # DAX30 daily returns
    dax30_ret = get_daily_returns(dax30)
    # SSE daily returns
    sse_ret = get_daily_returns(sse)

    # Ploting the histogram of daily returns
    n_bins = 10

    fig, axs = pyplot.subplots(
        1, 3, sharey=True, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('Returns', fontsize=16)

    # Setting the different subplots
    axs[0].hist(sp500_ret, bins=n_bins, color='red')
    axs[0].set_title('S&P 500')
    axs[1].hist(dax30_ret, bins=n_bins, color='green')
    axs[1].set_title('DAX 30')
    axs[2].hist(sse_ret, bins=n_bins, color='blue')
    axs[2].set_title('SSE')

    # Ploting ACF and PACF of returns
    fig, axs = pyplot.subplots(
        2, 3, sharey=False, figsize=(15, 8), constrained_layout=True)
    fig.suptitle('Autocorrelation', fontsize=16)

    # ACF and PACF of S&P500 daily returns
    fig = plot_acf(sp500_ret, ax=axs[0, 0], color='red')
    fig = plot_pacf(sp500_ret, ax=axs[1, 0], color='red')
    axs[0, 0].set_title('S&P 500')

    # ACF and PACF of DAX30 daily returns
    fig = plot_acf(dax30_ret, ax=axs[0, 1], color='green')
    fig = plot_pacf(dax30_ret, ax=axs[1, 1], color='green')
    axs[0, 1].set_title('DAX 30')

    # ACF and PACF of SSE daily returns
    fig = plot_acf(sse_ret, ax=axs[0, 2], color='blue')
    fig = plot_pacf(sse_ret, ax=axs[1, 2], color='blue')
    axs[0, 2].set_title('SSE')

    # ARCH effect test on series of returns
    sp500ret_archtest = diagnostic.het_arch(sp500_ret)
    print('SP500 arch test: ', sp500ret_archtest)
    dax30ret_archtest = diagnostic.het_arch(dax30_ret)
    print('DAX30 arch test: ', dax30ret_archtest)
    sseret_archtest = diagnostic.het_arch(sse_ret)
    print('SSE arch test: ', sseret_archtest)

    # Square returns
    sp500_ret_squared = np.square(sp500_ret)
    dax30_ret_squared = np.square(dax30_ret)
    sse_ret_squared = np.square(sse_ret)

    # Ploting the data
    fig, axs = pyplot.subplots(
        1, 3, sharey=False, figsize=(15, 8), constrained_layout=True)
    fig.suptitle('Autocorrelation', fontsize=16)

    # Setting the different subplots
    fig = plot_acf(sp500_ret_squared, ax=axs[0], color='red')
    axs[0].set_title('S&P 500')
    fig = plot_acf(dax30_ret_squared, ax=axs[1], color='green')
    axs[1].set_title('DAX 30')
    fig = plot_acf(sse_ret_squared, ax=axs[2], color='blue')
    axs[2].set_title('SSE')

    pyplot.show()


if __name__ == '__main__':
    main()
