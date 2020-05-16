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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats import diagnostic
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import sys
import seaborn as sns
from arch import arch_model
from matplotlib import pyplot
from random import seed
from random import gauss
import arch
import scipy as sp
import numpy as np
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')


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


def compute_returns(data):
    """returns Daily Returns based on provided prices
    Parameters
    ==========
    prices : array
        Prices of stock
    """
    data_ret = np.log(data).diff().dropna()
    return data_ret


def main():
    """
    Program's main entrypoint
    """
    # Getting data

    # Define start data and end date
    start = "2019-01-01"
    end = "2020-03-31"

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
        1, 3, sharex=True, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('Index Adj Close Price', fontsize=15)
    fig.autofmt_xdate()

    axs[0].plot(sp500, color='red')
    axs[0].set_title('S&P500')
    axs[1].plot(dax30, color='green')
    axs[1].set_title('DAX30')
    axs[2].plot(sse, color='blue')
    axs[2].set_title('SSE')

    fig.savefig('Images/prices')
    pyplot.close()

    # Computing daily returns

    # S&P500 daily returns
    sp500_ret = compute_returns(sp500)
    # DAX30 daily returns
    dax30_ret = compute_returns(dax30)
    # SSE daily returns
    sse_ret = compute_returns(sse)

    print(sse_ret.shape)
    print(sse_ret)

    # Ploting the histogram of daily returns
    n_bins = 10

    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle('Histogram of returns', fontsize=15)
    # Setting the different subplots
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(sp500_ret, bins=n_bins, color='red')
    axs[0].set_title('S&P500')
    axs[1].hist(dax30_ret, bins=n_bins, color='green')
    axs[1].set_title('DAX30')
    axs[2].hist(sse_ret, bins=n_bins, color='blue')
    axs[2].set_title('SSE')

    fig.savefig('Images/histogram')
    pyplot.close()

    # Ploting ACF and PACF of returns
    fig, axs = pyplot.subplots(
        2, 3, sharey=False, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        'Autocorrelation and Partial Autocorrelation of returns', fontsize=15)

    # ACF and PACF of S&P500 daily returns
    fig = plot_acf(sp500_ret, ax=axs[0, 0], color='red')
    fig = plot_pacf(sp500_ret, ax=axs[1, 0], color='red')
    axs[0, 0].set_title('S&P500 Autocorrelation')
    axs[1, 0].set_title('S&P500 Partial Autocorrelation')

    # ACF and PACF of DAX30 daily returns
    fig = plot_acf(dax30_ret, ax=axs[0, 1], color='green')
    fig = plot_pacf(dax30_ret, ax=axs[1, 1], color='green')
    axs[0, 1].set_title('DAX30 Autocorrelation')
    axs[1, 1].set_title('DAX30 Partial Autocorrelation')

    # ACF and PACF of SSE daily returns
    fig = plot_acf(sse_ret, ax=axs[0, 2], color='blue')
    fig = plot_pacf(sse_ret, ax=axs[1, 2], color='blue')
    axs[0, 2].set_title('SSE Autocorrelation')
    axs[1, 2].set_title('SSE Partial Autocorrelation')

    fig.savefig('Images/autocorrelation_partialautocorrelation')
    pyplot.close()

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
    fig.suptitle('Autocorrelation of squared returns', fontsize=15)

    # Setting the different subplots
    fig = plot_acf(sp500_ret_squared, ax=axs[0], color='red')
    axs[0].set_title('S&P500 autocorrelation')
    fig = plot_acf(dax30_ret_squared, ax=axs[1], color='green')
    axs[1].set_title('DAX30 autocorrelation')
    fig = plot_acf(sse_ret_squared, ax=axs[2], color='blue')
    axs[2].set_title('SSE autocorrelation')

    fig.savefig('Images/autocorrelation_squared_returns')
    pyplot.close()

    # GARCH model rolling window estimation S&P500

    # GARCH model specification
    gm_sp500 = arch_model(sp500_ret, vol='Garch', p=1, q=1, dist='Normal')

    # Rolling window estimation
    sp500_index = sp500_ret.index
    start_loc = 0
    end_loc = np.where(sp500_index >= '2020-03-03')[0].min()
    sp500_forecasts = {}
    for i in range(22):
        sys.stdout.write('.')
        sys.stdout.flush()
        sp500_res = gm_sp500.fit(first_obs=i, last_obs=i + end_loc, disp='off')
        sp500_temp = sp500_res.forecast(horizon=10).variance
        sp500_fcast = sp500_temp.iloc[i + end_loc - 1]
        sp500_forecasts[sp500_fcast.name] = sp500_fcast
    print()
    print(pd.DataFrame(sp500_forecasts).T)

    # Rolling window estimation of realized variance
    w = 22

    sp500_roller = sp500_ret.rolling(w)
    sp500_var = sp500_roller.var(ddof=0)
    sp500_var = pd.DataFrame(sp500_var)
    print(sp500_var.tail(22))

    # Merge forecasted variance with realized variance into one DataFrame
    sp500_forecast = pd.DataFrame(sp500_forecasts).T
    sp500_forecast_real = pd.concat(
        [sp500_forecast, sp500_var], axis=1).dropna()
    print(sp500_forecast_real)

    # Mean squared error S&P500
    sp500_mse = mean_squared_error(
        sp500_forecast_real['h.10'], sp500_forecast_real['Adj Close'])
    print('Mean Squared Error S&P500: ', sp500_mse)

    # Mean absolute error S&P500
    sp500_mae = mean_absolute_error(
        sp500_forecast_real['h.10'], sp500_forecast_real['Adj Close'])
    print('Mean Absolute Error S&P500: ', sp500_mae)

    # Plot forecasted variance against realized variance
    pyplot.close()
    pyplot.title('S&P500 forecast variance vs S&P500 realized variance')
    pyplot.plot(sp500_forecast_real['h.10'], label='forecast')
    pyplot.plot(sp500_forecast_real['Adj Close'], label='realized')
    pyplot.legend()
    pyplot.xticks(rotation=45)
    pyplot.tight_layout()
    pyplot.savefig('Images/sp500_forecast_realized')
    pyplot.close()

    # GARCH model rolling window estimation DAX30

    # GARCH model specification
    gm_dax30 = arch_model(dax30_ret, vol='Garch', p=1, q=1, dist='Normal')

    # Rolling window estimation
    dax30_index = dax30_ret.index
    start_loc = 0
    end_loc = np.where(dax30_index >= '2020-03-03')[0].min()
    dax30_forecasts = {}
    for i in range(22):
        sys.stdout.write('.')
        sys.stdout.flush()
        dax30_res = gm_dax30.fit(first_obs=i, last_obs=i + end_loc, disp='off')
        dax30_temp = dax30_res.forecast(horizon=10).variance
        dax30_fcast = dax30_temp.iloc[i + end_loc - 1]
        dax30_forecasts[dax30_fcast.name] = dax30_fcast
    print()
    print(pd.DataFrame(dax30_forecasts).T)

    # Rolling window estimation of realized variance
    w = 22

    dax30_roller = dax30_ret.rolling(w)
    dax30_var = dax30_roller.var(ddof=0)
    dax30_var = pd.DataFrame(dax30_var)
    print(dax30_var.tail(22))

    # Merge forecasted variance with realized variance into one DataFrame
    dax30_forecast = pd.DataFrame(dax30_forecasts).T
    dax30_forecast_real = pd.concat(
        [dax30_forecast, dax30_var], axis=1).dropna()
    print(dax30_forecast_real)

    # Mean squared error DAX30
    dax30_mse = mean_squared_error(
        dax30_forecast_real['h.10'], dax30_forecast_real['Adj Close'])
    print('Mean Squared Error DAX30: ', dax30_mse)

    # Mean absolute error DAX30
    dax30_mae = mean_absolute_error(
        dax30_forecast_real['h.10'], dax30_forecast_real['Adj Close'])
    print('Mean Absolute Error DAX30: ', dax30_mae)

    # Plot forecasted variance against realized variance
    pyplot.title('DAX30 forecast variance vs DAX30 realized variance')
    pyplot.plot(dax30_forecast_real['h.10'], label='forecast')
    pyplot.plot(dax30_forecast_real['Adj Close'], label='realized')
    pyplot.legend()
    pyplot.xticks(rotation=45)
    pyplot.tight_layout()
    pyplot.savefig('Images/dax30_forecast_realized')
    pyplot.close()

    # GARCH model rolling window estimation SSE

    # GARCH model specification
    gm_sse = arch_model(sse_ret, vol='Garch', p=1, q=1, dist='Normal')

    # Rolling window estimation
    sse_index = sse_ret.index
    start_loc = 0
    end_loc = np.where(sse_index >= '2020-03-03')[0].min()
    sse_forecasts = {}
    for i in range(22):
        sys.stdout.write('.')
        sys.stdout.flush()
        sse_res = gm_sse.fit(first_obs=i, last_obs=i + end_loc, disp='off')
        sse_temp = sse_res.forecast(horizon=10).variance
        sse_fcast = sse_temp.iloc[i + end_loc - 1]
        sse_forecasts[sse_fcast.name] = sse_fcast
    print()
    print(pd.DataFrame(sse_forecasts).T)

    # Rolling window estimation of realized variance
    w = 22

    sse_roller = sse_ret.rolling(w)
    sse_var = sse_roller.var(ddof=0)
    sse_var = pd.DataFrame(sse_var)
    print(sse_var.tail(22))

    # Merge forecasted variance with realized variance into one DataFrame
    sse_forecast = pd.DataFrame(sse_forecasts).T
    sse_forecast_real = pd.concat([sse_forecast, sse_var], axis=1).dropna()
    print(sse_forecast_real)

    # Mean squared error SSE
    sse_mse = mean_squared_error(
        sse_forecast_real['h.10'], sse_forecast_real['Adj Close'])
    print('Mean Squared Error SSE: ', sse_mse)

    # Mean absolute error SSE
    sse_mae = mean_absolute_error(
        sse_forecast_real['h.10'], sse_forecast_real['Adj Close'])
    print('Mean Absolute Error SSE: ', sse_mae)

    # Plot forecasted variance against realized variance
    pyplot.title('SSE forecast variance vs SSE realized variance')
    pyplot.plot(sse_forecast_real['h.10'], label='forecast')
    pyplot.plot(sse_forecast_real['Adj Close'], label='realized')
    pyplot.legend()
    pyplot.xticks(rotation=45)
    pyplot.tight_layout()
    pyplot.savefig('Images/sse_forecast_realized')
    pyplot.close()


if __name__ == '__main__':
    main()
