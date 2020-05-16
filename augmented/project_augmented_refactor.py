"""
Capstone Project: Evaluating the performance of GARCH models in times of high price fluctuation
Date Created | Author                 |
30-Apr-2020  | Alexandre Castro       |
30-Apr-2020  | Bruno Jordan Orfei Abe |
30-Apr-2020  | Francisco Costa        |

Code uses Python 3.7.7

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

import os
import sys
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_prices(symbol, start, end):
    """
    Get the Adj Close prices from the provided ticker using Yahoo Finance

    Parameters
    ==========
    symbol : string
        object to identify the stock
    start : datetime
        start date of the data collection
    end : datetime
        end date of the data collection

    Returns
    =======
    prices
        a list of daily prices
    """

    return web.DataReader(symbol, 'yahoo', start=start, end=end)['Adj Close']


def plot_time_series(data, lags=None):
    """
    Return time series plot figure of the provided data.

    Parameters
    ==========
    data : series
        One-dimensional ndarray with axis labels (including time series).
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis.

    Returns
    =======
    figure : ~matplotlib.figure.Figure
        Figure with the plotted time series
    """

    if not isinstance(data, pd.Series):
        data = pd.Series(data).dropna()

    with plt.style.context('bmh'):
        fig = plt.figure(figsize=(10, 8))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        data.plot(ax=ts_ax)

        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(data, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(data, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(data, sparams=(data.mean(), data.std()), plot=pp_ax)
        plt.tight_layout()

    return fig


def plot_acf_pacf(data, lags=None):
    """
    Return ACF PACF plot figure of the provided data.

    Parameters
    ==========
    data : series
        One-dimensional ndarray with axis labels (including time series).
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis.

    Returns
    =======
    figure : ~matplotlib.figure.Figure
        Figure with the plotted time series
    """

    if not isinstance(data, pd.Series):
        data = pd.Series(data).dropna()

    with plt.style.context('bmh'):
        fig = plt.figure(figsize=(10, 3))
        layout = (1, 2)
        acf_ax = plt.subplot2grid(layout, (0, 0))
        pacf_ax = plt.subplot2grid(layout, (0, 1))
        smt.graphics.plot_acf(data, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(data, lags=lags, ax=pacf_ax, alpha=0.5)
        plt.tight_layout()

    return fig


def find_best_arima_model(time_series):
    """
    Return best ARIMA model for provided Time Series.

    Parameters
    ==========
    time_series : series
        One-dimensional ndarray with axis labels (including time series).

    Returns
    =======
    best_aic : float
    best_order : float
    """

    best_aic = np.inf
    best_order = None

    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(4)  # [0,1,2,3]

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(time_series, order=(i, d, j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic

                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                except:
                    continue

    return best_aic, best_order


def fit_garch(time_series, best_order):
    """
    Return fit GARCH Model

    Parameters
    ==========
    time_series : series
        One-dimensional ndarray with axis labels (including time series).
    best_order : tuple
        One-dimensional ndarray with axis labels (including time series).

    Returns
    =======
    best_aic : float
    best_order : float
    """

    p_ = best_order[0]
    o_ = best_order[1]
    q_ = best_order[2]

    am = arch_model(time_series, p=p_, o=o_, q=q_, dist='StudentsT')

    return am.fit(update_freq=5, disp='off')


def main():
    """
    Main script.

    Steps
    =====

    1. Define Parameters: start, end and end of slice dates, symbols, lags
    2. Get adjusted close prices for symbols
    3. Time series analysis
    4. Find best ARIMA models for fitting GARCH models
    5. Fitting GARCH models

    """

    # Define start data and end date
    start = '2015-01-01'
    end = '2020-05-01'
    end_slice = '2020-01-01'

    # Define symbols
    SPX, DAX, SSE = '^GSPC', '^GDAXI', 'SSE.L'
    symbols = [SPX, DAX, SSE]
    lags_ = 30

    # raw adjusted close prices
    adj_close_prices = pd.DataFrame({sym: get_prices(sym, start, end) for sym in symbols})
    print(adj_close_prices)

    # Time Series Analysis
    scaled_log_returns = np.log(adj_close_prices/adj_close_prices.shift(1)).dropna() * 100

    plot_time_series(np.diff(adj_close_prices[SPX]), lags=lags_).savefig('images/spx_series_analysis')
    plot_time_series(np.diff(adj_close_prices[DAX]), lags=lags_).savefig('images/dax_series_analysis')
    plot_time_series(np.diff(adj_close_prices[SSE]), lags=lags_).savefig('images/sse_series_analysis')

    spx_estimation_time_series = scaled_log_returns[SPX].loc[start:end_slice]
    dax_estimation_time_series = scaled_log_returns[DAX].loc[start:end_slice]
    sse_estimation_time_series = scaled_log_returns[SSE].loc[start:end_slice]

    # Find Best ARIMA Model for fitting GARCH
    res_tup_spx = find_best_arima_model(spx_estimation_time_series)
    res_tup_dax = find_best_arima_model(dax_estimation_time_series)
    res_tup_sse = find_best_arima_model(sse_estimation_time_series)

    print('SPX -> aic: {:6.5f} | order: {}'.format(res_tup_spx[0], res_tup_spx[1]))
    print('DAX -> aic: {:6.5f} | order: {}'.format(res_tup_dax[0], res_tup_dax[1]))
    print('SSE -> aic: {:6.5f} | order: {}'.format(res_tup_sse[0], res_tup_sse[1]))

    # GARCH Model fitting
    spx_garch = fit_garch(scaled_log_returns[SPX].loc[start:end_slice], res_tup_spx[1])
    plot_time_series(spx_garch.resid, lags=lags_).savefig('images/spx_garch_residuals_analysis')
    plot_acf_pacf(spx_garch.resid ** 2, lags=lags_).savefig('images/spx_garch_squared_residuals_analysis')
    print(spx_garch.summary())

    dax_garch = fit_garch(scaled_log_returns[DAX].loc[start:end_slice], res_tup_dax[1])
    plot_time_series(dax_garch.resid, lags=lags_).savefig('images/dax_garch_residuals_analysis')
    plot_acf_pacf(dax_garch.resid ** 2, lags=lags_).savefig('images/dax_garch_squared_residuals_analysis')
    print(dax_garch.summary())

    sse_garch = fit_garch(scaled_log_returns[SSE].loc[start:end_slice], res_tup_sse[1])
    plot_time_series(sse_garch.resid, lags=lags_).savefig('images/sse_garch_residuals_analysis')
    plot_acf_pacf(sse_garch.resid ** 2, lags=lags_).savefig('images/sse_garch_squared_residuals_analysis')
    print(sse_garch.summary())


if __name__ == '__main__':
    """
    Main entry point of the Python Script.
    """
    main()
