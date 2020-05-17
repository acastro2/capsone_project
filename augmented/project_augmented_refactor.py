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
import arch
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import statsmodels.stats as sms
import scipy.stats as scs
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_prices(symbol, start, end):
    """Get the Adj Close prices from the provided ticker using Yahoo Finance

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


def plot_time_series(data, lags=None, title=None):
    """Return time series plot figure of the provided data.

    Parameters
    ==========
    data : series
        One-dimensional ndarray with axis labels (including time series).
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis.
    title : string, optional
        The title that will be set for the whole figure.

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
        ts_ax.set_title(title if title else 'Time Series Analysis Plots')
        smt.graphics.plot_acf(data, lags=lags, ax=acf_ax, alpha=0.5, zero=False)
        smt.graphics.plot_pacf(data, lags=lags, ax=pacf_ax, alpha=0.5, zero=False)
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(data, sparams=(data.mean(), data.std()), plot=pp_ax)
        plt.sca(acf_ax)
        plt.xticks(np.arange(1,  lags + 1, 2.0))
        plt.sca(pacf_ax)
        plt.xticks(np.arange(1,  lags + 1, 2.0))
        plt.tight_layout()

    return fig


def plot_acf_pacf(data, lags=None):
    """Return ACF PACF plot figure of the provided data.

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
        smt.graphics.plot_acf(data, lags=lags, ax=acf_ax, alpha=0.5, zero=False)
        smt.graphics.plot_pacf(data, lags=lags, ax=pacf_ax, alpha=0.5, zerop=False)
        plt.sca(acf_ax)
        plt.xticks(np.arange(1,  lags + 1, 2.0))
        plt.sca(pacf_ax)
        plt.xticks(np.arange(1,  lags + 1, 2.0))
        plt.tight_layout()

    return fig


def plot_histogram(data, n_bins=None, title=None):
    """Returns histogram plot figure of the provided data.

    Parameters
    ==========
    data : series
        One-dimensional ndarray with axis labels (including time series).
    n_bins : {int, array_like}, optional
        An int or array of number of histogram bins, used on horizontal axis.

    Returns
    =======
    figure : ~matplotlib.figure.Figure
        Figure with the plotted time series
    """

    if not isinstance(data, pd.Series):
        data = pd.Series(data).dropna()

    with plt.style.context('bmh'):
        fig = plt.figure(figsize=(10, 3))
        layout = (1, 1)
        hist_ax = plt.subplot2grid(layout, (0, 0))
        data.hist(bins=n_bins, ax=hist_ax)
        hist_ax.set_title(title if title else 'Histogram Analysis')
        plt.tight_layout()

    return fig

def plot_ljung_box_test(data, lags=[10], title=None):
    """Return best ARIMA model for provided Time Series.

    Parameters
    ==========
    time_series : series
        One-dimensional ndarray with axis labels (including time series).

    Returns
    =======
    best_aic : float
    best_order : float
    """

    if not isinstance(data, pd.Series):
        data = pd.Series(data).dropna()

    with plt.style.context('bmh'):
        tmp_acor = list(sms.diagnostic.acorr_ljungbox(data, lags=lags, boxpierce=True))
        p_vals = pd.Series(tmp_acor[1])
        p_vals.index += 1
        fig = plt.figure(figsize=(10, 3))
        layout = (1, 1)
        ljung_ax = plt.subplot2grid(layout, (0, 0))
        p_vals.plot(ax=ljung_ax, linestyle='', marker='o', legend=False)
        ljung_ax.set_title(title if title else 'p-values for Ljung-Box Test')
        plt.axhline(y=0.05, color='blue', linestyle='--')
        x = np.arange(p_vals.size) + 1
        for X, Y, Z in zip(x, p_vals, p_vals):
            plt.annotate(round(Z, 4), xy=(X,Y), xytext=(-5, 5), ha = 'left', textcoords='offset points')
    return fig


def find_best_arima_model(time_series):
    """Return best ARIMA model for provided Time Series.

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
    d_rng = range(2)  # [0,1]

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(time_series, order=(i, d, j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic

                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except:
                    continue

    return best_aic, best_order, best_mdl

def fit_arch(time_series, q=1):
    """Return fit ARCH Model

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

    am = arch.univariate.ConstantMean(time_series)
    am.volatility = arch.univariate.ARCH(q)
    am.distribution=arch.univariate.StudentsT()

    return am.fit(update_freq=5, disp='off')

def fit_garch(time_series, p=1, o=0, q=1):
    """Return fit GARCH Model

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

    am = arch.univariate.ConstantMean(time_series)
    am.volatility = arch.univariate.GARCH(p, o, q)
    am.distribution=arch.univariate.StudentsT()

    return am.fit(update_freq=5, disp='off')


def main():
    """Main script.

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
    hist_n_bins_ = 50

    # raw adjusted close prices
    adj_close_prices = pd.DataFrame({sym: get_prices(sym, start, end) for sym in symbols})
    print(adj_close_prices)

    adj_close_prices = adj_close_prices.dropna()
    # Time Series Analysis
    log_returns_percent = np.log(adj_close_prices/adj_close_prices.shift(1)).dropna() * 100

    # Lets inspect the time series to check for its properties.
    plot_time_series(adj_close_prices[SPX], lags=lags_, title='Time series property for SPX adjusted close prices').savefig('images/0_spx_series_analysis')
    plot_time_series(adj_close_prices[DAX], lags=lags_, title='Time series property for DAX adjusted close prices').savefig('images/0_dax_series_analysis')
    plot_time_series(adj_close_prices[SSE], lags=lags_, title='Time series property for SSE adjusted close prices').savefig('images/0_sse_series_analysis')

    # Unit root tests. The null hypothesis of the ADF test is that the data does not have a unit
    # root process. p-value <= alpha indicates that the data does not have a unit root, hence it is
    # stationary
    adf_spx = smt.stattools.adfuller(adj_close_prices[SPX])
    adf_dax = smt.stattools.adfuller(adj_close_prices[DAX])
    adf_sse = smt.stattools.adfuller(adj_close_prices[SSE])
    print(f'Augmented Dick-Fuller test for SPX time series: {adf_spx}')
    print(f'Augmented Dick-Fuller test for DAX time series: {adf_dax}')
    print(f'Augmented Dick-Fuller test for SSE time series: {adf_sse}')
    # Augmented Dick-Fuller test for SPX time series: (-1.317384935996279, 0.6211816307778435, 22, 1279, {'1%': -3.4354731218715946, '5%': -2.863802419761889, '10%': -2.5679745318585363}, 12131.83159293637)
    # Augmented Dick-Fuller test for DAX time series: (-2.874989782113799, 0.048334603056566845, 3, 1298, {'1%': -3.4353979810093374, '5%': -2.863769264797245, '10%': -2.5679568752685773}, 16285.168784562955)
    # Augmented Dick-Fuller test for SSE time series: (-2.82272190964574, 0.05510069006407777, 21, 1280, {'1%': -3.435469111362934, '5%': -2.8638006501960755, '10%': -2.567973589477539}, 11030.671073964486)

    # Obviously all the series are non-stationary analysing the test. We now check the presence of a
    # unit root in the first difference of the process.

    plot_time_series(np.diff(adj_close_prices[SPX]), lags=lags_, title='Time series property for first difference of SPX adjusted close prices').savefig('images/1_spx_lag_series_analysis')
    plot_time_series(np.diff(adj_close_prices[DAX]), lags=lags_, title='Time series property for first difference of DAX adjusted close prices').savefig('images/1_dax_lag_series_analysis')
    plot_time_series(np.diff(adj_close_prices[SSE]), lags=lags_, title='Time series property for first difference of SSE adjusted close prices').savefig('images/1_sse_lag_series_analysis')
    adf_spx_diff = smt.stattools.adfuller(np.diff(adj_close_prices[SPX]))
    adf_dax_diff = smt.stattools.adfuller(np.diff(adj_close_prices[DAX]))
    adf_sse_diff = smt.stattools.adfuller(np.diff(adj_close_prices[SSE]))
    print(f'Augmented Dick-Fuller test for first difference SPX time series: {adf_spx_diff}')
    print(f'Augmented Dick-Fuller test for first difference DAX time series: {adf_dax_diff}')
    print(f'Augmented Dick-Fuller test for first difference SSE time series: {adf_sse_diff}')
    # Augmented Dick-Fuller test for first difference SPX time series: (-8.134797566196124, 1.0663385364288605e-12, 21, 1279, {'1%': -3.4354731218715946, '5%': -2.863802419761889, '10%': -2.5679745318585363}, 12123.062894751427)
    # Augmented Dick-Fuller test for first difference DAX time series: (-19.01332615414863, 0.0, 2, 1298, {'1%': -3.4353979810093374, '5%': -2.863769264797245, '10%': -2.5679568752685773}, 16277.747589583605)
    # Augmented Dick-Fuller test for first difference SSE time series: (-8.12994457326282, 1.097114342327654e-12, 20, 1280, {'1%': -3.435469111362934, '5%': -2.8638006501960755, '10%': -2.567973589477539}, 11029.132013852293)
    
    # The first difference of the series are stationary, hence the returns should be stationary.

    adf_spx_log_ret = smt.stattools.adfuller(log_returns_percent[SPX])
    adf_dax_log_ret = smt.stattools.adfuller(log_returns_percent[DAX])
    adf_sse_log_ret = smt.stattools.adfuller(log_returns_percent[SSE])
    print(f'Augmented Dick-Fuller test for log-returns of SPX time series: {adf_spx_log_ret}')
    print(f'Augmented Dick-Fuller test for log-returns of DAX time series: {adf_dax_log_ret}')
    print(f'Augmented Dick-Fuller test for log-returns of SSE time series: {adf_sse_log_ret}')
    # Augmented Dick-Fuller test for log-returns of SPX time series: (-9.641743937201769, 1.5144357219710266e-16, 12, 1288, {'1%': -3.435437251933509, '5%': -2.863786592704128, '10%': -2.567966103183712}, -7883.020592282379)
    # Augmented Dick-Fuller test for log-returns of DAX time series: (-12.648551474194013, 1.384460262132333e-23, 7, 1293, {'1%': -3.4354175403897727, '5%': -2.8637778952086848, '10%': -2.5679614713589562}, -7448.209339704386)
    # Augmented Dick-Fuller test for log-returns of SSE time series: (-8.178999405470167, 8.228050786464477e-13, 20, 1280, {'1%': -3.435469111362934, '5%': -2.8638006501960755, '10%': -2.567973589477539}, -7037.030298013629)

    # The log-returns of the series are stationary. We will use the log-returns from here on as the
    # data we use to run our models.

    # Histogram of log-returns.
    plot_histogram(log_returns_percent[SPX], n_bins=hist_n_bins_, title='Histogram of log-returns for SPX').savefig('images/2_spx_log_ret_hist')
    plot_histogram(log_returns_percent[DAX], n_bins=hist_n_bins_, title='Histogram of log-returns for DAX').savefig('images/2_dax_log_ret_hist')
    plot_histogram(log_returns_percent[SSE], n_bins=hist_n_bins_, title='Histogram of log-returns for SSE').savefig('images/2_sse_log_ret_hist')

    plot_time_series(log_returns_percent[SPX], lags=lags_, title='Time series property for log-returns of SPX').savefig('images/3_spx_log_returns_analysis')
    plot_time_series(log_returns_percent[DAX], lags=lags_, title='Time series property for log-returns of DAX').savefig('images/3_dax_log_returns_analysis')
    plot_time_series(log_returns_percent[SSE], lags=lags_, title='Time series property for log-returns of SSE').savefig('images/3_sse_log_returns_analysis')

    # Because we have a stationary process and the time series plots show autocorrelation, we will
    # try to fit the best ARIMA model for each one of the series. Before we can fit a GARCH model,
    # we need to understand if there are certain characteristics as an outcome to other models being
    # estimated.

    spx_estimation_time_series = log_returns_percent[SPX].loc[start:end_slice]
    dax_estimation_time_series = log_returns_percent[DAX].loc[start:end_slice]
    sse_estimation_time_series = log_returns_percent[SSE].loc[start:end_slice]

    # Find Best ARIMA Model for fitting GARCH
    spx_arima_best_aic, spx_arima_best_order, spx_arima_best_mdl = find_best_arima_model(spx_estimation_time_series)
    dax_arima_best_aic, dax_arima_best_order, dax_arima_best_mdl = find_best_arima_model(dax_estimation_time_series)
    sse_arima_best_aic, sse_arima_best_order, sse_arima_best_mdl = find_best_arima_model(sse_estimation_time_series)
    print('SPX -> aic: {:6.5f} | order: {}'.format(spx_arima_best_aic, spx_arima_best_order))
    print('DAX -> aic: {:6.5f} | order: {}'.format(dax_arima_best_aic, dax_arima_best_order))
    print('SSE -> aic: {:6.5f} | order: {}'.format(sse_arima_best_aic, sse_arima_best_order))

    # Now lets see how our ARIMA models are performing.
    print(f'-----------------------------------------------------------------------------\n'
          f'SPX ARIMA {spx_arima_best_order}\n'
          f'-----------------------------------------------------------------------------\n'
          f'{spx_arima_best_mdl.summary()}\n')
    print(f'-----------------------------------------------------------------------------\n'
          f'DAX ARIMA {dax_arima_best_order}\n'
          f'-----------------------------------------------------------------------------\n'
          f'{dax_arima_best_mdl.summary()}\n')
    print(f'-----------------------------------------------------------------------------\n'
          f'SSE ARIMA {sse_arima_best_order}\n'
          f'-----------------------------------------------------------------------------\n'
          f'{sse_arima_best_mdl.summary()}\n')


    # Lets check for the presence of arch effects in the data.

    spx_arch_test = sms.diagnostic.het_arch(spx_arima_best_mdl.resid, ddof=spx_arima_best_order[0] + spx_arima_best_order[2])
    dax_arch_test = sms.diagnostic.het_arch(dax_arima_best_mdl.resid, ddof=dax_arima_best_order[0] + dax_arima_best_order[2])
    sse_arch_test = sms.diagnostic.het_arch(sse_arima_best_mdl.resid, ddof=sse_arima_best_order[0] + sse_arima_best_order[2])
    print(f"Engle's test for ARCH Effects for log-returns of SPX: {spx_arch_test}")
    print(f"Engle's test for ARCH Effects for log-returns of DAX: {dax_arch_test}")
    print(f"Engle's test for ARCH Effects for log-returns of SSE: {sse_arch_test}")
    # Engle's test for ARCH Effects for log-returns of SPX: (599.8920220820492, 4.830038167014557e-112, 48.232962005269286, 2.1201198464795764e-154)
    # Engle's test for ARCH Effects for log-returns of DAX: (332.5420788311802, 1.1497359787661418e-56, 19.176710106264657, 1.419189429062453e-66)
    # Engle's test for ARCH Effects for log-returns of SSE: (502.9838782231942, 8.450047255379039e-92, 35.384497205591, 7.932244194926643e-119)

    plot_time_series(spx_arima_best_mdl.resid, lags=lags_, title='Time series property for arima residuals of SPX').savefig('images/4_spx_arima_resid_analysis')
    plot_time_series(dax_arima_best_mdl.resid, lags=lags_, title='Time series property for arima residuals of DAX').savefig('images/4_dax_arima_resid_analysis')
    plot_time_series(sse_arima_best_mdl.resid, lags=lags_, title='Time series property for arima residuals of SSE').savefig('images/4_sse_arima_resid_analysis')

    # Ljung-Box test of the residuals, if the values are higher than the significance level, there
    # are no evidences to reject the null hypothesis that the data is independently distributed, or
    # not exhibiting serial correlation.

    plot_ljung_box_test(spx_arima_best_mdl.resid, lags=lags_, title='Ljung-Box test for arima residuals of SPX').savefig('images/5_spx_arima_resid_ljung_analysis')
    plot_ljung_box_test(dax_arima_best_mdl.resid, lags=lags_, title='Ljung-Box test for arima residuals of DAX').savefig('images/5_dax_arima_resid_ljung_analysis')
    plot_ljung_box_test(sse_arima_best_mdl.resid, lags=lags_, title='Ljung-Box test for arima residuals of SSE').savefig('images/5_sse_arima_resid_ljung_analysis')

    # As of here, the ARIMA looks like a good model, because its residues seems to be white noise.
    # We now need to understand if there is a relationship in the square of the residuals, which
    # would imply existence of ARCH effects.

    plot_time_series(spx_arima_best_mdl.resid ** 2, lags=lags_, title='Time series property for squared arima residuals of SPX').savefig('images/6_spx_arima_resid_square_analysis')
    plot_time_series(dax_arima_best_mdl.resid ** 2, lags=lags_, title='Time series property for sqaured arima residuals of DAX').savefig('images/6_dax_arima_resid_square_analysis')
    plot_time_series(sse_arima_best_mdl.resid ** 2, lags=lags_, title='Time series property for squared arima residuals of SSE').savefig('images/6_sse_arima_resid_square_analysis')

    plot_ljung_box_test(spx_arima_best_mdl.resid ** 2, lags=lags_, title='Ljung-Box test for arima residuals of SPX').savefig('images/7_spx_arima_square_resid_ljung_analysis')
    plot_ljung_box_test(dax_arima_best_mdl.resid ** 2, lags=lags_, title='Ljung-Box test for arima residuals of DAX').savefig('images/7_dax_arima_square_resid_ljung_analysis')
    plot_ljung_box_test(sse_arima_best_mdl.resid ** 2, lags=lags_, title='Ljung-Box test for arima residuals of SSE').savefig('images/7_sse_arima_square_resid_ljung_analysis')

    # The Ljung-Box test on the square of the residuals of the ARIMA model show exactly the expected
    # behaviour of having correlation. We need to fit a model with ARCH characteristics.

    # ARCH Model fitting -> We are fitting one ARCH, to check whether we need to do a GARCH, then
    # checking the standardized squared residuals of the model we fitted, we see the existence of
    # autocorrelation, which leads to the need of a GARCH model.
    spx_arch_1 = fit_arch(spx_estimation_time_series)
    print(spx_arch_1.summary())
    spx_arch_1_std_resid = spx_arch_1.resid / spx_arch_1.conditional_volatility
    plot_time_series(spx_arch_1_std_resid, lags=lags_).savefig('images/8_spx_arch_1_std_residuals_analysis')
    plot_time_series(spx_arch_1_std_resid ** 2, lags=lags_).savefig('images/9_spx_arch_1_std_resid_square_analysis')
    plot_ljung_box_test(spx_arch_1_std_resid ** 2, lags=lags_).savefig('images/10_spx_arch_1_std_resid_square_ljung_analysis')

    dax_arch_1 = fit_arch(dax_estimation_time_series)
    print(dax_arch_1.summary())
    dax_arch_1_std_resid = dax_arch_1.resid / dax_arch_1.conditional_volatility
    plot_time_series(dax_arch_1_std_resid, lags=lags_).savefig('images/8_dax_arch_1_std_residuals_analysis')
    plot_time_series(dax_arch_1_std_resid ** 2, lags=lags_).savefig('images/9_dax_arch_1_std_resid_square_analysis')
    plot_ljung_box_test(dax_arch_1_std_resid ** 2, lags=lags_).savefig('images/10_dax_arch_1_std_resid_square_ljung_analysis')

    sse_arch_1 = fit_arch(sse_estimation_time_series)
    print(sse_arch_1.summary())
    sse_arch_1_std_resid = sse_arch_1.resid / sse_arch_1.conditional_volatility
    plot_time_series(sse_arch_1_std_resid, lags=lags_).savefig('images/8_sse_arch_1_std_residuals_analysis')
    plot_time_series(sse_arch_1_std_resid ** 2, lags=lags_).savefig('images/9_sse_arch_1_std_resid_square_analysis')
    plot_ljung_box_test(sse_arch_1_std_resid ** 2, lags=lags_).savefig('images/10_sse_arch_1_std_resid_square_ljung_analysis')

    # GARCH Model fitting -> We are fitting two GARCHs, the first one is a GARCH(1,1). We selected
    # this model because there are evidence in the literature that a GARCH(1,1) outperforms higher
    # orders of GARCH models when checking the AIC criteria.
    spx_garch_1_1 = fit_garch(spx_estimation_time_series)
    print(spx_garch_1_1.summary())
    spx_garch_1_1_std_resid = spx_garch_1_1.resid / spx_garch_1_1.conditional_volatility
    plot_time_series(spx_garch_1_1_std_resid, lags=lags_).savefig('images/11_spx_garch_1_1_std_residuals_analysis')
    plot_time_series(spx_garch_1_1_std_resid ** 2, lags=lags_).savefig('images/12_spx_garch_1_1_std_resid_square_analysis')
    plot_ljung_box_test(spx_garch_1_1_std_resid ** 2, lags=lags_).savefig('images/13_spx_garch_1_1_std_resid_square_ljung_analysis')

    dax_garch_1_1 = fit_garch(dax_estimation_time_series)
    print(dax_garch_1_1.summary())
    dax_garch_1_1_std_resid = dax_garch_1_1.resid / dax_garch_1_1.conditional_volatility
    plot_time_series(dax_garch_1_1_std_resid, lags=lags_).savefig('images/11_dax_garch_1_1_std_residuals_analysis')
    plot_time_series(dax_garch_1_1_std_resid ** 2, lags=lags_).savefig('images/12_dax_garch_1_1_std_resid_square_analysis')
    plot_ljung_box_test(dax_garch_1_1_std_resid ** 2, lags=lags_).savefig('images/13_dax_garch_1_1_std_resid_square_ljung_analysis')

    sse_garch_1_1 = fit_garch(sse_estimation_time_series)
    print(sse_garch_1_1.summary())
    sse_garch_1_1_std_resid = sse_garch_1_1.resid / sse_garch_1_1.conditional_volatility
    plot_time_series(sse_garch_1_1_std_resid, lags=lags_).savefig('images/11_sse_garch_1_1_std_residuals_analysis')
    plot_time_series(sse_garch_1_1_std_resid ** 2, lags=lags_).savefig('images/12_sse_garch_1_1_std_resid_square_analysis')
    plot_ljung_box_test(sse_garch_1_1_std_resid ** 2, lags=lags_).savefig('images/13_sse_garch_1_1_std_resid_square_ljung_analysis')

    # The second model is a GARCH that uses the best ARIMA estimation for p, o and q to check if
    # this model outperforms the previous one.
    spx_garch_best = fit_garch(spx_estimation_time_series, spx_arima_best_order[0], spx_arima_best_order[1], spx_arima_best_order[2])
    print(spx_garch_best.summary())
    spx_garch_best_std_resid = spx_garch_best.resid / spx_garch_best.conditional_volatility
    plot_time_series(spx_garch_best_std_resid, lags=lags_).savefig('images/14_spx_garch_best_std_resid_square_analysis')
    plot_time_series(spx_garch_best_std_resid ** 2, lags=lags_).savefig('images/15_spx_garch_best_std_resid_square_analysis')
    plot_ljung_box_test(spx_garch_best_std_resid ** 2, lags=lags_).savefig('images/16_spx_garch_best_std_resid_square_ljung_analysis')

    dax_garch_best = fit_garch(dax_estimation_time_series, dax_arima_best_order[0], dax_arima_best_order[1], dax_arima_best_order[2])
    print(dax_garch_best.summary())
    dax_garch_best_std_resid = dax_garch_best.resid / dax_garch_best.conditional_volatility
    plot_time_series(dax_garch_best_std_resid, lags=lags_).savefig('images/14_dax_garch_best_std_resid_square_analysis')
    plot_time_series(dax_garch_best_std_resid ** 2, lags=lags_).savefig('images/15_dax_garch_best_std_resid_square_analysis')
    plot_ljung_box_test(dax_garch_best_std_resid ** 2, lags=lags_).savefig('images/16_dax_garch_best_std_resid_square_ljung_analysis')

    sse_garch_best = fit_garch(sse_estimation_time_series, sse_arima_best_order[0], sse_arima_best_order[1], sse_arima_best_order[2])
    print(sse_garch_best.summary())
    sse_garch_best_std_resid = sse_garch_best.resid / sse_garch_best.conditional_volatility
    plot_time_series(sse_garch_best_std_resid, lags=lags_).savefig('images/14_sse_garch_best_std_resid_square_analysis')
    plot_time_series(sse_garch_best_std_resid ** 2, lags=lags_).savefig('images/15_sse_garch_best_std_resid_square_analysis')
    plot_ljung_box_test(sse_garch_best_std_resid ** 2, lags=lags_).savefig('images/16_sse_garch_best_std_resid_square_ljung_analysis')


if __name__ == '__main__':
    """Main entry point of the Python Script.
    """
    main()
