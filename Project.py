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
    """
    Returns the Adj Close prices from the provided ticker using Yahoo Finance

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
    series
        a list of daily prices
    """

    return web.DataReader(symbol, 'yahoo', start=start, end=end)['Adj Close']


def plot_time_series(data, lags=None, title=None, filename=None):
    """
    Saves time series plot figure of the provided data in filename.

    Parameters
    ==========
    data : series
        One-dimensional ndarray with axis labels (including time series).
    lags : {int, array_like}
        An int or array of lag values, used on horizontal axis.
    title : string
        The title that will be set for the whole figure.
    filename : string
        File to save the plot result
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

    fig.savefig(filename.lower())
    plt.close()


def plot_histogram(data, n_bins=None, title=None, filename=None):
    """
    Saves histogram plot figure of the provided data in filename.

    Parameters
    ==========
    data : series
        One-dimensional ndarray with axis labels (including time series).
    n_bins : {int, array_like}
        An int or array of number of histogram bins, used on horizontal axis.
    title : string
        The title that will be set for the whole figure.
    filename : string
        File to save the plot result
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

    fig.savefig(filename.lower())
    plt.close()


def plot_ljung_box_test(data, lags=[10], title=None, filename=None):
    """
    Saves Ljung-box Test plot figure of the provided data in filename.

    Parameters
    ==========
    time_series : series
        One-dimensional ndarray with axis labels (including time series).

    Returns
    =======
    data : series
        One-dimensional ndarray with axis labels (including time series).
    n_bins : {int, array_like}
        An int or array of number of histogram bins, used on horizontal axis.
    title : string
        The title that will be set for the whole figure.
    filename : string
        File to save the plot result
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
            plt.annotate(round(Z, 4), xy=(X, Y), xytext=(-5, 5), ha='left', textcoords='offset points')

    fig.savefig(filename.lower())
    plt.close()


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
    best_mdl : ARIMAResultWrapper
    """

    best_aic = np.inf
    best_order = None

    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(2)  # [0,1]

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(time_series.dropna(), order=(i, d, j)).fit(method='mle', trend='nc', disp=False)
                    tmp_aic = tmp_mdl.aic

                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except:
                    continue

    return best_aic, best_order, best_mdl


def fit_arch(time_series, q=1):
    """
    Return fit ARCH Model

    Parameters
    ==========
    time_series : series
        One-dimensional ndarray with axis labels (including time series).
    q : int
        Order of the symmetric innovation

    Returns
    =======
    results : ARCHModelResult
        Object containing model results
    """

    am = arch.univariate.ConstantMean(time_series)
    am.volatility = arch.univariate.ARCH(q)
    am.distribution = arch.univariate.StudentsT()

    return am.fit(update_freq=5, disp='off')


def fit_garch(time_series, p=1, o=0, q=1):
    """
    Returns fitted GARCH Model

    Parameters
    ==========
    time_series : series
        One-dimensional ndarray with axis labels (including time series).
    p : int
        Order of the symmetric innovation
    o : int
        Order of the asymmetric innovation
    q : int
        Order of the lagged (transformed) conditional variance

    Returns
    =======
    results : ARCHModelResult
        Object containing model results
    """

    am = arch.univariate.ConstantMean(time_series)
    am.volatility = arch.univariate.GARCH(p, o, q)
    am.distribution = arch.univariate.StudentsT()

    return am.fit(update_freq=5, disp='off')


def plot_garch_forecast(full_data, garch_model, start_slice, end_slice, desc=None, filename=None):
    """
    Saves forecast plot figure of the provided data in filename.

    Parameters
    ==========
    full_data : series
        One-dimensional ndarray with axis labels (including time series).
    garch_model : ARCHModelResult
        GARCH Model for Forecast plotting
    start_slice : datetime
        start date of the slice
    end_slice : datetime
        end date of the slice
    desc : string
        Symbol naming for proper plot
    filename : string
        File to save the plot result
    """

    if not isinstance(full_data, pd.Series):
        full_data = pd.Series(full_data).dropna()

    with plt.style.context('bmh'):
        fig = plt.figure(figsize=(10, 8))
        layout = (2, 1)
        sliced_data = full_data.loc[start_slice:end_slice]
        sliced_volatility = garch_model.conditional_volatility.loc[start_slice:end_slice]
        conf_interval_top = sliced_data + 2 * garch_model.conditional_volatility
        conf_interval_bottom = sliced_data - 2 * garch_model.conditional_volatility
        future_data = full_data.loc[end_slice:]

        forecast = garch_model.forecast(horizon=len(future_data))
        forecast_mean = pd.Series(forecast.mean.dropna().squeeze())
        forecast_mean.index = future_data.index
        volatility_mean = pd.Series(forecast.residual_variance.dropna().squeeze())
        volatility_mean.index = future_data.index
        error_mean = pd.Series(forecast.variance.dropna().squeeze())
        error_mean.index = future_data.index
        forecast_conf_interval_top = forecast_mean + 2 * np.sqrt(error_mean)
        forecast_conf_interval_bottom = forecast_mean - 2 * np.sqrt(error_mean)

        returns_ax = plt.subplot2grid(layout, (0, 0))
        volatility_ax = plt.subplot2grid(layout, (1, 0))
        sliced_data.plot(ax=returns_ax, color='k', label='past', linewidth=1)
        conf_interval_top.plot(ax=returns_ax, color='darkorange', linestyle='dashed', label='$past + 2\widehat{\sigma}_t$', linewidth=1)
        conf_interval_bottom.plot(ax=returns_ax, color='darkorange', linestyle='dashed', label='$past - 2\widehat{\sigma}_t$', linewidth=1)
        future_data.plot(ax=returns_ax, color='darkblue', linewidth=1, label='realized')
        forecast_mean.plot(ax=returns_ax, color='darkgreen', linewidth=1, label='$\widehat{r}_{T+h}$')
        forecast_conf_interval_top.plot(ax=returns_ax, color='green', linestyle='dashed', label='$\widehat{r}_{T+h} + 2 \sigma$', linewidth=1)
        forecast_conf_interval_bottom.plot(ax=returns_ax, color='green', linestyle='dashed', label='$\widehat{r}_{T+h} - 2 \sigma$', linewidth=1)
        returns_ax.fill_between(forecast_mean.index, forecast_conf_interval_bottom, forecast_conf_interval_top, color='honeydew')
        returns_ax.axvline(x=end_slice, color='k', linestyle='dashed', linewidth=1)
        returns_ax.legend()
        returns_ax.set_title(f'Returns forecast for {desc} ($r_t$)')

        sliced_volatility.plot(ax=volatility_ax, color='k', label='past', linewidth=1)
        volatility_mean.plot(ax=volatility_ax, color='darkgreen', linewidth=1, label='$\widehat{\sigma^2}_{T+h}$')
        volatility_ax.axvline(x=end_slice, color='k', linestyle='dashed', linewidth=1)
        volatility_ax.legend()
        volatility_ax.set_title(f'Conditional volatility forecast for {desc} ($\sigma^2$)')

        plt.tight_layout()

    fig.savefig(filename.lower())
    plt.close()


def run_analysis(desc, symbol, start, end, start_forecast, end_slice, lags_, hist_n_bins_):
    """
    Main script.
    """

    with open(f'output/output_{desc.lower()}.txt', 'w') as f:

        # Raw adjusted close prices
        adj_close_prices = get_prices(symbol, start, end)

        # Time Series Analysis
        log_returns_percent = np.log(adj_close_prices/adj_close_prices.shift(1)).dropna() * 100

        # Lets inspect the time series to check for its properties.
        plot_time_series(adj_close_prices, lags=lags_, title=f'Time series property for {desc} adjusted close prices', filename=f'images/0_{desc}_series_analysis')

        # Unit root tests. The null hypothesis of the ADF test is that the data does not have a unit root process.
        # p-value <= alpha indicates that the data does not have a unit root, hence it is stationary
        adf = smt.stattools.adfuller(adj_close_prices)
        print(f'Augmented Dick-Fuller test for {desc}  time series: {adf}', file=f)

        # Obviously all the series are non-stationary analysing the test.
        # We now check the presence of a unit root in the first difference of the process.
        plot_time_series(np.diff(adj_close_prices), lags=lags_, title=f'Time series property for first difference of {desc} adjusted close prices', filename=f'images/1_{desc}_lag_series_analysis')

        adf_diff = smt.stattools.adfuller(np.diff(adj_close_prices))
        print(f'Augmented Dick-Fuller test for first difference {desc} time series: {adf_diff}', file=f)

        # The first difference of the series are stationary, hence the returns should be stationary.
        adf_log_ret = smt.stattools.adfuller(log_returns_percent)
        print(f'Augmented Dick-Fuller test for log-returns of {desc}  time series: {adf_log_ret}', file=f)

        # The log-returns of the series are stationary. We will use the log-returns from here on as the
        # data we use to run our models.

        # Histogram of log-returns.
        plot_histogram(log_returns_percent, n_bins=hist_n_bins_, title=f'Histogram of log-returns for {desc}', filename=f'images/2_{desc}_log_ret_hist')

        plot_time_series(log_returns_percent, lags=lags_, title=f'Time series property for log-returns of {desc}', filename=f'images/3_{desc}_log_returns_analysis')

        # Because we have a stationary process and the time series plots show autocorrelation, we will
        # try to fit the best ARIMA model for each one of the series. Before we can fit a GARCH model,
        # we need to understand if there are certain characteristics as an outcome to other models being
        # estimated.
        estimation_time_series = log_returns_percent.loc[:end_slice]

        # Find Best ARIMA Model for fitting GARCH
        arima_best_aic, arima_best_order, arima_best_mdl = find_best_arima_model(estimation_time_series)

        print('{} -> aic: {:6.5f} | order: {}'.format(desc, arima_best_aic, arima_best_order), file=f)

        # Now lets see how our ARIMA models are performing.
        print(f'-----------------------------------------------------------------------------\n'
              f'{desc} ARIMA {arima_best_order}\n'
              f'-----------------------------------------------------------------------------\n'
              f'{arima_best_mdl.summary()}\n', file=f)

        # Lets check for the presence of arch effects in the data.
        arch_test = sms.diagnostic.het_arch(arima_best_mdl.resid, ddof=arima_best_order[0] + arima_best_order[2])
        print(f"Engle's test for ARCH Effects for ARIMA Residuals of {desc}: {arch_test}", file=f)

        plot_time_series(arima_best_mdl.resid, lags=lags_, title=f'Time series property for ARIMA residuals of {desc}', filename=f'images/4_{desc}_arima_resid_analysis')

        # Ljung-Box test of the residuals, if the values are higher than the significance level, there
        # are no evidences to reject the null hypothesis that the data is independently distributed, or
        # not exhibiting serial correlation.
        plot_ljung_box_test(arima_best_mdl.resid, lags=lags_, title=f'Ljung-Box test for ARIMA residuals of {desc}', filename=f'images/5_{desc}_arima_resid_ljung_analysis')

        # As of here, the ARIMA looks like a good model, because its residues seems to be white noise.
        # We now need to understand if there is a relationship in the square of the residuals, which
        # would imply existence of ARCH effects.
        plot_time_series(arima_best_mdl.resid ** 2, lags=lags_, title=f'Time series property for squared ARIMA residuals of {desc}', filename=f'images/6_{desc}_arima_resid_square_analysis')

        # The Ljung-Box test on the square of the residuals of the ARIMA model show exactly the expected
        # behaviour of having correlation. We need to fit a model with ARCH characteristics.

        plot_ljung_box_test(arima_best_mdl.resid ** 2, lags=lags_, title=f'Ljung-Box test for ARIMA residuals of {desc}', filename=f'images/7_{desc}_arima_square_resid_ljung_analysis')

        # ARCH Model fitting -> We are fitting one ARCH, to check whether we need to do a GARCH, then
        # checking the standardized squared residuals of the model we fitted, we see the existence of
        # autocorrelation, which leads to the need of a GARCH model.
        arch_1 = fit_arch(estimation_time_series)
        arch_1_std_resid = arch_1.resid / arch_1.conditional_volatility

        print(f'-----------------------------------------------------------------------------\n'
              f'{desc} ARCH(1)\n'
              f'-----------------------------------------------------------------------------\n'
              f'{arch_1.summary()}\n', file=f)

        plot_time_series(arch_1_std_resid, lags=lags_, title=f'Time series property for ARCH(1) residuals of {desc}', filename=f'images/8_{desc}_arch_1_std_residuals_analysis')
        plot_time_series(arch_1_std_resid ** 2, lags=lags_, title=f'Time series property for squared ARCH(1) residuals of {desc}', filename=f'images/9_{desc}_arch_1_std_resid_square_analysis')
        plot_ljung_box_test(arch_1_std_resid ** 2, lags=lags_, title=f'Ljung-Box test for ARCH(1) residuals of {desc}', filename=f'images/10_{desc}_arch_1_std_resid_square_ljung_analysis')

        # GARCH Model fitting -> We are fitting two GARCHs, the first one is a GARCH(1,1). We selected
        # this model because there are evidence in the literature that a GARCH(1,1) outperforms higher
        # orders of GARCH models when checking the AIC criteria.
        garch_1_1 = fit_garch(estimation_time_series)
        garch_1_1_std_resid = garch_1_1.resid / garch_1_1.conditional_volatility

        print(f'-----------------------------------------------------------------------------\n'
              f'{desc} GARCH(1,1)\n'
              f'-----------------------------------------------------------------------------\n'
              f'{garch_1_1.summary()}\n', file=f)

        plot_time_series(garch_1_1_std_resid, lags=lags_, title=f'Time series property for GARCH(1,1) residuals of {desc}', filename=f'images/11_{desc}_garch_1_1_std_residuals_analysis')
        plot_time_series(garch_1_1_std_resid ** 2, lags=lags_, title=f'Time series property for squared GARCH(1,1) residuals of {desc}', filename=f'images/12_{desc}_garch_1_1_std_resid_square_analysis')
        plot_ljung_box_test(garch_1_1_std_resid ** 2, lags=lags_, title=f'Ljung-Box test for GARCH(1,1) residuals of {desc}', filename=f'images/13_{desc}_garch_1_1_std_resid_square_ljung_analysis')

        # The second model is a GARCH that uses the best ARIMA estimation for p, o and q to check if
        # this model outperforms the previous one.
        garch_best = fit_garch(estimation_time_series, arima_best_order[0], arima_best_order[1], arima_best_order[2])
        garch_best_std_resid = garch_best.resid / garch_best.conditional_volatility

        print(f'-----------------------------------------------------------------------------\n'
              f'{desc} GARCH with ARIMA estimations\n'
              f'-----------------------------------------------------------------------------\n'
              f'{garch_best.summary()}\n', file=f)

        plot_time_series(garch_best_std_resid, lags=lags_,  title=f'Time series property for GARCH with ARIMA estimation residuals of {desc}', filename=f'images/14_{desc}_garch_best_std_resid_square_analysis')
        plot_time_series(garch_best_std_resid ** 2, lags=lags_, title=f'Time series property for squared GARCH with ARIMA estimation residuals of {desc}', filename=f'images/15_{desc}_garch_best_std_resid_square_analysis')
        plot_ljung_box_test(garch_best_std_resid ** 2, lags=lags_, title=f'Ljung-Box test for GARCH with ARIMA estimation residuals of {desc}', filename=f'images/16_{desc}_garch_best_std_resid_square_ljung_analysis')

        # GARCH Model fitting -> We are fitting two GARCHs, the first one is a GARCH(1,1). We selected
        # this model because there are evidence in the literature that a GARCH(1,1) outperforms higher
        # orders of GARCH models when checking the AIC criteria.
        garch_1_1_full = fit_garch(log_returns_percent)
        garch_1_1_full_std_resid = garch_1_1_full.resid / garch_1_1_full.conditional_volatility

        print(f'-----------------------------------------------------------------------------\n'
              f'{desc} GARCH(1,1) estimation with COVID-19\n'
              f'-----------------------------------------------------------------------------\n'
              f'{garch_1_1_full.summary()}\n', file=f)

        plot_time_series(garch_1_1_full_std_resid, lags=lags_, title=f'Time series property for GARCH(1,1) estimation with COVID-19 residuals of {desc}', filename=f'images/17_{desc}_garch_1_1_full_std_residuals_analysis')
        plot_time_series(garch_1_1_full_std_resid ** 2, lags=lags_, title=f'Time series property for squared GARCH(1,1) estimation with COVID-19 residuals of {desc}', filename=f'images/18_{desc}_garch_1_1_full_std_resid_square_analysis')
        plot_ljung_box_test(garch_1_1_full_std_resid ** 2, lags=lags_, title=f'Ljung-Box test for GARCH(1,1) estimation with COVID-19 residuals of {desc}', filename=f'images/19_{desc}_garch_1_1_full_std_resid_square_ljung_analysis')

        plot_garch_forecast(log_returns_percent, garch_1_1, start_forecast, end_slice, filename=f'images/20_{desc}_garch_1_1_forecast', desc=desc)


def main():
    """
    Define attributes for GARCH Model and run analysis
    """

    start = '2015-01-01'
    end = '2020-05-01'
    start_forecast = '2019-01-01'
    end_slice = '2020-01-01'

    symbols = ['SPX', '^GSPC'], ['DAX', '^GDAXI'], ['SSE', 'SSE.L']

    lags_ = 30
    hist_n_bins_ = 50

    for symbol in symbols:
        run_analysis(symbol[0], symbol[1], start, end, start_forecast, end_slice, lags_, hist_n_bins_)


if __name__ == '__main__':
    """
    Main entry point of the Python Script.
    """
    main()
