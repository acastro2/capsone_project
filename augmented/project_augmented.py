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
p = print

p('Machine: {} {}\n'.format(os.uname().sysname,os.uname().machine))
p(sys.version)

end = '2020-05-01'
start = '2015-01-01'
get_px = lambda x: web.DataReader(x, 'yahoo', start=start, end=end)['Adj Close']

SPX, DAX, SSE = '^GSPC', '^GDAXI','SSE.L'

symbols = [SPX, DAX, SSE]
# raw adjusted close prices
data = pd.DataFrame({sym:get_px(sym) for sym in symbols})
p(data)
# log returns
lrets = np.log(data/data.shift(1)).dropna()

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y).dropna()
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return fig

def acf_pacf_plot(y, lags=None, figsize=(10, 3), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y).dropna()
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (1, 2)
        acf_ax = plt.subplot2grid(layout, (0, 0))
        pacf_ax = plt.subplot2grid(layout, (0, 1))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        plt.tight_layout()
    return fig

def tsdiag(y, figsize = (14,8), title = "", lags = 10):
    #The data:
    tmp_data = pd.Series(y)
    #The Ljung-Box test results for the first k lags:
    tmp_acor = list(sm_stat.diagnostic.acorr_ljungbox(tmp_data, lags = lags, boxpierce = True))
    # get the p-values
    p_vals = pd.Series(tmp_acor[1])
    #Start the index from 1 instead of 0 (because Ljung-Box test is for lag values from 1 to k)
    p_vals.index += 1
    fig = plt.figure(figsize = figsize)
    #Plot the p-values:
    p_vals.plot(ax = fig.add_subplot(313), linestyle='', marker='o', title = "p-values for Ljung-Box statistic", legend = False)
    #Add the horizontal 0.05 critical value line
    plt.axhline(y = 0.05, color = 'blue', linestyle='--')
    # Annotate the p-value points above and to the left of the vertex
    x = np.arange(p_vals.size) + 1
    for X, Y, Z in zip(x, p_vals, p_vals):
        plt.annotate(round(Z, 4), xy=(X,Y), xytext=(-5, 5), ha = 'left', textcoords='offset points')
    plt.show()
    # Return the statistics:
    col_index = ["Ljung-Box: X-squared", "Ljung-Box: p-value", "Box-Pierce: X-squared", "Box-Pierce: p-value"]
    return pd.DataFrame(tmp_acor, index = col_index, columns = range(1, len(tmp_acor[0]) + 1))
    
# First difference of SPY prices
tsplot(np.diff(data[SPX]), lags=30).savefig('images/spx_series_analysis')
tsplot(np.diff(data[DAX]), lags=30).savefig('images/dax_series_analysis')
tsplot(np.diff(data[SSE]), lags=30).savefig('images/sse_series_analysis')

def _get_best_arima_model(TS):
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(4) # [0,1,2,3]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    return best_aic, best_order, best_mdl

# DataScaleWarning: y is poorly scaled, which may affect convergence of the optimizer when
# estimating the model parameters. The scale of y is 0.0001611. Parameter
# estimation work better when this value is between 1 and 1000. The recommended
# rescaling is 100 * y.
scaled_lrets = lrets * 100

spx_estimation_time_series = scaled_lrets[SPX].loc['2015-01-01':'2020-01-01']
dax_estimation_time_series = scaled_lrets[DAX].loc['2015-01-01':'2020-01-01']
sse_estimation_time_series = scaled_lrets[SSE].loc['2015-01-01':'2020-01-01']

res_tup_spx = _get_best_arima_model(spx_estimation_time_series)
res_tup_dax = _get_best_arima_model(dax_estimation_time_series)
res_tup_sse = _get_best_arima_model(sse_estimation_time_series)

p('SPX -> aic: {:6.5f} | order: {}'.format(res_tup_spx[0], res_tup_spx[1]))
p('DAX -> aic: {:6.5f} | order: {}'.format(res_tup_dax[0], res_tup_dax[1]))
p('SSE -> aic: {:6.5f} | order: {}'.format(res_tup_sse[0], res_tup_sse[1]))


def _fit_garch_model(ts, best_order):
    p_ = best_order[0]
    o_ = best_order[1]
    q_ = best_order[2]

    # Using student T distribution usually provides better fit
    am = arch_model(ts, p=p_, o=o_, q=q_, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    return res

spx_garch = _fit_garch_model(scaled_lrets[SPX].loc['2015-01-01':'2020-01-01'], res_tup_spx[1])
p(spx_garch.summary())
tsplot(spx_garch.resid, lags=30).savefig('images/spx_slice_garch_residuals_analysis')
acf_pacf_plot(spx_garch.resid ** 2, lags=30).savefig('images/spx_slice_garch_squared_residuals_analysis')

spx_garch = _fit_garch_model(scaled_lrets[SPX], res_tup_spx[1])
p(spx_garch.summary())
tsplot(spx_garch.resid, lags=30).savefig('images/spx_garch_residuals_analysis')
acf_pacf_plot(spx_garch.resid ** 2, lags=30).savefig('images/spx_garch_squared_residuals_analysis')


dax_garch = _fit_garch_model(scaled_lrets[DAX].loc['2015-01-01':'2020-01-01'], res_tup_dax[1])
p(dax_garch.summary())
tsplot(dax_garch.resid, lags=30).savefig('images/dax_slice_garch_residuals_analysis')
acf_pacf_plot(dax_garch.resid ** 2, lags=30).savefig('images/dax_slice_garch_squared_residuals_analysis')

dax_garch = _fit_garch_model(scaled_lrets[DAX], res_tup_dax[1])
p(dax_garch.summary())
tsplot(dax_garch.resid, lags=30).savefig('images/dax_garch_residuals_analysis')
acf_pacf_plot(dax_garch.resid ** 2, lags=30).savefig('images/dax_garch_squared_residuals_analysis')



sse_garch = _fit_garch_model(scaled_lrets[SSE].loc['2015-01-01':'2020-01-01'], res_tup_sse[1])
p(sse_garch.summary())
tsplot(sse_garch.resid, lags=30).savefig('images/sse_slice_garch_residuals_analysis')
acf_pacf_plot(sse_garch.resid ** 2, lags=30).savefig('images/sse_slice_garch_squared_residuals_analysis')

sse_garch = _fit_garch_model(scaled_lrets[SSE], res_tup_sse[1])
p(sse_garch.summary())
tsplot(sse_garch.resid, lags=30).savefig('images/sse_garch_residuals_analysis')
acf_pacf_plot(sse_garch.resid ** 2, lags=30).savefig('images/sse_garch_squared_residuals_analysis')
