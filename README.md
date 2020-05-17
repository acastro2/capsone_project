# World Quant University - Capstone Project

>Group 25 June 2020

## Description

The main goal of this research is to understand if GARCH models maintain superior performance in forecasting volatility, despite the high price fluctuations in the financial markets due to extreme shocks such as the one that happened with the COVID-19 phenomena. In this study, we will focus on three different markets: the USA, Germany, and China. To achieve that, we use data composed of daily quotations of the S\&P 500, DAX 30, and SSE, and spanning over a period of 180 days until the 31st of March 2020. The GARCH models' performance will be assessed using loss functions such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Platforms

This application is platform agnostic.

## Installation

1. [Download](https://www.python.org/downloads/) and install Python 3.7.7
2. Unzip files to local drive in desired folder (example: C:\capstone_project).
3. Open cmd prompt / shell.
4. Navigate to created folder.
5. Install requirements:
   1. Type `pip install -r requirements.txt` in cmd prompt/shell.
   2. Install all requirements.
6. Run `Project.py` by typing Python main.py in command prompt/shell.

## Main Requirements

- Python version 3.7.7 - See https://www.python.org/downloads/ for installation.
- `pip` - Is included with Python 3.7. See https://pip.pypa.io/en/stable/installing/ for more.
- Pandas - see https://pandas.pydata.org/pandas-docs/stable/ for more information.
- NumPy - see https://numpy.org/doc/ for more information.
- SciPy - see https://docs.scipy.org/doc/ for more information.
- Matplotlib  - see https://matplotlib.org/contents.html for more information.
- Statsmodels - see https://www.statsmodels.org for more information.
- Arch - see https://arch.readthedocs.io/en/latest/ for more information.

## Usage examples

This python module does not require any input in order to provide the desire output. Run `Project.py` file.

## References

- [Time Series Analysis (TSA) in Python - Linear Models to GARCH](http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016)
- [Financial Volatility Modelling](http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/03/02_GARCH.html)
- [How to Model Volatility with ARCH and GARCH for Time Series Forecasting in Python](https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/)
- [ARCH Modeling](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html)
- [Time Series Analysis for Financial Data VI— GARCH model and predicting SPX returns](https://medium.com/auquan/time-series-analysis-for-finance-arch-garch-models-822f87f1d755)
- [https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html)
