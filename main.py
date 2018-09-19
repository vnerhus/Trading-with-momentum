
import sys
import pandas as pd
from pandas_datareader import data
import numpy as np
import helper
import project_helper

print()
#df = pd.read_csv('../../data/project_1/eod-quotemedia.csv', parse_dates=['date'], index_col=False)
#close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')

tickers = ['ABT', 'ABBV', 'ACN', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'APD', 'AKAM', 'AA', 'AGN', 'ALXN',
'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI',
'AON','APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY', 'BLL', 'BAC', 'BK', 'BCR', 'BAX', 'BBT',
'BDX', 'BBBY', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BMY', 'CHRW', 'CA',
'COG', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX',
'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CCE', 'CTSH', 'CL']
#'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL',
#'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN',
#'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES',
#'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI',
#'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL',
#'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP',
#'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ']

start_date = '2017-01-01'
end_date = '2017-12-31'
df = data.DataReader(tickers, 'iex', start_date, end_date)
close = df['close']
close.index = pd.to_datetime(close.index)

print('Loaded Data')

#apple_ticker = 'AAPL'

def resample_prices(close_prices, freq='M'):
    """
    Resample close prices for each ticker at specified frequency.

    Parameters
    ----------
    close_prices : DataFrame
        Close prices for each ticker and date
    freq : str
        What frequency to sample at
        For valid freq choices, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    Returns
    -------
    prices_resampled : DataFrame
        Resampled prices for each ticker and date
    """

    return close_prices.resample(freq).last()

monthly_close = resample_prices(close)
def compute_log_returns(prices):
    """
    Compute log returns for each ticker.

    Parameters
    ----------
    prices : DataFrame
        Prices for each ticker and date

    Returns
    -------
    log_returns : DataFrame
        Log returns for each ticker and date
    """

    prices = np.log(prices)-np.log(prices.shift(1))
    return prices

monthly_close_returns = compute_log_returns(monthly_close)

def shift_returns(returns, shift_n):
    """
    Generate shifted returns

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    shift_n : int
        Number of periods to move, can be positive or negative

    Returns
    -------
    shifted_returns : DataFrame
        Shifted returns for each ticker and date
    """

    return returns.shift(shift_n)

prev_returns = shift_returns(monthly_close_returns, 1)
lookahead_returns = shift_returns(monthly_close_returns, -1)

def get_top_n(prev_returns, top_n):
    """
    Select the top performing stocks

    Parameters
    ----------
    prev_returns : DataFrame
        Previous shifted returns for each ticker and date
    top_n : int
        The number of top performing stocks to get

    Returns
    -------
    top_stocks : DataFrame
        Top stocks for each ticker and date marked with a 1
    """
    top_stocks = prev_returns.copy()
    for index, row in prev_returns.iterrows():
        top = row.nlargest(top_n)
        top_stocks.loc[index, top.index] = 1
        top_stocks.loc[index, top_stocks.columns.drop(top.index)] = 0
    return top_stocks.astype(int)

top_bottom_n = 50
df_long = get_top_n(prev_returns, top_bottom_n)
df_short = get_top_n(-1*prev_returns, top_bottom_n)
project_helper.print_top(df_long, 'Longed Stocks')
project_helper.print_top(df_short, 'Shorted Stocks')

def portfolio_returns(df_long, df_short, lookahead_returns, n_stocks):
    """
    Compute expected returns for the portfolio, assuming equal investment in each long/short stock.

    Parameters
    ----------
    df_long : DataFrame
        Top stocks for each ticker and date marked with a 1
    df_short : DataFrame
        Bottom stocks for each ticker and date marked with a 1
    lookahead_returns : DataFrame
        Lookahead returns for each ticker and date
    n_stocks: int
        The number number of stocks chosen for each month

    Returns
    -------
    portfolio_returns : DataFrame
        Expected portfolio returns for each ticker and date
    """

    return (lookahead_returns*df_long+(lookahead_returns*-1)*df_short)/n_stocks

expected_portfolio_returns = portfolio_returns(df_long, df_short, lookahead_returns, 2*top_bottom_n)

expected_portfolio_returns_by_date = expected_portfolio_returns.T.sum().dropna()
portfolio_ret_mean = expected_portfolio_returns_by_date.mean()
portfolio_ret_ste = expected_portfolio_returns_by_date.sem()
portfolio_ret_annual_rate = (np.exp(portfolio_ret_mean * 12) - 1) * 100

print("""
Mean:                       {:.6f}
Standard Error:             {:.6f}
Annualized Rate of Return:  {:.2f}%
""".format(portfolio_ret_mean, portfolio_ret_ste, portfolio_ret_annual_rate))

from scipy import stats

def analyze_alpha(expected_portfolio_returns_by_date):
    """
    Perform a t-test with the null hypothesis being that the expected mean return is zero.

    Parameters
    ----------
    expected_portfolio_returns_by_date : Pandas Series
        Expected portfolio returns for each date

    Returns
    -------
    t_value
        T-statistic from t-test
    p_value
        Corresponding p-value
    """
    t, p = stats.ttest_1samp(expected_portfolio_returns_by_date, 0)
    return t, p/2


t_value, p_value = analyze_alpha(expected_portfolio_returns_by_date)
print("""
Alpha analysis:
 t-value:        {:.3f}
 p-value:        {:.6f}
""".format(t_value, p_value))
