import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr #get Yahoo specific data

# import data
def get_data( stocks , start , end ):
    stockData = pdr.get_data_yahoo( stocks , start , end )
    stockData = stockData[ 'Close' ] # want close prices
    returns = stockData.pct_change() #percent change to get the daily changes
    meanReturns = returns.mean()
    covMatrix = returns.cov() #covariance matrix
    return meanReturns, covMatrix

stockList = [ 'CBA' , 'BHP' , 'TLS' , 'NAB' , 'WBC' , 'STO' ] #random stock quotes
stocks = [ stock + '.AX' for stock in stockList ]
endDate = datetime.datetime.now()
startDate = endDate - datetime.timedelta( days = 300 )

meanReturns , covMatrix = get_data( stocks , startDate , endDate )

weights = np.random.random( len( meanReturns ) ) #random gives us between 0-1
weights /= np.sum( weights ) #want all of them to sum to 1



# Monte Carlo Method

mc_sims = 400 # number of simulations
T = 100 #timeframe in days

meanM = np.full( shape = ( T , len( weights ) ) , fill_value = meanReturns )
meanM = meanM.T 

portfolio_sims = np.full( shape = ( T , mc_sims ) , fill_value = 0.0 ) #so floats can be added

initialPortfolio = 10000 #starting portfolio value

for m in range( 0 , mc_sims ):
    Z = np.random.normal( size = ( T , len( weights ) ) ) #uncorrelated RV's
    L = np.linalg.cholesky( covMatrix ) #Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner( L , Z ) #Correlated daily returns for individual stocks # dot product between the different stocks in the portfolio
    portfolio_sims[ : , m ] = np.cumprod( np.inner( weights , dailyReturns.T ) + 1 ) * initialPortfolio

plt.plot( portfolio_sims )
plt.ylabel( 'Portfolio Value ($)' )
plt.xlabel( 'Days' )
plt.title( 'MC simulation of a stock portfolio' )
plt.show()

def mcVaR( returns , alpha = 5 ):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance( returns , pd.Series ):
        return np.percentile( returns , alpha )
    else:
        raise TypeError( "Expected a pandas data series." )

def mcCVaR( returns , alpha = 5 ):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance( returns , pd.Series ):
        belowVaR = returns <= mcVaR( returns , alpha = alpha )
        return returns[ belowVaR ].mean()
    else:
        raise TypeError( "Expected a pandas data series." )


portResults = pd.Series( portfolio_sims[ -1 , : ] )

VaR = initialPortfolio - mcVaR( portResults , alpha = 5 )
CVaR = initialPortfolio - mcCVaR( portResults,  alpha = 5 )

print( 'VaR_5 ${}'.format( round( VaR , 2 ) ) )
print( 'CVaR_5 ${}'.format( round( CVaR , 2 ) ) )