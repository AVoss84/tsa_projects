import numpy as np
import pandas as pd


def embed(data : pd.DataFrame, lags : int, dropnan=True):
  df = pd.DataFrame(data) 
  colnames = data.columns
  cols, names = list(), list()
  k = data.shape[1]
  for j in range(0,k):
    ts = df.iloc[:,j]
    for i in range(0,lags+1):
      cols.append(ts.shift(i))                                  # lag series/shift series up
      names.append(str(colnames[j]) + '_lag' + str(i))             # make names
  agg = pd.concat(cols, axis=1)                          # concatenate the matrix list elements to dataframe -> cbind
  agg.columns = names  
  if dropnan:
    agg.dropna(inplace=True)              	# drop rows with NaN values
  return agg


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def logsumexp(x):
    """Log-Sum-Exp Trick"""
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def indBrkL(k : np.array):
  """
  Break indicator matrices for drift/trend breaks
  """
  k[0] = 0.
  nofb = len(k) - 2    # substract start and end date
  tim = np.array(range(int(k[len(k)-1])))
  ei1 = np.zeros((int(k[len(k)-1]), len(k) - 1))    # drift
  ei2 = np.zeros((int(k[len(k)-1]), len(k) - 1))    # trend
  i = 1
  while i < len(k):
	  ei1[:, i - 1] = (k[i - 1] <= tim) & (tim < k[i])     # drift break
	  ei2[:, i - 1] = ei1[:, i - 1] * tim     # trend break
	  i += 1
  #drift = pd.DataFrame(ei1, columns=['Drift_'+str(s) for s in range(nofb+1)])	
  #trend = pd.DataFrame(ei2, columns=['Trend_'+str(s) for s in range(nofb+1)])	
  return dict(ei=np.concatenate((ei1,ei2), axis=1), ei1=ei1)

