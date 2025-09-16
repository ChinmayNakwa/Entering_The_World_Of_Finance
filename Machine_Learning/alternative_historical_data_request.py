import pandas_datareader as pdr
import numpy as np

start_date = '2000-01-01'
end_date = '2023-06-01'

data = np.array((pdr.grt_data_fred('DEXUSEU', start = start_date, end = end_date)).dropna())

data = np.diff(data[:, 0])

