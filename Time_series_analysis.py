import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def Test():
    for i in range(1, 4):
        source = (f'~/Desktop/time_series_data/simulated/signal_{i}.csv')
        series = pd.read_csv(source, header=0, index_col=0)
        print(series.head())
        # series.plot()
        val = series['y']
        idx = np.array(range(len(val)))
        print('This is idx: ', idx)
        is_out = series['is_anomaly'] == 1
        plt.figure()
        plt.plot(idx, val)
        plt.plot(idx[is_out], val[is_out], 'ro')
    plt.show()
    # print(df)


if __name__ == '__main__':
    Test()
