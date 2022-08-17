#data processing
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Model evaluation
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def TestReal():
    for i in range(3, 4):
        source = (f'~/Desktop/time_series_data/real/signal_{i}.csv')
        series = pd.read_csv(source)
        series_unl = series[['timestamp','y']]
        print(type(series_unl))
        #print(series_unl.head())
        series_unl.rename(columns={'timestamp': 'ds'}, inplace=True)
        #print(series.info())
        print(series_unl.tail())

        # series.plot()
        val = series['y']
        idx = np.array(range(len(val)))
        is_out = series['is_anomaly'] == 1
        sns.set(rc={'figure.figsize': (12, 8)})

        sns.lineplot(x=idx, y=series['y'])
        plt.plot(idx[is_out], val[is_out], 'ro')
        plt.figure()
        #plt.plot(idx, val)
    plt.show()
    # print(df)

    #Add seasonality
    model = Prophet(interval_width=0.99, weekly_seasonality=True)
    model.fit(series_unl)

    #Make predictions:
    forecast = model.predict(series_unl)

    #visualize
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    model.plot(forecast); #Add semi-colon to remove the duplicate chart
    plt.figure()
    plt.show()

    #Check the components plot for the trend, weekly seasonality, and yearly seasonality
    model.plot_components(forecast)
    plt.show()
    series_unl['ds'] = series_unl['ds'].astype('datetime64')
    performance = pd.merge(series_unl, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    #Check MAE value:

    performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
    print(f'The MAE for the model is {performance_MAE}')
    performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
    print(f'The MAPE for the model is {performance_MAPE}')

    #Identify anomaly:

    performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y < rows.yhat_lower) | (rows.y > rows.yhat_upper))
                                                else 0, axis=1)
    print(performance['anomaly'].value_counts())
    anomalies = performance[performance['anomaly'] == 1].sort_values(by='ds')
    print(anomalies)

    final_table = pd.merge(performance, series, on='y')
    final_table = final_table.drop(['timestamp', 'yhat_upper', 'yhat_lower'], axis=1)
    #print(final_table.head())

    true_positive = np.sum(np.array(final_table['anomaly'] == 1, dtype=bool) * np.array(final_table['is_anomaly']))
    print(true_positive)

    false_positive = np.sum(np.array(final_table['anomaly'] == 1, dtype=bool) * np.array(final_table['is_anomaly'] == 0))
    true_negative = np.sum(np.array(final_table['anomaly'] == 0, dtype=bool) * np.array(final_table['is_anomaly'] == 0))
    false_negative = np.sum(np.array(final_table['anomaly'] == 0, dtype=bool) * np.array(final_table['is_anomaly'] == 1))
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    print(f'True positive{true_positive} False negative {false_negative} False positive {false_positive}')
    print(f'This is the precision: {precision} and this is the Recall{recall}')
    sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly')
    sns.lineplot(x='ds', y='yhat', data=performance, color='black')

    #plt.plot(idx[is_out], val[is_out], 'ro')
    plt.figure()
    plt.show()





if __name__ == '__main__':
    TestReal()
