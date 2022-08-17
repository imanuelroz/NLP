import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as sc
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics import tsaplots
import seaborn as sns


def Plot():
    for i in range(1, 20):
        source = (f'~/Desktop/time_series_data/simulated/signal_{i}.csv')
        series = pd.read_csv(source, header=0, index_col=0)
        # series.plot()
        val = series['y']
        idx = np.array(range(len(val)))
        is_out = series['is_anomaly'] == 1
        plt.figure()
        plt.plot(idx, val)
        plt.plot(idx[is_out], val[is_out], 'ro')
    plt.show()

def TestSTLSimulati():
    for i in range(1, 101):
        source = (f'~/Desktop/time_series_data/simulated/signal_{i}.csv')
        series = pd.read_csv(source)
        #is_real_anomaly = series['timestamp'][series['is_anomaly'] == 1]
        real_anomalies = series[series['is_anomaly'] == 1]
        #print(real_anomalies)
        n_real_an = len(real_anomalies)
        temp = np.array([series['y'][i] for i in range(len(series['y']))])
        #print(temp)
        series = pd.Series(temp, index=pd.date_range("2014-11-23 07:00:00", periods=len(series['y']), freq="H"),
                           name="y")

        stl = STL(series)
        result = stl.fit()
        seasonal, trend, resid = result.seasonal, result.trend, result.resid

        mu, sigma = resid.mean(), resid.std()
        l = 4.9  #Start with 3 and then check if there are improvements with bigger numbers
        lower, upper = mu - l * sigma, mu + l * sigma
        new_residual = seasonal + resid #simulated data are mainly sines with linear trends plus a noise
        # pred_anomalies = series[(resid < lower) | (resid > upper)]
        pred_anomalies = series[(new_residual < lower) | (new_residual > upper)]
        n_predicted_an = len(pred_anomalies)
        if n_predicted_an != n_real_an:
            print(f'Index: {i}, predicted: {n_predicted_an}, real: {n_real_an}')

        # plt.figure(figsize=(8, 6))
        #
        # plt.subplot(4, 1, 1)
        # plt.plot(series, '-b')
        # plt.plot(series[is_real_anomaly], '.r')
        # plt.title('Original Series', fontsize=16)
        #
        # plt.subplot(4, 1, 2)
        # plt.plot(trend, '-b')
        # plt.title('Trend', fontsize=16)
        #
        # # plt.subplot(4, 1, 3)
        # # plt.plot(seasonal, '-b')
        # # plt.title('Seasonal', fontsize=16)
        #
        # plt.subplot(4, 1, 3)
        # plt.plot(new_residual, '-b')
        # plt.title('Residual', fontsize=16)
        #
        # plt.subplot(4, 1, 4)
        # plt.plot(series)
        # plt.plot(pred_anomalies, '.r')
        #
        # plt.tight_layout()
        #
        # plt.show()

def TestSTLReali():
    for i in range(1, 68):
        source = (f'~/Desktop/time_series_data/real/signal_{i}.csv')
        series = pd.read_csv(source)
        #is_real_anomaly = series['timestamp'][series['is_anomaly'] == 1]
        real_anomalies = series[series['is_anomaly'] == 1]
        n_real_an = len(real_anomalies)
        # print(len(series['y']))
        temp = np.array([series['y'][i] for i in range(len(series['y']))])
        # print(temp)
        series = pd.Series(temp, index=pd.date_range("2014-11-23 07:00:00", periods=len(series['y']), freq="H"),
                           name="y")

        stl = STL(series)
        result = stl.fit()
        seasonal, trend, resid = result.seasonal, result.trend, result.resid
        mu, sigma = resid.mean(), resid.std()
        l = 6.5
        lower, upper = mu - l * sigma, mu + l * sigma
        # new_residual = seasonal + resid
        pred_anomalies = series[(resid < lower) | (resid > upper)]
        print(pred_anomalies)
        # pred_anomalies = series[(new_residual < lower) | (new_residual > upper)]
        n_predicted_an = len(pred_anomalies)
        if n_predicted_an != n_real_an:
            print(f'Index: {i}, predicted: {n_predicted_an}, real: {n_real_an}')

        # plt.figure(figsize=(8, 6))
        #
        # plt.subplot(5, 1, 1)
        # plt.plot(series, '-b')
        # plt.plot(series[is_real_anomaly], '.r')
        # # plt.plot(real_anomalies, '.r')
        # plt.title('Original Series', fontsize=16)
        #
        # plt.subplot(5, 1, 2)
        # plt.plot(trend, '-b')
        # plt.title('Trend', fontsize=16)
        #
        # plt.subplot(5, 1, 3)
        # plt.plot(seasonal, '-b')
        # plt.title('Seasonal', fontsize=16)
        #
        # plt.subplot(5, 1, 4)
        # plt.plot(resid, '-b')
        # plt.title('Residual', fontsize=16)
        #
        # plt.subplot(5, 1, 5)
        # plt.plot(series)
        # plt.plot(pred_anomalies, '.r')
        #
        # plt.tight_layout()
        #
        # plt.show()

def TestProphetReali():
    for i in range(1, 10):
        source = (f'~/Desktop/time_series_data/real/signal_{i}.csv')
        series = pd.read_csv(source)
        real_anomalies = series[series['is_anomaly'] == 1]
        print(real_anomalies)
        print(type(real_anomalies))
        #series_unl = pd.DataFrame(series[['timestamp', 'y']])
        series_unl = series.loc[:, series.columns.drop(['is_anomaly'])]
        series_unl.rename(columns={'timestamp': 'ds'}, inplace=True)

        # Add seasonality
        model = Prophet(interval_width=0.99, weekly_seasonality=True)
        model.fit(series_unl)

        # Make predictions:
        forecast = model.predict(series_unl)

        series_unl['ds'] = series_unl['ds'].astype('datetime64')
        performance = pd.merge(series_unl, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        performance['predicted_anomaly'] = performance.apply(
            lambda rows: 1 if ((rows.y < rows.yhat_lower) | (rows.y > rows.yhat_upper))
            else 0, axis=1)
        #print(performance['predicted_anomaly'].value_counts())
        pred_anomalies = performance[performance['predicted_anomaly'] == 1].sort_values(by='ds')
        #print(pred_anomalies)
        anomaly_table = pd.merge(performance, series, on='y')
        anomaly_table = anomaly_table.drop(['y','timestamp', 'yhat_upper', 'yhat_lower'], axis=1)
        # print(final_table.head())
        true_positive = np.sum(np.array(anomaly_table['predicted_anomaly'] == 1, dtype=bool) * np.array(anomaly_table['is_anomaly']))
        #print(true_positive)
        false_positive = np.sum(
            np.array(anomaly_table['predicted_anomaly'] == 1, dtype=bool) * np.array(anomaly_table['is_anomaly'] == 0))
        true_negative = np.sum(
            np.array(anomaly_table['predicted_anomaly'] == 0, dtype=bool) * np.array(anomaly_table['is_anomaly'] == 0))
        false_negative = np.sum(
            np.array(anomaly_table['predicted_anomaly'] == 0, dtype=bool) * np.array(anomaly_table['is_anomaly'] == 1))
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        accuracy = true_positive + true_negative /(true_positive + true_negative + false_negative + false_positive)
        #print(f'True positive{true_positive} False negative {false_negative} False positive {false_positive}')
        print(f'The precision is : {precision} and the Recall is : {recall}')
        n_predicted_an = len(pred_anomalies)
        real_anomalies = series[series['is_anomaly'] == 1]
        n_real_an = len(real_anomalies)
        if n_predicted_an != n_real_an:
            print(f'Index: {i}, predicted: {n_predicted_an}, real: {n_real_an}')
        sns.scatterplot(x='ds', y='y', data=performance, hue='predicted_anomaly')
        #plt.plot(real_anomalies['timestamp'].astype('datetime64'), real_anomalies['y'], 'ro')
        sns.lineplot(x='ds', y='yhat', data=performance, color='black')
        plt.figure()
        plt.show()

if __name__ == '__main__':
    #Plot()
    TestSTLSimulati()
    #TestSTLReali()
    #TestProphetReali()