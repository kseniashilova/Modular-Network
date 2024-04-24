import numpy as np

# TODO
def cross_correlation(data1, data2, lag_limit=10):
    ts1 = np.asarray(data1)
    ts2 = np.asarray(data2)
    len_ts = min(len(ts1), len(ts2))

    ts1 = ts1 - np.mean(ts1)
    ts2 = ts2 - np.mean(ts2)

    corr_values = []
    lags = range(-lag_limit, lag_limit + 1)
    for lag in lags:
        if lag < 0:
            try:
                corr = np.correlate(ts1[:lag], ts2[-lag:])
            except ValueError as e:
                print(e)
        else:
            corr = np.correlate(ts1[lag:], ts2[:len_ts - lag])
        corr_values.append(corr / (np.std(ts1) * np.std(ts2) * (len_ts - abs(lag))))

    correlations = np.array(corr_values).flatten()
    max_corr_value = correlations[np.argmax(correlations)]
    return max_corr_value
