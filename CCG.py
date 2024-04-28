import numpy as np
from tqdm import tqdm

# TODO
# def cross_correlation(data1, data2, lag_limit=10):
#     ts1 = np.asarray(data1)
#     ts2 = np.asarray(data2)
#     len_ts = min(len(ts1), len(ts2))
#
#     ts1 = ts1 - np.mean(ts1)
#     ts2 = ts2 - np.mean(ts2)
#
#     corr_values = []
#     lags = range(-lag_limit, lag_limit + 1)
#     for lag in lags:
#         if lag < 0:
#             try:
#                 corr = np.correlate(ts1[:lag], ts2[-lag:])
#             except ValueError as e:
#                 print(e)
#         else:
#             corr = np.correlate(ts1[lag:], ts2[:len_ts - lag])
#         corr_values.append(corr / (np.std(ts1) * np.std(ts2) * (len_ts - abs(lag))))
#
#     correlations = np.array(corr_values).flatten()
#     max_corr_value = correlations[np.argmax(correlations)]
#     return max_corr_value
#
#
# def cross_correlation_matrix(spike_trains, lag_limit=10):
#     ccg_mat = np.zeros((len(spike_trains), len(spike_trains)))
#     for i in tqdm(range(len(spike_trains))):
#         for j in range(i+1, len(spike_trains)):
#
#             ccg_mat[i, j] = cross_correlation(spike_trains[i], spike_trains[j], lag_limit=lag_limit)
#             ccg_mat[j, i] = ccg_mat[i, j]
#
#     return ccg_mat


def cross_correlation(data1, data2, lag_limit=10):
    ts1 = np.asarray(data1)
    ts2 = np.asarray(data2)
    ts1 = ts1 - np.mean(ts1)
    ts2 = ts2 - np.mean(ts2)

    # Compute full correlation and then extract the valid range
    full_corr = np.correlate(ts1, ts2, mode='full')
    valid_corr = full_corr[len(ts1) - 1 - lag_limit:len(ts1) + lag_limit]

    # Normalize
    norm = np.std(ts1) * np.std(ts2) * len(ts1)
    normalized_corr = valid_corr / norm

    max_corr_value = np.max(np.abs(normalized_corr))
    return max_corr_value


def cross_correlation_matrix(spike_trains, lag_limit=10):
    n = len(spike_trains)
    ccg_mat = np.zeros((n, n))
    print('CCG is working')
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            ccg_mat[i, j] = cross_correlation(spike_trains[i], spike_trains[j], lag_limit=lag_limit)
            ccg_mat[j, i] = ccg_mat[i, j]
    return ccg_mat
