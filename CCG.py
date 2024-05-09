import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore


# # old
# def cross_correlation(data1, data2, lag_limit=10):
#     ts1 = np.asarray(data1)
#     ts2 = np.asarray(data2)
#     ts1 = ts1 - np.mean(ts1)
#     ts2 = ts2 - np.mean(ts2)
#
#     # Compute full correlation and then extract the valid range
#     full_corr = np.correlate(ts1, ts2, mode='full')
#     valid_corr = full_corr[len(ts1) - 1 - lag_limit:len(ts1) + lag_limit]
#
#     # Normalize
#     norm = np.std(ts1) * np.std(ts2) * len(ts1)
#     normalized_corr = valid_corr / norm
#
#     max_corr_value = np.max(np.abs(normalized_corr))
#     return max_corr_value
#
#
# # old
# def cross_correlation_matrix(spike_trains, lag_limit=10):
#     n = len(spike_trains)
#     ccg_mat = np.zeros((n, n))
#     print('CCG is working')
#     for i in tqdm(range(n)):
#         for j in range(i + 1, n):
#             ccg_mat[i, j] = cross_correlation(spike_trains[i], spike_trains[j], lag_limit=lag_limit)
#             ccg_mat[j, i] = ccg_mat[i, j]
#     return ccg_mat


def find_significant_peaks(ccg, lags, significance_level=4):
    mean = np.mean(ccg)
    std_dev = np.std(ccg)
    # z_scores = (ccg - mean) / std_dev
    significant_lags = []

    for ind in range(len(ccg)):
        if ccg[ind] > mean + std_dev * significance_level:
            significant_lags.append((ind, 1))  # 1 for excitatory
        elif ccg[ind] < mean - std_dev * significance_level:
            significant_lags.append((ind, -1))  # -1 for inhibitory
    sign_ccg = {}
    for significant_lag in significant_lags:
        if significant_lag[1] == 1:
            sign_ccg[lags[significant_lag[0]]] = ccg[significant_lag[0]]
        elif significant_lag[1] == -1:
            sign_ccg[lags[significant_lag[0]]] = -1 * ccg[significant_lag[0]]

    return sign_ccg


def CCG_tau(tau, N, x_A, x_B, lambda_A, lambda_B):
    theta = N - tau  # corrects for the overlap time bins
    numerator = sum([x_A[t] * x_B[t + tau] for t in range(N - tau)])
    denominator = theta * np.sqrt(lambda_A * lambda_B)
    return numerator / denominator


def CCG(M, N, x_A_arr, x_B_arr, t_overall, significance_level=4, trials_jitter=10):
    original_ccg = compute_CCG(M, N, x_A_arr, x_B_arr, t_overall, significance_level)
    expected_jittered_ccg = compute_expected_jittered_ccg(
        M, N, x_A_arr, x_B_arr, t_overall, significance_level, trials_jitter)

    jitter_corrected_ccg = {key: original_ccg[key] - expected_jittered_ccg[key] for key in original_ccg}

    significant_ccg = find_significant_peaks(list(jitter_corrected_ccg.values()),
                                             list(jitter_corrected_ccg.keys()),
                                             significance_level=significance_level)
    return significant_ccg


def jitter_spikes(M, spike_train, jitter_probability=0.3):
    jittered_spike_trains = []
    for i_M in range(M):
        n_bins = len(spike_train[i_M])
        jittered_spike_train = np.copy(spike_train[i_M])

        # Iterate through each bin in the spike train
        for i in range(n_bins):
            if np.random.rand() < jitter_probability:
                if i > 0 and jittered_spike_train[i] > 0:  # if not the first bin and spikes are present
                    spikes_to_move = np.random.randint(0, jittered_spike_train[i] + 1)
                    jittered_spike_train[i] -= spikes_to_move
                    jittered_spike_train[i - 1] += spikes_to_move
                if i < n_bins - 1 and jittered_spike_train[i] > 0:  # if not the last bin and spikes are present
                    spikes_to_move = np.random.randint(0, jittered_spike_train[i] + 1)
                    jittered_spike_train[i] -= spikes_to_move
                    jittered_spike_train[i + 1] += spikes_to_move
        jittered_spike_trains.append(jittered_spike_train)

    return jittered_spike_trains


def compute_expected_jittered_ccg(M, N, x_A_arr, x_B_arr, t_overall, significance_level,
                                  trials):
    jittered_ccgs = []
    for _ in range(trials):
        jittered_spikes_A = jitter_spikes(M, x_A_arr)
        jittered_spikes_B = jitter_spikes(M, x_B_arr)
        jittered_ccg = compute_CCG(M, N, jittered_spikes_A, jittered_spikes_B,
                                   t_overall, significance_level)
        jittered_ccgs.append(jittered_ccg)

    keys = jittered_ccgs[0].keys()
    data = np.array([[d[key] for key in keys] for d in jittered_ccgs])
    mean_values = np.mean(data, axis=0)
    mean_ccg = dict(zip(keys, mean_values))

    return mean_ccg


def compute_CCG(M, N, x_A_arr, x_B_arr, t_overall, significance_level=4):
    if len(x_A_arr) == 0 or len(x_B_arr) == 0:
        return None
    """
    Calculate CCG array (see STIMULUS-DEPENDENT FUNCTIONAL NETWORK TOPOLOGY IN MOUSE VISUAL CORTEX : Methods)
    :param M: number of trials
    :param N: number of bins in the trains
    :param x_A_arr: A neuron spike trains (for M trials)
    :param x_B_arr: B neuron spike train (for M trials)
    :param t_overall: simulation time to compute mean firing rates
    :return:
    """
    taus = range(0, 50)
    CCG_lags = {}
    for tau in taus:
        CCG_lags[tau] = 0

    for i in range(M):
        lambda_A = x_A_arr[i].sum() / t_overall  # mean fr
        lambda_B = x_B_arr[i].sum() / t_overall  # mean fr

        for tau in range(0, 50):
            CCG_value = CCG_tau(tau, N, x_A_arr[i], x_B_arr[i], lambda_A, lambda_B)
            CCG_lags[tau] += CCG_value

    # average by M trials
    for lag in CCG_lags:
        CCG_lags[lag] /= M

    return CCG_lags


def get_min_key_value(ccg_dict):
    if all(np.isnan(value) for value in ccg_dict.values()):
        return 0
    # min_key = min(ccg_dict.keys())
    # value = ccg_dict[min_key]
    # return value

    key_with_max_abs = max(ccg_dict, key=lambda k: abs(ccg_dict[k]))
    max_abs_value = ccg_dict[key_with_max_abs]
    return max_abs_value


def get_functional_matrix(spike_trains, simulation_time, significance_level=4, trials_jitter=10):
    N = spike_trains.shape[1]
    N_neurons = spike_trains.shape[0]
    func_mat = np.full((N_neurons, N_neurons), 0.0, dtype=np.float64)
    for i in tqdm(range(N_neurons)):
        for j in range(N_neurons):
            if i != j:
                ccg_dict = CCG(M=1, N=N, x_A_arr=[spike_trains[i]], x_B_arr=[spike_trains[j]],
                               t_overall=simulation_time, significance_level=significance_level,
                               trials_jitter=trials_jitter)
                if len(ccg_dict) > 0:
                    func_mat[i][j] = float(get_min_key_value(ccg_dict))

    min_val = np.min(func_mat)
    max_val = np.max(func_mat)
    # func_mat = 2*(func_mat - min_val) / (max_val - min_val) - 1
    return func_mat
