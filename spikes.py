import numpy as np


def get_spike_trains(number_neurons, simulation_time, spike_recorder):
    senders = spike_recorder.get('events', 'senders')
    times = spike_recorder.get('events', 'times')
    num_bins = int(simulation_time)
    binary_spike_trains = np.zeros((number_neurons, num_bins), dtype=int)

    min_senders = min(senders)
    neuron_indices = senders - min_senders
    bin_indices = times.astype(int)
    valid_mask = bin_indices < num_bins
    binary_spike_trains[neuron_indices[valid_mask], bin_indices[valid_mask]] = 1

    # NOT OPTIMIZED code is below
    # for neuron_id, spike_time in zip(senders, times):
    #     neuron_index = neuron_id - min_senders
    #     bin_index = int(spike_time)
    #     if bin_index < num_bins:
    #         binary_spike_trains[neuron_index, bin_index] = 1

    return binary_spike_trains


def get_spike_trains_sliding_window(binary_spike_trains, window_size=20):
    window = np.ones(window_size)
    res = []
    # for each neuron
    for i in range(len(binary_spike_trains)):
        res.append(np.convolve(binary_spike_trains[i], window, 'valid'))

    return res

