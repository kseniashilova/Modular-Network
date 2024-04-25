from spikes import *
from custom_network import *
from CCG import *
from motifs import *
from utils import *

# Read image
image = get_any_image(path='naturalscene.png', draw=False, scaling=0.05)

# Simulate the network
simulation_time = 200
number_of_neurons, spike_recorder = run_network(image, neurons_per_module=50,
                                                simulation_time=simulation_time)

# Getting spike trains
binary_spike_trains = get_spike_trains(number_of_neurons, simulation_time,
                                       spike_recorder)
print('Shape: ', binary_spike_trains.shape)
spike_trains = get_spike_trains_sliding_window(binary_spike_trains, window_size=20)

# Get functional matrix
func_mat = cross_correlation_matrix(spike_trains, lag_limit=10)
save_matrix(func_mat, 'func_matrix.npy')