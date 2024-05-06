from spikes import *
from custom_network import *
from CCG import *
from motifs import *
from utils import *


def draw_matrices(func_matrix, conn_matrix):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax1 = axs[0].matshow(func_matrix, cmap='viridis')
    fig.colorbar(ax1, ax=axs[0])
    axs[0].set_title('Functional')

    ax2 = axs[1].matshow(conn_matrix, cmap='viridis')
    fig.colorbar(ax2, ax=axs[1])
    axs[1].set_title('Connectivity')

    plt.show()


def draw_matrix(conn_matrix):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    ax2 = axs.matshow(conn_matrix, cmap='viridis')
    fig.colorbar(ax2, ax=axs)
    axs.set_title('Connectivity')

    plt.show()


def count_n_neurons(modules):
    number_of_neurons = 0
    for module in modules:
        number_of_neurons += len(module)
    return number_of_neurons


np.random.seed(15)
nest.ResetKernel()
# Read image
image = get_any_image(path='naturalscene.png', draw=True, scaling=0.05)

# Simulate the network
neurons_per_module = 20
simulation_time = 200
betw_mods_weight = 10
multiplier_input = 250
multiplier_weights = 10
significance_level = 2
mode = 'noise'
mode = 'natural scene'

number_of_neurons, spike_recorder, modules = run_network(image, neurons_per_module=neurons_per_module,
                                                         simulation_time=simulation_time,
                                                         multiplier_input=multiplier_input,
                                                         multiplier_weights=multiplier_weights,
                                                         betw_mods_weight=betw_mods_weight,
                                                         mode=mode)

number_of_neurons = count_n_neurons(modules)
# Getting spike trains
binary_spike_trains = get_spike_trains(number_of_neurons, simulation_time,
                                       spike_recorder)

spike_trains = get_spike_trains_sliding_window(binary_spike_trains, window_size=10)
spike_trains = np.array(spike_trains)

func_matrix = get_functional_matrix(spike_trains, simulation_time, significance_level=significance_level)
conn_matrix = get_conn_matrix(modules, number_of_neurons, betw_mods_weight=betw_mods_weight)

draw_matrices(func_matrix, conn_matrix)

save_matrix(func_matrix, 'func_matrix.npy')
