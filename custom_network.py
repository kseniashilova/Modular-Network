import h5py
import nest
import networkx as nx
import matplotlib.pyplot as plt
from images_work import *


def plot_network_structure():
    # Extract connections
    conns = nest.GetConnections()

    sources = nest.GetStatus(conns, 'source')
    targets = nest.GetStatus(conns, 'target')

    # Create graph
    edges = list(zip(sources, targets))

    G = nx.DiGraph(edges=edges)
    # G.add_edges_from(edges)

    # Draw network
    nx.draw(G)
    plt.show()


def plot_spike_data(spike_recorder, num_steps=None):
    spike_data = nest.GetStatus(spike_recorder, 'events')[0]
    senders = spike_data['senders']
    times = spike_data['times']
    print(senders.shape, times.shape)
    if num_steps is not None:
        mask = times < num_steps
        times = times[mask]
        senders = senders[mask]
    print(senders.shape, times.shape)
    # Plotting the spike data
    plt.figure(figsize=(10, 5))
    plt.plot(times, senders, '.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.title('Spike Raster Plot')
    plt.show()


def add_input(image, module, multiplier=1):
    flat_image = image.flatten()
    dc_generators = []
    i = 0
    for pixel_value in flat_image:
        if i >= len(module):
            break
        dc_gen = nest.Create('dc_generator', 1)
        nest.SetStatus(dc_gen, {'amplitude': pixel_value * multiplier})
        dc_generators.append(dc_gen)
        nest.Connect(dc_gen, module[i])
        i += 1

    # dc_gen = nest.Create('dc_generator')
    # nest.SetStatus(dc_gen, {'amplitude': 800.0})
    # nest.Connect(dc_gen, module)


# old
def create_structure(num_modules=3, neurons_per_module=1):
    modules = []
    for i in range(num_modules):
        module = nest.Create('iaf_psc_alpha', neurons_per_module)
        modules.append(module)

    # Connect Neurons within Modules
    syn_spec_intra = {'weight': 2.0, 'delay': 1.0}
    for module in modules:
        nest.Connect(module, module, 'all_to_all', syn_spec=syn_spec_intra)

    modules = connect_neurons_between_modules(modules)
    return modules


def create_module_spatial_random(neurons_per_module=10, mult=20):
    x_positions = np.random.uniform(low=0.0, high=1.0, size=neurons_per_module)
    y_positions = np.random.uniform(low=0.0, high=1.0, size=neurons_per_module)
    random_positions = [[x, y] for x, y in zip(x_positions, y_positions)]

    # Create neurons with random positions
    random_layer = nest.Create('iaf_psc_alpha',
                               positions=nest.spatial.free(random_positions))

    conn_dict = {
        'rule': 'pairwise_bernoulli',
        'p': 1,  # Connection probability
        'mask': {'circular': {'radius': 0.4}}
    }

    # Define synaptic specification separately if needed
    syn_spec = {
        'weight': mult * nest.spatial.distance,  # Using distance to scale the weight
        'delay': 1.5
    }

    # Connect the layers
    nest.Connect(random_layer, random_layer, conn_dict, syn_spec)

    return random_layer


def create_module_spatial_grid(grid_shape=[10, 10], mult=20):
    grid_layer = nest.Create('iaf_psc_alpha',
                             positions=nest.spatial.grid(shape=grid_shape))
    conn_dict = {
        'rule': 'pairwise_bernoulli',
        'p': 1,  # Connection probability
        'mask': {'circular': {'radius': 0.4}}
    }

    # Define synaptic specification separately if needed
    syn_spec = {
        'weight': mult * nest.spatial.distance,  # Using distance to scale the weight
        'delay': 1.5
    }

    # Connect the layers
    nest.Connect(grid_layer, grid_layer, conn_dict, syn_spec)
    return grid_layer


def connect_neurons_between_modules(modules, connection_probability=0.2, weight=1.0, delay=1.5):
    syn_spec_inter = {'weight': weight, 'delay': delay}
    # syn_spec_inter = {'weight': 100*nest.spatial.distance, 'delay': delay}

    for i in range(len(modules)):
        for j in range(len(modules)):
            if i != j:
                nest.Connect(modules[i], modules[j],
                             conn_spec={'rule': 'pairwise_bernoulli', 'p': connection_probability},
                             syn_spec=syn_spec_inter)

    return modules


def create_modules_spatial(num_modules=3, grid_shape=[10, 10], neurons_per_module=10):
    # Connect Neurons between Modules
    modules = []
    layer1 = create_module_spatial_grid(grid_shape=grid_shape)
    modules.append(layer1)

    for i in range(1, num_modules):
        modules.append(create_module_spatial_random(neurons_per_module=neurons_per_module))

    modules = connect_neurons_between_modules(modules, connection_probability=0.3,
                                              weight=45, delay=1.5)
    return modules


def set_spike_rec(modules):
    spike_recorder = nest.Create('spike_recorder')
    # nest.Connect(modules[1], spike_recorder)
    for module in modules:
        nest.Connect(module, spike_recorder)
    return spike_recorder


def run_network(image, neurons_per_module, simulation_time, multiplier):

    modules = create_modules_spatial(num_modules=2, grid_shape=image.shape,
                                     neurons_per_module=neurons_per_module)
    print('Image shape = ', image.shape,
          'Number of neurons = ', image.shape[0] * image.shape[1])

    add_input(image, modules[0], multiplier=multiplier)

    spike_recorder = set_spike_rec(modules)

    nest.Simulate(simulation_time)

    plot_spike_data(spike_recorder, num_steps=None)
    number_of_neurons = (neurons_per_module * (len(modules) - 1) +
                         image.shape[0] * image.shape[1])
    return number_of_neurons, spike_recorder
