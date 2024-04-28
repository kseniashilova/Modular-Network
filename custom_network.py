import h5py
import nest
import networkx as nx
import matplotlib.pyplot as plt
from images_work import *
from nest.lib.hl_api_exceptions import NESTErrors


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


def set_spike_rec(modules):
    spike_recorder = nest.Create('spike_recorder')
    # nest.Connect(modules[1], spike_recorder)
    for module in modules:
        nest.Connect(module, spike_recorder)
    return spike_recorder


def plot_spike_data(spike_recorder, num_steps=None):
    spike_data = nest.GetStatus(spike_recorder, 'events')[0]
    senders = spike_data['senders']
    times = spike_data['times']
    if num_steps is not None:
        mask = times < num_steps
        times = times[mask]
        senders = senders[mask]
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


def valid_parameters(params, V_reset):
    if np.all(params['V_th'] > V_reset):
        return True
    else:
        return False


def create_heterogeneous_neurons(size, positions, std_dev_factor=0.1):
    defaults = nest.GetDefaults('iaf_psc_alpha')

    ready_params = False  # flag to set random params (sometimes there is nest error because of thresholds)
    while not ready_params:
        # new params dict
        new_params = {
            'tau_m': np.random.normal(loc=defaults['tau_m'], scale=defaults['tau_m'] * std_dev_factor, size=size),
            'tau_syn_ex': np.random.normal(loc=defaults['tau_syn_ex'], scale=defaults['tau_syn_ex'] * std_dev_factor,
                                           size=size),
            'tau_syn_in': np.random.normal(loc=defaults['tau_syn_in'], scale=defaults['tau_syn_in'] * std_dev_factor,
                                           size=size),
            't_ref': np.random.normal(loc=defaults['t_ref'], scale=defaults['t_ref'] * std_dev_factor, size=size),
            'V_th': np.random.normal(loc=defaults['V_th'], scale=np.abs(defaults['V_th']) * std_dev_factor, size=size),

        }
        if (valid_parameters(new_params, defaults['V_reset'])):
            neurons = nest.Create('iaf_psc_alpha', positions=positions)
            for param, values in new_params.items():
                nest.SetStatus(neurons, param, values.flatten().tolist())
            ready_params = True
        else:
            ready_params = False




    return neurons


def create_module_spatial(neurons, mult=20):
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
    nest.Connect(neurons, neurons, conn_dict, syn_spec)
    return neurons


def create_module_spatial_grid(heterogeneous=True, grid_shape=[10, 10], mult=20):
    size = grid_shape[0] * grid_shape[1]
    positions = nest.spatial.grid(shape=grid_shape)

    if heterogeneous:
        neurons = create_heterogeneous_neurons(size, positions, std_dev_factor=0.1)
    else:
        neurons = nest.Create('iaf_psc_alpha', positions=positions)

    #nest.SetStatus(neurons, {'positions': positions})
    neurons = create_module_spatial(neurons, mult=mult)
    return neurons




def create_module_spatial_random(heterogeneous=True, size=100, mult=20):
    x_positions = np.random.uniform(low=0.0, high=1.0, size=size)
    y_positions = np.random.uniform(low=0.0, high=1.0, size=size)
    random_positions = [[x, y] for x, y in zip(x_positions, y_positions)]
    random_positions = nest.spatial.free(random_positions)

    if heterogeneous:
        neurons = create_heterogeneous_neurons(size, random_positions, std_dev_factor=0.1)
    else:
        neurons = nest.Create('iaf_psc_alpha', positions=random_positions)


    neurons = create_module_spatial(neurons, mult=mult)
    return neurons


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


def create_modules_spatial(num_modules=3, grid_shape=[10, 10], neurons_per_module=10, mult=20):
    # Connect Neurons between Modules
    modules = []
    layer1 = create_module_spatial_grid(heterogeneous=True, grid_shape=grid_shape, mult=mult)
    modules.append(layer1)

    for i in range(1, num_modules):
        modules.append(create_module_spatial_random(heterogeneous=True, size=neurons_per_module, mult=mult))

    modules = connect_neurons_between_modules(modules, connection_probability=0.3,
                                              weight=45, delay=1.5)
    return modules


def run_network(image, neurons_per_module, simulation_time, multiplier_input, multiplier_weights):
    modules = create_modules_spatial(num_modules=2, grid_shape=image.shape,
                                     neurons_per_module=neurons_per_module, mult=multiplier_weights)
    print('Image shape = ', image.shape,
          'Number of neurons = ', image.shape[0] * image.shape[1])

    add_input(image, modules[0], multiplier=multiplier_input)

    spike_recorder = set_spike_rec(modules)

    nest.Simulate(simulation_time)

    plot_spike_data(spike_recorder, num_steps=None)
    number_of_neurons = (neurons_per_module * (len(modules) - 1) +
                         image.shape[0] * image.shape[1])
    return number_of_neurons, spike_recorder
