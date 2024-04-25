import itertools
import networkx as nx
import numpy as np
from collections import Counter
from multiprocessing import Pool

from matplotlib import pyplot as plt
from tqdm import tqdm


def draw_motif(arr, title, name):
    G = nx.from_numpy_array(arr, create_using=nx.DiGraph)
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(G)  # Generate a layout for nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700,
            edge_color='k', linewidths=1, font_size=15,
            arrows=True, arrowsize=20)
    plt.title(title, size=15)
    plt.savefig(name, bbox_inches='tight')


def enumerate_motifs(G, hash_to_motif):
    """
    Enumerate all 3-node motifs in the directed graph G.
    Returns a Counter object mapping motif IDs to their counts.
    """
    motifs_counter = Counter()
    for triplet in itertools.combinations(G.nodes(), 3):
        subgraph = G.subgraph(triplet)
        motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph)
        motifs_counter[motif_hash] += 1

        if motif_hash not in hash_to_motif:
            hash_to_motif[motif_hash] = nx.to_numpy_array(subgraph)

    return motifs_counter, hash_to_motif


def generate_reference_network(G):

    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    randomized_graph = nx.directed_configuration_model(in_degrees, out_degrees, create_using=nx.DiGraph())

    randomized_graph = nx.DiGraph(randomized_graph)  # Removes parallel edges
    randomized_graph.remove_edges_from(nx.selfloop_edges(randomized_graph))
    return randomized_graph


def motif_z_scores(original_motifs, reference_motifs):
    """
    Calculate Z-scores for each motif by comparing their counts in the original
    network to an ensemble of reference networks.
    """
    z_scores = {}
    for motif_id in original_motifs:
        original_count = original_motifs[motif_id]
        reference_counts = [motifs.get(motif_id, 0) for motifs in reference_motifs]
        mean_ref = np.mean(reference_counts)
        std_ref = np.std(reference_counts, ddof=1)  # ddof=1 for sample standard deviation
        z_scores[motif_id] = (original_count - mean_ref) / std_ref if std_ref > 0 else float('nan')
    return z_scores



def parallel_enumerate_motifs(args):
    G, triplet = args
    subgraph = G.subgraph(triplet)
    motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph)
    return motif_hash, nx.to_numpy_array(subgraph)


def enumerate_motifs_parallel(G,hash_to_motif):
    pool = Pool()
    triplets = list(itertools.combinations(G.nodes(), 3))
    results = pool.map(parallel_enumerate_motifs, [(G, triplet) for triplet in triplets])
    pool.close()
    pool.join()

    motifs_counter = Counter()

    for motif_hash, arr in tqdm(results):
        motifs_counter[motif_hash] += 1
        if motif_hash not in hash_to_motif:
            hash_to_motif[motif_hash] = arr
    return motifs_counter, hash_to_motif


def save_motifs(matrix):
    print('save_motifs is running')
    z_scores_list = []
    hash_to_motif = {}

    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    if G.size() >= 3:
        print('enumerate_motifs for G is running')
        original_motifs, hash_to_motif = enumerate_motifs_parallel(G, hash_to_motif)

        # generate reference networks and count motifs
        num_references = 5  # number of reference networks
        reference_motifs_list = []

        print('enumerate_motifs for G_ref is running')
        for _ in tqdm(range(num_references)):
            G_ref = generate_reference_network(G)
            motifs_ref, hash_to_motif = enumerate_motifs_parallel(G_ref, hash_to_motif)
            reference_motifs_list.append(motifs_ref)

        z_scores = motif_z_scores(original_motifs, reference_motifs_list)
        z_scores_list.append(z_scores)

    print('select z scores for G is running')
    # select only high z scores
    motif_dict = {}
    motif_dict_num = {}
    for z_scores in z_scores_list:
        for hash, score in z_scores.items():
            if score > 1.6:
                if hash not in motif_dict.keys():
                    motif_dict[hash] = score
                    motif_dict_num[hash] = 1
                else:
                    motif_dict[hash] += score
                    motif_dict_num[hash] += 1
    for hash, score in motif_dict.items():
        motif_dict[hash] = score * 1.0 / motif_dict_num[hash]

    used_hashes = []
    for hash, score in motif_dict.items():
        draw_motif(hash_to_motif[hash], "z score = " + str(score), 'iaf_' + hash + '.png')
