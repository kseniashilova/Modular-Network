import itertools
import networkx as nx
import numpy as np
from collections import Counter
from multiprocessing import Pool
import time

from matplotlib import pyplot as plt
from tqdm import tqdm


def draw_motif(arr, title, name):
    print(arr)
    G = nx.from_numpy_array(arr, create_using=nx.DiGraph)
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(G)  # Generate a layout for nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700,
            edge_color='k', linewidths=1, font_size=15,
            arrows=True, arrowsize=20)
    plt.title(title, size=15)
    plt.savefig(name, bbox_inches='tight')


def draw_motif_sign(arr, title, name):
    print(arr)
    G = nx.from_numpy_array(arr, create_using=nx.DiGraph)
    plt.figure(figsize=(4, 3))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)

    edges = G.edges(data=True)
    positive_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
    negative_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]

    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='green',
                           arrows=True, arrowsize=20, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='red',
                           arrows=True, arrowsize=20, width=2, style='dashed')

    # nx.draw_networkx_labels(G, pos, font_size=15)

    plt.title(title, size=15)
    plt.savefig(name, bbox_inches='tight')
    plt.close()


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


def signed_pair_preserving_shuffle(G):
    G_new = nx.DiGraph()
    G_new.add_nodes_from(G.nodes(data=True))
    pos_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    neg_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    def shuffle_edges(edges):
        # Get the out-degree for each node based on edge list
        from collections import defaultdict
        out_degree = defaultdict(list)
        for u, v, w in edges:
            out_degree[u].append((v, w))

        # a new edge list with shuffled targets
        available_targets = [(v, w) for _, v, w in edges]  # list of all targets
        np.random.shuffle(available_targets)

        new_edges = []
        target_idx = 0
        for u in out_degree:
            for _ in range(len(out_degree[u])):
                v, w = available_targets[target_idx]
                new_edges.append((u, v, w))
                target_idx += 1
        return new_edges

    new_pos_edges = shuffle_edges(pos_edges)
    new_neg_edges = shuffle_edges(neg_edges)

    for u, v, w in new_pos_edges:
        G_new.add_edge(u, v, weight=w)
    for u, v, w in new_neg_edges:
        G_new.add_edge(u, v, weight=w)

    return G_new


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


def create_graph(edges):
    """ Helper function to create a graph based on the list of edges """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def motif_hash_dictionary():
    """ Construct specific motifs manually and return their hashes """
    motifs = [
        [(1, 2), (2, 3)],  # 1 -> 2 -> 3
        [(1, 2), (2, 3), (3, 1)],  # 1 -> 2 -> 3 -> 1
        [(1, 2), (2, 1), (2, 3), (3, 2)],  # 1 <-> 2 <-> 3
        [(1, 2), (1, 3)]  # 1 -> 2, 1 -> 3
    ]
    motif_hashes = {}
    for name, edges in motifs:
        G = create_graph(edges)
        motif_hashes[name] = nx.weisfeiler_lehman_graph_hash(G, iterations=2)
    return motif_hashes


def find_motifs_in_graph(G, motif_hashes):
    """ Find and count occurrences of predefined motifs in a given graph G """
    count = Counter()
    for triplet in itertools.permutations(G.nodes(), 3):
        subgraph = G.subgraph(triplet)
        subgraph_hash = nx.weisfeiler_lehman_graph_hash(subgraph, iterations=2)
        for motif_name, hash_val in motif_hashes.items():
            if subgraph_hash == hash_val:
                count[motif_name] += 1
    return count


def hash_of_2_edges(subgraph):
    pass


def parallel_enumerate_motifs(args):
    G, triplet = args
    subgraph = G.subgraph(triplet)
    motif_hash = nx.weisfeiler_lehman_graph_hash(subgraph)
    # print(subgraph.edges(data=True))
    return list(subgraph.edges(data=True)), motif_hash, nx.to_numpy_array(subgraph)


def parallel_enumerate_motifs_edges(args):
    edges = args
    edges_dict = {}
    for edge in edges:
        min_edge = edge[0]
        max_edge = edge[1]
        weight = edge[2]['weight']
        if min_edge > max_edge:
            tmp = min_edge
            min_edge = max_edge
            max_edge = tmp
        key_str = str(min_edge) + "," + str(max_edge)
        if key_str in edges_dict:
            edges_dict[key_str].append(weight)
        else:
            edges_dict[key_str] = [weight]
    sums = []

    for key in edges_dict:
        sums.append((np.sum(edges_dict[key])))

    sums = np.sort(sums)
    res_str = "".join(map(str, sums))
    return res_str



def enumerate_motifs_parallel(G, hash_to_motif):
    pool = Pool()
    triplets = list(itertools.combinations(G.nodes(), 3))
    results = pool.map(parallel_enumerate_motifs, [(G, triplet) for triplet in triplets])
    pool.close()
    pool.join()
    pool = Pool()
    subgraph_edges = [result[0] for result in results]

    results_edges = pool.map(parallel_enumerate_motifs_edges, subgraph_edges)
    pool.close()
    pool.join()
    motifs_counter = Counter()

    i_res = 0
    for _, motif_hash, arr in tqdm(results):
        add_hash = results_edges[i_res]
        motif_hash += add_hash
        motifs_counter[motif_hash] += 1
        if motif_hash not in hash_to_motif:
            hash_to_motif[motif_hash] = arr
        i_res += 1
    return motifs_counter, hash_to_motif


def save_motifs(matrix, path):
    matrix = np.nan_to_num(matrix, nan=0)
    matrix = np.sign(matrix)

    print('save_motifs is running')
    z_scores_list = []
    hash_to_motif = {}

    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    if G.size() >= 3:
        print('enumerate_motifs for G is running')
        start_time = time.time()
        original_motifs, hash_to_motif = enumerate_motifs_parallel(G, hash_to_motif)
        end_time = time.time()
        print("Matrix size: ", len(G.nodes()))
        print("Time spent - the original graph: {:.2f} seconds".format(end_time - start_time))

        # original_motifs, hash_to_motif = enumerate_motifs(G, hash_to_motif)

        # generate reference networks and count motifs
        num_references = 5  # number of reference networks
        reference_motifs_list = []

        print('enumerate_motifs for G_ref is running')
        for _ in tqdm(range(num_references)):
            # G_ref = generate_reference_network(G)
            G_ref = signed_pair_preserving_shuffle(G)
            start_time = time.time()
            motifs_ref, hash_to_motif = enumerate_motifs_parallel(G_ref, hash_to_motif)
            end_time = time.time()
            print("Matrix size: ", len(G_ref.nodes()))
            print("Time spent - ref random graph: {:.2f} seconds".format(end_time - start_time))
            # motifs_ref, hash_to_motif = enumerate_motifs(G_ref, hash_to_motif)
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
        draw_motif_sign(hash_to_motif[hash], "z score = " + str(score), path + '/iaf_' + hash + '.png')
