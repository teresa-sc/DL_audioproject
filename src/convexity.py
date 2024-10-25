import os
import glob
import numpy as np
import torch
import pynndescent
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from typing import Dict, List, Tuple
import itertools
import random

#fix random seed
random.seed(42)

def nn(features, num_neig):
    m,n = features.shape
    dis = np.zeros((m,m))
    index = pynndescent.NNDescent(features, n_neighbors=num_neig+1, metric='euclidean', n_jobs=-1)
    ind = index.neighbor_graph[0]
    distances = index.neighbor_graph[1]
    for i in range(m):
        dis[i,ind[i]] = distances[i]
    dis_sym = np.maximum(dis, dis.T)
    return dis_sym

def get_path(Pr: np.ndarray, i: int, j: int) -> List[int]:
    """
    Get the shortest path from i to j.
    Source: https://stackoverflow.com/a/5307890
    """
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]

def get_concept_idx(labels,
                    ) -> Dict[str, List[int]]:
    """
    Arguments:
        labels          List of all classes
    Returns:
        concepts        Dictionary of ids from columns_names belonging to each concept
    """
    classes = np.unique(labels)

    class_dic = {}
    for idx, name in enumerate(labels):
        for key in classes: 
            if name == key: 
                if key in class_dic.keys():
                    class_dic[key].append(idx)
                else:
                    class_dic[key] = [idx]      
    return class_dic


def is_path_in_concept(shortest_path, indices):
    """
    Compute the proportion of the path that is within the concept.
    Arguments:
        shortest_path:  list of all vertices on the path
        indices:        list of all vertices belonging to the concept
    Returns:
        prop:           the proportion of the path that is inside the concept
    """

    if len(shortest_path) <= 2:
        prop = 1
    else:
        length = 0
        outside = 0
        for idx in shortest_path[1:-1]:
            length += 1
            if idx not in indices:
                outside += 1
        prop = (length - outside)/length
    return prop

def compute_paths(dist_matrix, concept, indices, predecessors):
    """
    dist_matrix     output from djikstra
    concept         name of concept
    indices         indeces of all points beloning to concept
    predecessores   output from dijkstra
    """
    proportion = []
    path_exists = []
    all_paths = list(itertools.permutations(indices, r=2))
    n_paths_max = min(len(all_paths), 5000)
    sampled_indices = np.random.choice(list(range(len(all_paths))), n_paths_max, replace=False)
    sampled_paths = [all_paths[index] for index in sampled_indices]
    for id1, id2 in sampled_paths:
        if dist_matrix[id1, id2] == float('inf'):
            exists = False
            proportion.append(0)
        else:
            shortest_path = get_path(predecessors, id1, id2)
            prop = is_path_in_concept(shortest_path, indices)
            proportion.append(prop)
            exists = True
        path_exists.append(exists)
    #print(f"Concept {concept}: "
    #      f"{'{:.2f}'.format(np.mean(proportion) * 100)}% mean proportion of path in concept")
    return proportion, path_exists

def extract_indices(labels: List[str], min_per_class) -> Dict[str, List[int]]:
    class_counts = {}
    for label in labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    indices = []
    for label, count in class_counts.items():
        if count > min_per_class:
            indices.append([i for i, l in enumerate(labels) if l == label])
    
    return indices

def graph_convexity(features, labels, num_neighbours=10): 
    """
    Arguments:
        features:       3D tensor of shape (n_samples, n_layers, n_features)
        labels:         list of all classes
        num_neighbours: number of neighbours to consider in the graph
    Returns:
        proportion_all: list of tuples (mean, std) of the proportion of the path that is within the concept averaged across classes
        proportion_class_all: dictionary of dictionaries with the proportion of the path that is within the concept
    """
    proportion_all = []
    proportion_class_all = {}
    num_layers = features.shape[1]
    for lay in range(num_layers):
        print(f"Start Layer {lay}")
        prop = []
        proportion_class = {}
        dis_sym = nn(features[:,lay,:], num_neighbours)
        graph = csr_matrix(dis_sym)
        concept_indices = get_concept_idx(labels)
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=True,
                                                    return_predecessors=True)
        for concept, indices in concept_indices.items():
            proportion, path_exists = compute_paths(dist_matrix,concept,indices,predecessors)
            prop.extend(proportion)
            proportion_class[concept] = (np.mean(proportion), np.std(proportion)/ np.sqrt(len(indices)))
        proportion_all.append((np.mean(prop), np.std(prop)/ np.sqrt(len(indices))))
        proportion_class_all[f'Layer {lay}'] = proportion_class
        print(f"Layer {lay}: ", proportion_all[-1])

    return proportion_all, proportion_class_all