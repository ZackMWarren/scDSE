import numpy as np
import graphtools
from scipy import sparse

def compute_diffusion_matrix(
    X: np.array,
    n_pca = 100,
    n_landmark = 2000,
    knn_dist = "euclidean",
    precomputed = None,
    knn = 5,
    knn_max = None,
    decay = 40,
    n_jobs = 1,
    verbose = 1,
    random_state = None,
    thresh=1e-4,):
    '''
    Taken from graphtools. Mimics PHATE's diffusion op
    '''
    n_samples = X.shape[0]

    # graphtools requires n_landmark < n_samples; disable landmarking otherwise.
    if n_landmark is not None and n_landmark >= n_samples:
        n_landmark = None

    # graphtools requires n_pca < min(n_samples, n_features); disable otherwise.
    if n_pca is not None and n_pca >= min(X.shape):
        n_pca = None

    graph_params = {
        "n_pca": n_pca,
        "n_landmark": n_landmark,
        "distance": knn_dist,
        "precomputed": precomputed,
        "knn": knn,
        "knn_max": knn_max,
        "decay": decay,
        "thresh": thresh,
        "n_jobs": n_jobs,
        "verbose": verbose,
        "random_state": random_state,
    }
    graph = graphtools.Graph(X, **graph_params)

    if isinstance(graph, graphtools.graphs.LandmarkGraph):
        diff_op = graph.landmark_op
    else:
        diff_op = graph.diff_aff
    if sparse.issparse(diff_op):
        diff_op = diff_op.toarray()
        
    return diff_op
        