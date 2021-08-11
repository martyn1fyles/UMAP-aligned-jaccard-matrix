import umap
import numpy as np
import scipy as sci

n_obs_per_cluster = 100
symptom_names = [f'Symptom {letter}' for letter in 'ABCDEFGH']

# create cluster 1
p1 = np.array([.4, .3, .4, .35, .1, .2, .05, .1])
p2 = np.array([.1, .05, .1, .15, .5, .6, .45, .35])
c1 = np.random.binomial(n = 1, p = p1, size = (n_obs_per_cluster, 8))
c2 = np.random.binomial(n = 1, p = p2, size = (n_obs_per_cluster, 8))
x1 = np.concatenate((c1, c2), axis = 0)

# create cluster 2, which is related to cluster 1
p1 = np.array([.4, .3, .4, .35, .1, .2, .05, .1]) + np.array([0, 0, 0, 0, 0.25, .25, 0.25, 0.25])
c1 = np.random.binomial(n = 1, p = p1, size = (n_obs_per_cluster, 8))
c2 = np.random.binomial(n = 1, p = p2, size = (n_obs_per_cluster, 8))
x2 = np.concatenate((c1, c2), axis = 0)

# compute distance matrices
distance_matrix_x1 = sci.spatial.distance.pdist(X = x1.transpose(), metric = 'jaccard')
distance_matrix_x1 = sci.spatial.distance.squareform(distance_matrix_x1)
distance_matrix_x2 = sci.spatial.distance.pdist(X = x2.transpose(), metric = 'jaccard')
distance_matrix_x2 = sci.spatial.distance.squareform(distance_matrix_x2)

# perform alignments
aligned_mapper = umap.AlignedUMAP(
    n_neighbors=2,
    min_dist = 0.1,
    n_components = 2,
    metric='euclidean')

aligned_mapper.fit([distance_matrix_x1, distance_matrix_x2], relations = [{i:i for i in range(8 - 1)}])

