import numpy as np
from scipy.special import psi
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
from scipy.stats import norm


def _dirichlet_exp_E_log_prior(prior):
    exp_E_log_prior = psi(prior) - psi(np.sum(prior)) if len(prior.shape) == 1 \
            else psi(prior) - psi(np.sum(prior, axis=1))[:, np.newaxis]

    return np.exp(exp_E_log_prior)


def _exp_E_log_sim(theta,
                   thetas_neigh,
                   d_neigh,
                   radius, 
                   sd=np.reciprocal(np.sqrt(2 * np.pi))):
    thetas_neigh[0] = theta
    
    thetas_neigh /= np.linalg.norm(thetas_neigh, axis=1, keepdims=True)
    
    cos = np.sum(theta * thetas_neigh, axis=1) / np.linalg.norm(theta)
    spatial = norm.pdf(d_neigh, loc=0, scale=sd)
        
    cos_times_spatial = cos * spatial

    exp_E_log_sim = np.mean(thetas_neigh.T * cos_times_spatial, axis=1)
    exp_E_log_sim /= np.linalg.norm(exp_E_log_sim)

    return 1 + exp_E_log_sim


def _spatial_e_step(X, 
                    alpha,
                    exp_E_log_beta,
                    max_iter_d,
                    Gamma, 
                    Neigh,
                    Dist,
                    radius, 
                    conv_threshold=0.0001):
    modalities = list(X.keys())
    # number of documents
    D = X[modalities[0]].shape[0] 
    # number of modalities
    M = len(X)
    # number of features 
    N = dict()
    for m in modalities:
        N[m] = X[m].shape[1]
    # number of topicspr
    K = exp_E_log_beta[modalities[0]].shape[0]
    # Initialize sufficient statistics
    new_lambda = dict()
    for m in modalities:
        new_lambda[m] = np.zeros((K, N[m]))
    # Initialize normalizers
    phi_norm = dict()
    for m in modalities:
        phi_norm[m] = np.zeros(N[m])
    # Initialize topic proportions
    gamma = np.ones((D, K))
    sim = np.zeros((D, K))

    # Find gamma for each document d
    for d in range(D):
        gamma_d = np.ones(K)
        exp_E_log_theta_d = _dirichlet_exp_E_log_prior(gamma_d)

        idx_d = dict()
        counts_d = dict()
        exp_E_log_beta_d = dict()
        for m in modalities:
            idx_d[m] = X[m][d].nonzero()[1]
            counts_d[m] = X[m][d][:, idx_d[m]].toarray().flatten()
            exp_E_log_beta_d[m] = exp_E_log_beta[m][:, idx_d[m]]

        # Find topic proportions gamma
        for _ in range(max_iter_d):
            prev_gamma = gamma_d

            for m in modalities:
                phi_norm[m] = np.dot(exp_E_log_theta_d, exp_E_log_beta_d[m]) + 1e-100
            # Gamma update
            gamma_d = np.ones(K) * alpha
            for m in modalities:
                gamma_d += exp_E_log_theta_d * np.dot(counts_d[m] / phi_norm[m], exp_E_log_beta_d[m].T)

            exp_E_log_theta_d = _dirichlet_exp_E_log_prior(gamma_d)
            # Check convergence
            meanchange = np.mean(abs(gamma_d - prev_gamma))
            if (meanchange < conv_threshold):
                break

        gamma_d_sum = np.sum(gamma_d)

        exp_E_log_sim = _exp_E_log_sim(gamma_d, Gamma[Neigh[d]], Dist[d], radius)
        
        gamma_d *= exp_E_log_sim
        gamma_d *= gamma_d_sum / np.sum(gamma_d)

        exp_E_log_theta_d = _dirichlet_exp_E_log_prior(gamma_d)

        # Save converged topic proportions
        gamma[d, :] = gamma_d
        
        sim[d, :] = exp_E_log_sim

        for m in modalities:
            phi_norm[m] = np.dot(exp_E_log_theta_d, exp_E_log_beta_d[m]) + 1e-100
            new_lambda[m][:, idx_d[m]] += np.outer(exp_E_log_theta_d, counts_d[m] / phi_norm[m])

    return gamma, new_lambda, sim


class sMTM():
    """
    Spatial Multimodal Topic Model (sMTM) for single-cell spatial data analysis.

    This class implements a Spatial Multimodal Topic Model (sMTM) designed for analyzing 
    single-cell spatial data across multiple modalities. The model captures spatial 
    relationships by constructing a spatial neighborhood graph and uses Variational Inference (VI) 
    to identify spatially-aware topics. These topics represent patterns across features and modalities 
    while incorporating spatial information.

    :param mdata: 
        A `MuData` object containing multimodal single-cell spatial data, including spatial 
        coordinates in the `obsm` attribute.
    :type mdata: muon.MuData
    :param n_topics: 
        Number of topics to infer. Each topic represents a distinct spatial pattern across 
        features and modalities. Default is 20.
    :type n_topics: int, optional
    :param radius: 
        Radius for constructing a spatial neighborhood graph. Used if `n_neighbors` is None. Default is 0.05.
    :type radius: float, optional
    :param n_neighbors: 
        Number of neighbors to consider when constructing the spatial neighborhood graph. 
        Overrides `radius` if set. Default is None.
    :type n_neighbors: int, optional
    :param cache_similarities: 
        If True, caches spatial similarity information for each update during Variational Inference. 
        Default is False.
    :type cache_similarities: bool, optional
    :param seed: 
        Random seed for reproducibility. Ensures consistent initialization and results. Default is 2291.
    :type seed: int, optional
    :param spatial_key: 
        Key in the `obsm` attribute of `MuData` specifying spatial coordinates. Default is 'coords'.
    :type spatial_key: str, optional
    :param n_jobs: 
        Number of CPU cores to use for parallel computation. If set to -1, all available cores are used. Default is 10.
    :type n_jobs: int, optional

    :ivar n_topics: Number of topics initialized in the model.
    :vartype n_topics: int
    :ivar radius: Radius used for spatial neighborhood graph construction.
    :vartype radius: float
    :ivar seed: Random seed used for initializing the model.
    :vartype seed: int
    :ivar rng: Random number generator initialized with the seed.
    :vartype rng: numpy.random.Generator
    :ivar n_jobs: Number of parallel jobs used for computation.
    :vartype n_jobs: int
    :ivar spatial_key: Key for accessing spatial coordinates in `MuData`.
    :vartype spatial_key: str
    :ivar modalities: List of modalities in the dataset.
    :vartype modalities: list
    :ivar features: Dictionary of feature names for each modality.
    :vartype features: dict
    :ivar barcodes: List of barcodes corresponding to the samples.
    :vartype barcodes: list
    :ivar D: Number of samples (observations) in the dataset.
    :vartype D: int
    :ivar M: Number of modalities in the dataset.
    :vartype M: int
    :ivar N: Dictionary with the number of features per modality.
    :vartype N: dict
    :ivar coords: Spatial coordinates of the samples.
    :vartype coords: numpy.ndarray
    :ivar coords_scaled: Scaled spatial coordinates normalized to [0, 1].
    :vartype coords_scaled: numpy.ndarray
    :ivar neighborhood_dist: Distances between each sample and its neighbors.
    :vartype neighborhood_dist: numpy.ndarray
    :ivar neighborhood_graph: Indices of neighbors for each sample.
    :vartype neighborhood_graph: numpy.ndarray
    :ivar gamma: Variational parameters for topic proportions.
    :vartype gamma: numpy.ndarray
    :ivar lambda_: Variational parameters for topics across modalities.
    :vartype lambda_: dict
    :ivar exp_E_log_beta: Expected log topic distributions.
    :vartype exp_E_log_beta: dict
    :ivar similarities: Cached spatial similarity information, if enabled.
    :vartype similarities: dict

    :methods:
        .. method:: VI(n_iter=20, max_iter_d=100)
            Perform Variational Inference (VI) to fit the model to the data.

            :param n_iter: Number of iterations for the VI algorithm. Default is 20.
            :type n_iter: int, optional
            :param max_iter_d: Maximum number of iterations for the E-step in each VI update. Default is 100.
            :type max_iter_d: int, optional
        
            :returns: None
            :rtype: None
        
            :example:
                .. code-block:: python
                
                    model = mtopic.tl.spatialMTM(mdata, n_topics=20, radius=0.05)
                    model.VI(n_iter=20)

    :example:

        .. code-block:: python

            import mtopic

            # Load spatial multimodal single-cell data
            mdata = mtopic.read.h5mu("path/to/file.h5mu")
        
            # Initialize and train the model
            model = mtopic.tl.spatialMTM(mdata, n_topics=20, radius=0.05)
            model.VI(n_iter=20)
    """
    
    def __init__(self,
                 mdata,
                 n_topics=20,
                 radius=0.05, 
                 n_neighbors=None, 
                 cache_similarities=False, 
                 seed=2291, 
                 spatial_key='coords',
                 n_jobs=10):
        
        self.n_topics = n_topics
        self.radius = radius
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.n_jobs = n_jobs

        if self.n_jobs > cpu_count():
            self.n_jobs = cpu_count()
        
        self.cache_similarities = cache_similarities
        self.spatial_key = spatial_key
        self.n_neighbors = None

        self._load_data(mdata)
        self._build_neighborhood_graph()
        self._init_params()

    def _build_neighborhood_graph(self):

        if self.n_neighbors is not None:
            neigh = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.coords_scaled)
            distances, indices = neigh.kneighbors(self.coords_scaled)

        else:
            neigh = NearestNeighbors(radius=self.radius).fit(self.coords_scaled)
            distances, indices = neigh.radius_neighbors(self.coords_scaled, sort_results=True)
      
        self.neighborhood_dist = distances
        self.neighborhood_graph = indices

    def _load_data(self,
                   X):
        self.X = {mod: X[mod].X for mod in X.mod}
        self.modalities = list(self.X.keys())
        self.features = {mod: np.asarray(X[mod].var_names) for mod in X.mod}
        self.barcodes = X.obs.index.tolist()
        self.D = self.X[self.modalities[0]].shape[0]
        self.M = len(self.X)
        self.N = dict()
        for mod in self.modalities:
            self.N[mod] = self.X[mod].shape[1]

        self.coords = X.obsm[self.spatial_key]
        self.coords_scaled = (self.coords - np.min(self.coords, axis=0)) / np.max(np.max(self.coords, axis=0) - np.min(self.coords, axis=0))
    
    def _init_params(self):
        # Topic prior
        self.eta = 0.01
        # Topic proportions prior
        self.alpha = 1 / self.n_topics
        # Initialize variational topic proportions
        self.gamma = np.ones((self.D, self.n_topics)) * self.alpha
        # Initiate variational topics
        self.lambda_ = dict()
        self.exp_E_log_beta = dict()

        for m in self.modalities:
            self.lambda_[m] = self.rng.gamma(100., 1./100., (self.n_topics, self.N[m]))
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior(self.lambda_[m])

        if self.cache_similarities:
            self.similarities = dict()

    def _set_batch_n_jobs(self, 
                          batch):
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
            
        D = len(list(batch))
        
        if D < self.n_jobs:
            self.n_jobs = D
    
    def _VI_update(self,
                   batch):
        self._set_batch_n_jobs(batch)
        batch_split = np.array_split(batch, self.n_jobs)

        output = Parallel(n_jobs=self.n_jobs)(
            delayed(_spatial_e_step)(
                X = {m: self.X[m][batch_job] for m in self.modalities}, 
                alpha = self.alpha,
                exp_E_log_beta = self.exp_E_log_beta,
                max_iter_d = self.max_iter_d,
                Gamma = self.gamma, 
                Neigh = self.neighborhood_graph[batch_job], 
                Dist = self.neighborhood_dist[batch_job],
                radius = self.radius) for batch_job in batch_split)
        
        gamma_list, new_lambda_list, sim_list = zip(*output)

        self.gamma = np.vstack(gamma_list)
        if self.cache_similarities:
            self.similarities[self.n_update] = np.vstack(sim_list)
        
        new_lambda = dict()
        for m in self.modalities:
            new_lambda[m] = np.zeros(self.lambda_[m].shape)
            for lambda_update in new_lambda_list:
                new_lambda[m] += lambda_update[m]
            new_lambda[m] = self.eta + new_lambda[m] * self.exp_E_log_beta[m]

            self.lambda_[m] = new_lambda[m]
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior(self.lambda_[m])

    def VI(self, 
           n_iter = 20, 
           max_iter_d = 100):

        self.n_iter = n_iter
        self.max_iter_d = max_iter_d

        batch = [i for i in range(self.D)]

        self.n_update = 1
        for _ in tqdm(range(self.n_iter)):
            self._VI_update(batch)
            self.n_update += 1
