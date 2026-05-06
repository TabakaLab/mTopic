import warnings

import numpy as np
import scipy
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def _dirichlet_exp_E_log_prior_cuda(prior):
    digamma_prior = torch.special.psi(prior)
    if prior.ndim == 1:
        digamma_sum = torch.special.psi(torch.sum(prior))
    elif prior.ndim == 2:
        digamma_sum = torch.special.psi(torch.sum(prior, dim=-1, keepdim=True))
    exp_E_log_prior = torch.exp(digamma_prior - digamma_sum)
    return exp_E_log_prior


@torch.compile(mode="default")
def _compute_phi_fused(exp_E_log_theta_docs, beta_m):
    phi = exp_E_log_theta_docs * beta_m
    phi /= (phi.sum(dim=1, keepdim=True) + 1e-30)
    return phi


def slice_sparse_csr_rows(csr_tensor, row_indices):
    device = csr_tensor.device
    num_cols = csr_tensor.shape[1]
    num_selected_rows = len(row_indices)

    if num_selected_rows == 0:
        return torch.sparse_csr_tensor(
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=csr_tensor.values().dtype, device=device),
            size=(0, num_cols),
            device=device
        )

    original_crow_indices = csr_tensor.crow_indices()
    original_col_indices = csr_tensor.col_indices()
    original_values = csr_tensor.values()

    start_indices = original_crow_indices[row_indices]
    end_indices = original_crow_indices[row_indices + 1]
    lengths = end_indices - start_indices

    new_crow_indices = torch.cat((torch.tensor([0], dtype=torch.int64, device=device), torch.cumsum(lengths, dim=0)))
    total_nnz = new_crow_indices[-1].item()

    if total_nnz == 0:
        col_indices_batch = torch.empty(0, dtype=original_col_indices.dtype, device=device)
        values_batch = torch.empty(0, dtype=original_values.dtype, device=device)
    else:
        range_total_nnz = torch.arange(total_nnz, device=device)
        start_indices_repeated = start_indices.repeat_interleave(lengths)
        range_within_rows = range_total_nnz - new_crow_indices[:-1].repeat_interleave(lengths)
        final_indices = start_indices_repeated + range_within_rows
        col_indices_batch = original_col_indices[final_indices]
        values_batch = original_values[final_indices]

    batch_csr = torch.sparse_csr_tensor(
        new_crow_indices,
        col_indices_batch,
        values_batch,
        size=(num_selected_rows, num_cols),
        device=device
    )
    return batch_csr


def _exp_E_log_sim_cuda(gamma_batch, global_gamma, neigh_idx, distances):
    thetas_neigh = global_gamma[neigh_idx].clone()
    thetas_neigh[:, 0, :] = gamma_batch 
    
    gamma_norm = F.normalize(gamma_batch, p=2, dim=1)
    neigh_norm = F.normalize(thetas_neigh, p=2, dim=2)
    
    cos = torch.bmm(neigh_norm, gamma_norm.unsqueeze(2)).squeeze(2)
    sd = 1.0 / np.sqrt(2 * np.pi)
    
    spatial = torch.exp(-0.5 * (distances / sd)**2)
    spatial[distances > 1e8] = 0.0
    
    cos_times_spatial = cos * spatial
    
    exp_E_log_sim = torch.bmm(neigh_norm.transpose(1, 2), cos_times_spatial.unsqueeze(2)).squeeze(2)
    exp_E_log_sim = F.normalize(exp_E_log_sim + 1e-30, p=2, dim=1)
    
    return 1.0 + exp_E_log_sim


def _spatial_e_step_cuda(X_batch, alpha, exp_E_log_beta, max_iter_d, global_gamma, neigh_idx, distances, conv_threshold=0.0001):
    modalities = list(X_batch.keys())
    device = X_batch[modalities[0]].device
    D_batch = X_batch[modalities[0]].shape[0]
    K = exp_E_log_beta[modalities[0]].shape[0]

    gamma = torch.ones((D_batch, K), device=device, dtype=torch.float32)
    prev_gamma = torch.empty_like(gamma)

    for _ in range(max_iter_d):
        prev_gamma.copy_(gamma)
        exp_E_log_theta = _dirichlet_exp_E_log_prior_cuda(gamma)
        gamma.fill_(alpha)
        
        for m in modalities:
            Xm_batch = X_batch[m]
            if Xm_batch.values().shape[0] > 0:
                crow = Xm_batch.crow_indices()
                docs = torch.repeat_interleave(torch.arange(D_batch, device=device), crow[1:] - crow[:-1])
                words = Xm_batch.col_indices()
                
                phi_normalized = _compute_phi_fused(exp_E_log_theta[docs], exp_E_log_beta[m][:, words].T)
                gamma_update = (Xm_batch.values()[:, None] * phi_normalized)
                gamma.scatter_add_(0, docs[:, None].expand(-1, K), gamma_update)
                
        if torch.mean(torch.abs(gamma - prev_gamma)) < conv_threshold: 
            break

    gamma_batch_sum = torch.sum(gamma, dim=1, keepdim=True)
    exp_E_log_sim = _exp_E_log_sim_cuda(gamma, global_gamma, neigh_idx, distances)
    
    gamma *= exp_E_log_sim
    gamma *= gamma_batch_sum / (torch.sum(gamma, dim=1, keepdim=True) + 1e-30)

    exp_E_log_theta_final = _dirichlet_exp_E_log_prior_cuda(gamma)
    suff_stats_lambda = {m: torch.zeros_like(exp_E_log_beta[m], device=device) for m in modalities}
    
    for m in modalities:
        Xm_batch = X_batch[m]
        if Xm_batch.values().shape[0] > 0:
            crow = Xm_batch.crow_indices()
            docs = torch.repeat_interleave(torch.arange(D_batch, device=device), crow[1:] - crow[:-1])
            words = Xm_batch.col_indices()
            phi_normalized = _compute_phi_fused(exp_E_log_theta_final[docs], exp_E_log_beta[m][:, words].T)
            lambda_update = (Xm_batch.values()[:, None] * phi_normalized)
            suff_stats_lambda[m].scatter_add_(1, words[None, :].expand(K, -1), lambda_update.T)

    return gamma, suff_stats_lambda


class sMTM_GPU:
    """
    GPU-accelerated Spatial Multimodal Topic Model.

    This class implements a CUDA-accelerated version of the Spatial Multimodal Topic Model 
    (sMTM) for analyzing single-cell spatial data across multiple modalities. The model 
    captures spatial relationships by constructing a spatial neighborhood graph and uses 
    Variational Inference (VI) to identify spatially-aware topics. The model is mathematically 
    equivalent to :class:`sMTM` but executes E-step and M-step updates on the GPU using 
    sparse CSR tensors and batched neighborhood computations for substantial speedups on 
    large datasets.

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
    :param seed: 
        Random seed for reproducibility. Ensures consistent initialization and results. Default is 2291.
    :type seed: int, optional
    :param spatial_key: 
        Key in the `obsm` attribute of `MuData` specifying spatial coordinates. Default is 'coords'.
    :type spatial_key: str, optional
    :param verbose: 
        If True, displays a progress bar during training. Default is True.
    :type verbose: bool, optional

    :ivar n_topics: Number of topics initialized in the model.
    :vartype n_topics: int
    :ivar radius: Radius used for spatial neighborhood graph construction.
    :vartype radius: float
    :ivar seed: Random seed used for initializing the model.
    :vartype seed: int
    :ivar rng: Random number generator initialized with the seed.
    :vartype rng: numpy.random.Generator
    :ivar device: Compute device used by the model (always ``"cuda"``).
    :vartype device: str
    :ivar spatial_key: Key for accessing spatial coordinates in `MuData`.
    :vartype spatial_key: str
    :ivar modalities: List of modalities in the dataset.
    :vartype modalities: list
    :ivar features: Dictionary of feature names for each modality.
    :vartype features: dict
    :ivar barcodes: List of barcodes corresponding to the samples.
    :vartype barcodes: list
    :ivar n_obs: Number of samples (observations) in the dataset.
    :vartype n_obs: int
    :ivar n_var: Dictionary with the number of features per modality.
    :vartype n_var: dict
    :ivar coords_scaled: Spatial coordinates normalized to [0, 1] (numpy).
    :vartype coords_scaled: numpy.ndarray
    :ivar coords: Scaled spatial coordinates as a CUDA tensor.
    :vartype coords: torch.Tensor
    :ivar neighborhood_dist: Distances between each sample and its neighbors (numpy, padded for radius mode).
    :vartype neighborhood_dist: numpy.ndarray
    :ivar neighborhood_graph: Indices of neighbors for each sample (numpy, padded for radius mode).
    :vartype neighborhood_graph: numpy.ndarray
    :ivar dist: Neighbor distances as a CUDA tensor.
    :vartype dist: torch.Tensor
    :ivar neigh: Neighbor indices as a CUDA tensor.
    :vartype neigh: torch.Tensor
    :ivar gamma: Variational parameters for topic distributions.
    :vartype gamma: torch.Tensor or numpy.ndarray
    :ivar lambda_: Variational parameters for topics across modalities.
    :vartype lambda_: dict
    :ivar exp_E_log_beta: Expected log topic distributions.
    :vartype exp_E_log_beta: dict

    :methods:
        .. method:: VI(n_iter=20, max_iter_d=100)
            Perform Variational Inference (VI) to fit the model to the data.

            All observations are processed in a single GPU batch each iteration; padding 
            in the neighborhood arrays is handled internally so that neighborhoods of 
            varying size (radius mode) contribute correctly to the spatial similarity term.

            :param n_iter: Number of iterations for the VI algorithm. Default is 20.
            :type n_iter: int, optional
            :param max_iter_d: Maximum number of iterations for the E-step in each VI update. Default is 100.
            :type max_iter_d: int, optional

            :returns: None
            :rtype: None

            :example:
                .. code-block:: python

                    model = mtopic.tl.sMTM_GPU(mdata, n_topics=20, radius=0.05)
                    model.VI(n_iter=20)

    :example:

        .. code-block:: python

            import mtopic

            # Load spatial multimodal single-cell data
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Initialize and train the model
            model = mtopic.tl.sMTM_GPU(mdata, n_topics=20, radius=0.05)
            model.VI(n_iter=20)
    """

    def __init__(self, mdata, n_topics=20, radius=0.05, n_neighbors=None, seed=2291, spatial_key='coords', verbose=True):
        self.n_topics = n_topics
        self.radius = radius
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.device = "cuda"
        self.spatial_key = spatial_key
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        print(f"Program will run in {self.device} mode.")
        self._load_data(mdata)
        self._build_neighborhood_graph()
        self._init_params()

    def _load_data(self, X):
        self.modalities = list(X.mod.keys())
        self.n_obs = X.shape[0]
        self.barcodes = X.obs.index.tolist()
        self.features = {m: X[m].var.index.tolist() for m in self.modalities}
        
        coords = np.asarray(X.obsm[self.spatial_key])
        self.coords_scaled = (coords - np.min(coords, axis=0)) / np.max(np.max(coords, axis=0) - np.min(coords, axis=0))

        self.X_csr = dict()
        for mod in self.modalities:
            sp_mat = X[mod].X if isinstance(X[mod].X, scipy.sparse.csr_matrix) else X[mod].X.tocsr()
            self.X_csr[mod] = torch.sparse_csr_tensor(
                torch.from_numpy(sp_mat.indptr).to(torch.int64),
                torch.from_numpy(sp_mat.indices).to(torch.int64),
                torch.from_numpy(sp_mat.data).to(torch.float32),
                size=sp_mat.shape, device=self.device)
        self.n_var = {mod: X[mod].shape[1] for mod in self.modalities}
        self.coords = torch.from_numpy(self.coords_scaled).to(device=self.device, dtype=torch.float32)

    def _build_neighborhood_graph(self):
        if self.n_neighbors is not None:
            neigh = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.coords_scaled)
            distances, indices = neigh.kneighbors(self.coords_scaled)
        else:
            neigh = NearestNeighbors(radius=self.radius).fit(self.coords_scaled)
            distances_raw, indices_raw = neigh.radius_neighbors(self.coords_scaled, sort_results=True)
            
            max_n = max(len(x) for x in indices_raw)
            distances = np.full((self.n_obs, max_n), 1e9, dtype=np.float32)
            indices = np.zeros((self.n_obs, max_n), dtype=np.int64)
            
            for i, (d, idx) in enumerate(zip(distances_raw, indices_raw)):
                distances[i, :len(d)] = d
                indices[i, :len(idx)] = idx

        self.neighborhood_dist = distances
        self.neighborhood_graph = indices
        self.dist = torch.from_numpy(distances).to(device=self.device, dtype=torch.float32)
        self.neigh = torch.from_numpy(indices).to(device=self.device, dtype=torch.int64)

    def _init_params(self):
        self.eta, self.alpha = 0.01, 1 / self.n_topics
        self.lambda_ = dict()
        self.exp_E_log_beta = dict()
        
        for m in self.modalities:
            lambda_np = self.rng.gamma(100., 1./100., size=(self.n_topics, self.n_var[m]))
            self.lambda_[m] = torch.from_numpy(lambda_np).to(device=self.device, dtype=torch.float32)
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[m])
        self.gamma = torch.ones((self.n_obs, self.n_topics), device=self.device) * self.alpha

    def _predict_theta(self, var_lambda, max_iter_d=100):
        self.max_iter_d = max_iter_d
        self.lambda_ = {k: (v.clone() if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(self.device, dtype=torch.float32)) for k, v in var_lambda.items()}

        for mod in self.modalities:
            self.exp_E_log_beta[mod] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[mod])

        if not isinstance(self.gamma, torch.Tensor):
            self.gamma = torch.ones((self.n_obs, self.n_topics), device=self.device) * self.alpha

        batch = torch.arange(self.n_obs, device=self.device)
        batch_X = {m: slice_sparse_csr_rows(self.X_csr[m], batch) for m in self.modalities}
        
        gamma_batch, _ = _spatial_e_step_cuda(
            batch_X, self.alpha, self.exp_E_log_beta, self.max_iter_d, 
            self.gamma, self.neigh, self.dist
        )
        self.gamma = gamma_batch.detach().cpu().numpy()

    def _VI_update(self, batch):
        batch_X = {m: slice_sparse_csr_rows(self.X_csr[m], batch) for m in self.modalities}
        
        gamma_batch, lambda_update_batch = _spatial_e_step_cuda(
            batch_X, self.alpha, self.exp_E_log_beta, self.max_iter_d, 
            self.gamma, self.neigh, self.dist
        )
        
        self.gamma[batch] = gamma_batch
        for m in self.modalities: 
            self.lambda_[m] = self.eta + lambda_update_batch[m]
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[m])

    def VI(self, n_iter=20, max_iter_d=100):
        self.n_iter = n_iter
        self.max_iter_d = max_iter_d
        batch = torch.arange(self.n_obs, device=self.device)
        self.n_update = 1
        
        for _ in tqdm(range(self.n_iter), desc="VI (CUDA)") if self.verbose else range(self.n_iter):
            self._VI_update(batch)
            self.n_update += 1
            
        self.gamma = self.gamma.detach().cpu().numpy()
        for m in self.modalities:
            self.lambda_[m] = self.lambda_[m].detach().cpu().numpy()
            self.exp_E_log_beta[m] = self.exp_E_log_beta[m].cpu().numpy()
