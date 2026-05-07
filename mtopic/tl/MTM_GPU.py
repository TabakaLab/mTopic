import warnings

import numpy as np
import scipy
import torch
from tqdm import tqdm

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
    phi = (exp_E_log_theta_docs * beta_m) + 1e-30
    phi /= phi.sum(dim=1, keepdim=True)
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


def _e_step_cuda(X_batch, alpha, exp_E_log_beta, max_iter_d, conv_threshold=0.0001):
    modalities = list(X_batch.keys())
    device = X_batch[modalities[0]].device
    D_batch = X_batch[modalities[0]].shape[0]
    K = exp_E_log_beta[modalities[0]].shape[0]

    gamma = torch.ones((D_batch, K), device=device, dtype=torch.float32)
    prev_gamma = torch.empty_like(gamma)

    prepared_data = {}
    for m in modalities:
        Xm = X_batch[m]
        if Xm.values().shape[0] > 0:
            crow = Xm.crow_indices()
            col = Xm.col_indices()
            docs = torch.repeat_interleave(
                torch.arange(D_batch, device=device), 
                crow[1:] - crow[:-1]
            )
            beta_m = exp_E_log_beta[m][:, col].T 
            val = Xm.values().unsqueeze(1)
            prepared_data[m] = (docs, beta_m, val)
            
    for _ in range(max_iter_d):
        prev_gamma.copy_(gamma)
        exp_E_log_theta = _dirichlet_exp_E_log_prior_cuda(gamma)
        gamma.fill_(alpha)

        for m in modalities:
            if m not in prepared_data: 
                continue
            docs, beta_m, val = prepared_data[m]
            phi = _compute_phi_fused(exp_E_log_theta[docs], beta_m)
            gamma.scatter_add_(0, docs[:, None].expand(-1, K), val * phi)

        meanchange = torch.mean(torch.abs(gamma - prev_gamma))
        if meanchange < conv_threshold:
            break
            
    exp_E_log_theta_final = _dirichlet_exp_E_log_prior_cuda(gamma)
    suff_stats_lambda = {m: torch.zeros_like(exp_E_log_beta[m]) for m in modalities}

    for m in modalities:
        if m not in prepared_data: 
            continue
        docs, beta_m, val = prepared_data[m]
        phi = _compute_phi_fused(exp_E_log_theta_final[docs], beta_m)
        words = X_batch[m].col_indices()
        suff_stats_lambda[m].scatter_add_(1, words[None, :].expand(K, -1), (val * phi).T)

    return gamma, suff_stats_lambda

class MTM_GPU:
    """
    GPU-accelerated Multimodal Topic Model.

    This class implements a CUDA-accelerated version of the Multimodal Topic Model (MTM) 
    for analyzing single-cell data across multiple modalities. It is designed to discover 
    latent topics that capture patterns and relationships between features across modalities. 
    MTM_GPU can be trained using Variational Inference (VI) or Stochastic Variational 
    Inference (SVI) for efficient learning from large datasets. The model is mathematically 
    equivalent to :class:`MTM` but executes E-step and M-step updates on the GPU using 
    sparse CSR tensors for substantial speedups on large datasets.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data. Each modality represents 
        a feature space (e.g., RNA, ATAC, protein), which is used for topic modeling.
    :type mdata: muon.MuData
    :param n_topics: 
        The number of latent topics to infer. Each topic corresponds to a distinct pattern or 
        feature distribution across modalities. Default is 20.
    :type n_topics: int, optional
    :param seed: 
        Random seed for reproducibility. Ensures consistent initialization and results. Default is 2291.
    :type seed: int, optional
    :param verbose: 
        If True, displays a progress bar during training. Default is True.
    :type verbose: bool, optional

    :ivar n_topics: Number of topics initialized by the model.
    :vartype n_topics: int
    :ivar seed: Random seed used for initializing the model.
    :vartype seed: int
    :ivar rng: Random number generator initialized with the provided seed.
    :vartype rng: numpy.random.Generator
    :ivar device: Compute device used by the model (always ``"cuda"``).
    :vartype device: str
    :ivar X: Dictionary containing sparse CSR data tensors for each modality.
    :vartype X: dict
    :ivar X_csr: Dictionary of sparse CSR tensors for each modality (alias of ``X``).
    :vartype X_csr: dict
    :ivar modalities: List of modalities in the dataset.
    :vartype modalities: list
    :ivar features: Dictionary of feature names for each modality.
    :vartype features: dict
    :ivar barcodes: List of sample barcodes.
    :vartype barcodes: list
    :ivar n_obs: Number of samples (observations) in the dataset.
    :vartype n_obs: int
    :ivar n_mod: Number of modalities in the dataset.
    :vartype n_mod: int
    :ivar n_var: Dictionary containing the number of features for each modality.
    :vartype n_var: dict
    :ivar eta: Prior for topics.
    :vartype eta: float
    :ivar alpha: Prior for topic distributions.
    :vartype alpha: float
    :ivar gamma: Variational parameters for topic distributions.
    :vartype gamma: torch.Tensor or numpy.ndarray
    :ivar lambda_: Variational parameters for topics.
    :vartype lambda_: dict
    :ivar exp_E_log_beta: Expected log topic distributions.
    :vartype exp_E_log_beta: dict

    :methods:
        .. method:: VI(n_iter=20, max_iter_d=100, batch_size=2048)
            Perform Variational Inference (VI) to infer topics from the data.

            VI processes all observations each iteration. To bound GPU memory, observations 
            are processed in chunks of ``batch_size`` cells per E-step call; sufficient 
            statistics are accumulated across chunks before the M-step update. The result is 
            mathematically equivalent to a single full-batch VI iteration.

            :param n_iter: Number of iterations for the VI algorithm. Default is 20.
            :type n_iter: int, optional
            :param max_iter_d: Maximum iterations for the E-step in each VI update. Controls convergence criteria. Default is 100.
            :type max_iter_d: int, optional
            :param batch_size: Number of cells processed per E-step chunk. Larger values use more GPU memory but reduce kernel-launch overhead. Default is 2048.
            :type batch_size: int, optional

            :returns: None

            :example:

                .. code-block:: python

                    import mtopic

                    # Load data and initialize MTM_GPU model
                    mdata = mtopic.read.h5mu("path/to/file.h5mu")
                    model = mtopic.tl.MTM_GPU(mdata, n_topics=20)

                    # Perform Variational Inference
                    model.VI(n_iter=20)

        .. method:: SVI(n_batches=100, batch_size=512, tau=1., kappa=0.75, max_iter_d=100)
            Perform Stochastic Variational Inference (SVI) for large-scale data.

            SVI samples random mini-batches of cells and uses stochastic updates to infer 
            topics. This method is efficient for large datasets where processing the entire 
            dataset at once is computationally expensive.

            :param n_batches: Number of stochastic updates performed. Default is 100.
            :type n_batches: int, optional
            :param batch_size: Number of samples per batch. Smaller batch sizes use less memory but result in noisier updates. Default is 512.
            :type batch_size: int, optional
            :param tau: Initial learning rate offset for SVI. Default is 1.0.
            :type tau: float, optional
            :param kappa: Learning rate decay parameter. Typically between 0.5 and 1.0. Default is 0.75.
            :type kappa: float, optional
            :param max_iter_d: Maximum iterations for the E-step in each SVI update. Default is 100.
            :type max_iter_d: int, optional

            :returns: None

            :example:

                .. code-block:: python

                    import mtopic

                    # Load data and initialize MTM_GPU model
                    mdata = mtopic.read.h5mu("path/to/file.h5mu")
                    model = mtopic.tl.MTM_GPU(mdata, n_topics=20)

                    # Perform Stochastic Variational Inference
                    model.SVI()

    :example:

        .. code-block:: python

            import mtopic

            # Load multimodal single-cell data
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Initialize MTM_GPU model
            model = mtopic.tl.MTM_GPU(mdata, n_topics=20)

            # Fit model using Variational Inference
            model.VI(n_iter=20)

            # Fit model using Stochastic Variational Inference
            model.SVI(n_batches=100, batch_size=512)
    """

    def __init__(self, 
                 mdata,
                 n_topics=20,
                 seed=2291,
                 verbose=True):
        self.n_topics = n_topics
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.device = "cuda"
        self.features = dict()
        self.X_csr = dict()
        
        self.verbose = verbose
        self._load_data(mdata)
        self._init_params()

    def _load_data(self, X):
        self.X = {}
        self.modalities = list(X.mod)
        self.n_mod = len(self.modalities)
        self.n_obs = X[self.modalities[0]].X.shape[0]
        
        all_data = [X[mod] for mod in self.modalities]
        self.n_var = dict()
        for i, modality in enumerate(self.modalities):
            adata = all_data[i]
            sparse_matrix = adata.X

            if isinstance(sparse_matrix, scipy.sparse.csc_matrix):
                sparse_matrix = sparse_matrix.tocsr()
            elif not isinstance(sparse_matrix, scipy.sparse.csr_matrix):
                sparse_matrix = scipy.sparse.csr_matrix(sparse_matrix)

            n_obs, n_var = sparse_matrix.shape
            self.n_var[modality] = n_var

            crow_indices = torch.from_numpy(sparse_matrix.indptr).to(dtype=torch.int64, device=self.device)
            col_indices = torch.from_numpy(sparse_matrix.indices).to(dtype=torch.int64, device=self.device)
            values = torch.from_numpy(sparse_matrix.data).to(dtype=torch.float32, device=self.device)

            self.X_csr[modality] = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, size=(n_obs, n_var), device=self.device)
            self.X[modality] = self.X_csr[modality]

            self.features[modality] = adata.var_names.to_list()
            
            current_barcodes = adata.obs_names.to_list()
            if i == 0:
                self.barcodes = current_barcodes

    def _init_params(self):
        self.eta = 0.01
        self.alpha = 1.0 / self.n_topics
        self.lambda_ = dict()
        self.exp_E_log_beta = dict()
        
        self.gamma = torch.ones((self.n_obs, self.n_topics), device=self.device) * self.alpha
            
        for m in self.modalities:
            lambda_np = self.rng.gamma(100., 1./100., (self.n_topics, self.n_var[m]))
            self.lambda_[m] = torch.from_numpy(lambda_np).to(dtype=torch.float32, device=self.device)
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[m])

    def _predict_theta(self, var_lambda, max_iter_d=100, batch_size=2048):
        self.max_iter_d = max_iter_d
        self.lambda_ = {k: v.clone() if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(self.device) for k, v in var_lambda.items()}
        
        for mod in self.modalities:
            self.exp_E_log_beta[mod] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[mod])

        all_indices = np.arange(self.n_obs)
        n_batches = int(np.ceil(self.n_obs / batch_size))
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, self.n_obs)
            current_batch_indices = torch.from_numpy(all_indices[start_idx:end_idx]).to(device=self.device)

            batch_X = {m: slice_sparse_csr_rows(self.X_csr[m], current_batch_indices) for m in self.modalities}

            gamma_batch, _ = _e_step_cuda(
                X_batch=batch_X, alpha=self.alpha, exp_E_log_beta=self.exp_E_log_beta,
                conv_threshold=0.0001, max_iter_d=self.max_iter_d
            )
            self.gamma.data[current_batch_indices] = gamma_batch.data

        self.gamma = self.gamma.detach().cpu().numpy()

    def _VI_update(self, batch_size):
        all_indices = np.arange(self.n_obs)
        n_batches = int(np.ceil(len(all_indices) / batch_size))
        
        suff_stats_lambda_accum = {
            m: torch.zeros_like(self.lambda_[m], device=self.device) for m in self.modalities
        }

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_indices))
            current_batch_indices_np = all_indices[start_idx:end_idx]
            current_batch_indices = torch.from_numpy(current_batch_indices_np).to(device=self.device, non_blocking=True)

            batch_X = {
                m: slice_sparse_csr_rows(self.X_csr[m], current_batch_indices)
                for m in self.modalities
            }

            gamma_batch, lambda_update_batch = _e_step_cuda(
                X_batch=batch_X,
                alpha=self.alpha,
                exp_E_log_beta=self.exp_E_log_beta,
                conv_threshold=0.0001,
                max_iter_d=self.max_iter_d
            )

            self.gamma.data[current_batch_indices] = gamma_batch.data

            for m in self.modalities:
                suff_stats_lambda_accum[m] += lambda_update_batch[m]

        for m in self.modalities:
            self.lambda_[m].data = self.eta + suff_stats_lambda_accum[m]
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[m])

    def VI(self, n_iter=20, max_iter_d=100, batch_size=2048):
        self.max_iter_d = max_iter_d

        for _ in tqdm(range(n_iter)) if self.verbose else range(n_iter):
            self._VI_update(batch_size)

        self.gamma = self.gamma.detach().cpu().numpy()
        for m in self.modalities:
            self.lambda_[m] = self.lambda_[m].detach().cpu().numpy()
            self.exp_E_log_beta[m] = self.exp_E_log_beta[m].cpu().numpy()

    def _SVI_update(self, batch):
        batch = torch.from_numpy(batch).to(device=self.device, non_blocking=True)

        batch_X = {
            m: slice_sparse_csr_rows(self.X_csr[m], batch)
            for m in self.modalities
        }

        gamma_batch, suff_stats_lambda_batch = _e_step_cuda(
            X_batch=batch_X,
            alpha=self.alpha,
            exp_E_log_beta=self.exp_E_log_beta,
            max_iter_d=self.max_iter_d
        )

        self.gamma.data[batch] = gamma_batch.data

        rhot = pow(self.tau + self.n_update, -self.kappa)

        for m in self.modalities:
            scaled_suff_stats = (self.n_obs / len(batch)) * suff_stats_lambda_batch[m]
            lambda_proposal_m = self.eta + scaled_suff_stats

            self.lambda_[m].data = (1 - rhot) * self.lambda_[m].data + rhot * lambda_proposal_m
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior_cuda(self.lambda_[m])

    def SVI(self,
            n_batches=100,
            batch_size=512,
            tau=1., 
            kappa=0.75, 
            max_iter_d=100):

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.max_iter_d = max_iter_d
        self.tau = tau
        self.kappa = kappa
        self.n_update = 1

        batches = self.rng.choice(self.n_obs, (self.n_batches, self.batch_size))

        for batch in tqdm(batches):
            self._SVI_update(batch)
            self.n_update += 1

        self.gamma = self.gamma.detach().cpu().numpy()
        for m in self.modalities:
            self.lambda_[m] = self.lambda_[m].detach().cpu().numpy()
            self.exp_E_log_beta[m] = self.exp_E_log_beta[m].cpu().numpy()
