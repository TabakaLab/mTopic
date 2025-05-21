import numpy as np
from scipy.special import psi
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import os
import pandas as pd


def _dirichlet_exp_E_log_prior(prior):
    exp_E_log_prior = psi(prior) - psi(np.sum(prior)) if len(prior.shape) == 1 \
            else psi(prior) - psi(np.sum(prior, axis=1))[:, np.newaxis]

    return np.exp(exp_E_log_prior)


def _e_step(X, 
            alpha,
            exp_E_log_beta,
            max_iter_d, 
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
    # Initialize topic distributions
    gamma = np.ones((D, K))

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

        # Find topic distributions gamma
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

        # Save converged topic distributions
        gamma[d, :] = gamma_d

        for m in modalities:
            phi_norm[m] = np.dot(exp_E_log_theta_d, exp_E_log_beta_d[m]) + 1e-100
            new_lambda[m][:, idx_d[m]] += np.outer(exp_E_log_theta_d, counts_d[m] / phi_norm[m])

    return gamma, new_lambda


class MTM:
    """
    Multimodal Topic Model (MTM) for single-cell data analysis.

    This class implements a Multimodal Topic Model (MTM) for analyzing single-cell data 
    across multiple modalities. It is designed to discover latent topics that capture 
    patterns and relationships between features across modalities. MTM can be trained using 
    Variational Inference (VI) or Stochastic Variational Inference (SVI) for efficient 
    learning from large datasets.

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
    :param n_jobs: 
        Number of CPU cores to use for parallel processing. If set to -1, uses all available cores. Default is 10.
    :type n_jobs: int, optional

    :ivar n_topics: Number of topics initialized by the model.
    :vartype n_topics: int
    :ivar seed: Random seed used for initializing the model.
    :vartype seed: int
    :ivar rng: Random number generator initialized with the provided seed.
    :vartype rng: numpy.random.Generator
    :ivar n_jobs: Number of parallel jobs used during computation.
    :vartype n_jobs: int
    :ivar X: Dictionary containing data matrices for each modality.
    :vartype X: dict
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
    :vartype gamma: numpy.ndarray
    :ivar lambda_: Variational parameters for topics.
    :vartype lambda_: dict
    :ivar exp_E_log_beta: Expected log topic distributions.
    :vartype exp_E_log_beta: dict

    :methods:
        .. method:: VI(n_iter=20, max_iter_d=100)
            Perform Variational Inference (VI) to infer topics from the data.

            VI is a deterministic approximation method that updates the model's variational parameters 
            over several iterations to optimize its fit to the data. Use VI for moderate-sized datasets 
            where the full dataset can be used in each iteration.

            :param n_iter: Number of iterations for the VI algorithm. Default is 20.
            :type n_iter: int, optional
            :param max_iter_d: Maximum iterations for the E-step in each VI update. Controls convergence criteria. Default is 100.
            :type max_iter_d: int, optional

            :returns: None
        
            :example:

                .. code-block:: python

                    import mtopic

                    # Load data and initialize MTM model
                    mdata = mtopic.read.h5mu("path/to/file.h5mu")
                    model = mtopic.tl.MTM(mdata, n_topics=20)

                    # Perform Variational Inference
                    model.VI(n_iter=20)

        .. method:: SVI(n_batches=100, batch_size=100, tau=1., kappa=0.75, max_iter_d=100)
            Perform Stochastic Variational Inference (SVI) for large-scale data.

            SVI divides the dataset into batches and uses stochastic updates to infer topics. This method 
            is efficient for large datasets where processing the entire dataset at once is computationally expensive.

            :param n_batches: Number of batches to divide the data into. Default is 100.
            :type n_batches: int, optional
            :param batch_size: Number of samples per batch. Smaller batch sizes use less memory but result in noisier updates. Default is 100.
            :type batch_size: int, optional
            :param tau: Initial learning rate for SVI. Default is 1.0.
            :type tau: float, optional
            :param kappa: Learning rate decay parameter. Typically between 0.5 and 1.0. Default is 0.75.
            :type kappa: float, optional
            :param max_iter_d: Maximum iterations for the E-step in each SVI update. Default is 100.
            :type max_iter_d: int, optional

            :returns: None

            :example:

                .. code-block:: python

                    import mtopic

                    # Load data and initialize MTM model
                    mdata = mtopic.read.h5mu("path/to/file.h5mu")
                    model = mtopic.tl.MTM(mdata, n_topics=20)

                    # Perform Stochastic Variational Inference
                    model.SVI(n_batches=100, batch_size=100)

    :example:

        .. code-block:: python

            import mtopic

            # Load multimodal single-cell data
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Initialize MTM model
            model = mtopic.tl.MTM(mdata, n_topics=20)

            # Fit model using Variational Inference
            model.VI(n_iter=20)

            # Fit model using Stochastic Variational Inference
            model.SVI(n_batches=100, batch_size=100)
    """


    def __init__(self, 
                 mdata,
                 n_topics=20,
                 seed=2291, 
                 n_jobs=10):
        self.n_topics = n_topics
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.n_jobs = n_jobs

        if self.n_jobs > cpu_count():
            self.n_jobs = cpu_count()

        self._load_data(mdata)
        self._init_params()

    def _load_data(self, 
                   X):
        self.X = {mod: X[mod].X for mod in X.mod}
        self.modalities = list(self.X.keys())
        self.features = {mod: np.asarray(X[mod].var_names) for mod in X.mod}
        self.barcodes = X.obs.index.tolist()
        self.n_obs = self.X[self.modalities[0]].shape[0]
        self.n_mod = len(self.X)
        self.n_var = dict()
        for mod in self.modalities:
            self.n_var[mod] = self.X[mod].shape[1]
    
    def _init_params(self):
        self.eta = 0.01
        self.alpha = 1 / self.n_topics
        self.gamma = np.ones((self.n_obs, self.n_topics)) * self.alpha
        self.lambda_ = dict()
        self.exp_E_log_beta = dict()

        for m in self.modalities:
            self.lambda_[m] = self.rng.gamma(100., 1./100., (self.n_topics, self.n_var[m]))
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior(self.lambda_[m])

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
            delayed(_e_step)(
            {m: self.X[m][batch_job] for m in self.modalities}, 
            self.alpha,
            self.exp_E_log_beta,
            self.max_iter_d) for batch_job in batch_split)
        
        gamma_list, new_lambda_list = zip(*output)

        self.gamma = np.vstack(gamma_list)
        
        new_lambda = dict()
        for m in self.modalities:
            new_lambda[m] = np.zeros(self.lambda_[m].shape)
            for lambda_update in new_lambda_list:
                new_lambda[m] += lambda_update[m]
            new_lambda[m] = self.eta + new_lambda[m] * self.exp_E_log_beta[m]

            self.lambda_[m] = new_lambda[m]
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior(self.lambda_[m])

    def VI(self, 
           n_iter=20, 
           max_iter_d=100):

        self.n_iter = n_iter
        self.max_iter_d = max_iter_d

        batch = [i for i in range(self.n_obs)]

        self.n_update = 1
        for _ in tqdm(range(self.n_iter)):
            self._VI_update(batch)
            self.n_update += 1  

    def _SVI_update(self, 
                    batch):
        self._set_batch_n_jobs(batch)
        batch_split = np.array_split(batch, self.n_jobs)

        output = Parallel(n_jobs=self.n_jobs)(
            delayed(_e_step)(
            {m: self.X[m][batch_job] for m in self.modalities}, 
            self.alpha,
            self.exp_E_log_beta,
            self.max_iter_d) for batch_job in batch_split)

        gamma_list, new_lambda_list = zip(*output)

        self.gamma[batch] = np.vstack(gamma_list)

        rhot = pow(self.tau + self.n_update, -self.kappa)

        new_lambda = dict()
        for m in self.modalities:
            new_lambda[m] = np.zeros(self.lambda_[m].shape)
            for lambda_update in new_lambda_list:
                new_lambda[m] += lambda_update[m]
        
            new_lambda[m] = self.eta + self.n_obs * new_lambda[m] * self.exp_E_log_beta[m] / self.batch_size

            self.lambda_[m] = (1 - rhot) * self.lambda_[m] + rhot * new_lambda[m]
            self.exp_E_log_beta[m] = _dirichlet_exp_E_log_prior(self.lambda_[m])
        
    def SVI(self,
            n_batches=100,
            batch_size=100,
            tau=1., 
            kappa=0.75, 
            max_iter_d=100):
        
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.max_iter_d = max_iter_d
        self.tau = tau
        self.kappa = kappa

        batches = self.rng.choice(self.n_obs, (self.n_batches, self.batch_size))
        self.batches = batches

        self.n_update = 1
        for batch in tqdm(self.batches):
            self._SVI_update(batch)
            self.n_update += 1
