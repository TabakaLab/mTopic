import torch
import numpy as np
import scipy

def dirichlet_exp_E_log_prior(prior):
    if isinstance(prior, np.ndarray):
        if prior.ndim == 1:
            exp_E_log_prior = np.exp(scipy.special.psi(prior) - scipy.special.psi(np.sum(prior)))
        elif prior.ndim == 2:
            exp_E_log_prior = np.exp(scipy.special.psi(prior) - scipy.special.psi(np.sum(prior, axis=1, keepdims=True)))
        else:
            raise ValueError("prior must be 1D or 2D")

    elif torch.is_tensor(prior):
        digamma_prior = torch.special.psi(prior)

        if prior.ndim == 1:
            digamma_sum = torch.special.psi(torch.sum(prior))
        elif prior.ndim == 2:
            digamma_sum = torch.special.psi(torch.sum(prior, dim=1, keepdim=True))
        else:
            raise ValueError("prior must be 1D or 2D")

        exp_E_log_prior = torch.exp(digamma_prior - digamma_sum)

    else:
        raise TypeError("prior must be numpy.ndarray or torch.Tensor")

    return exp_E_log_prior


def slice_sparse_csr_rows(csr_tensor: torch.Tensor, row_indices: torch.Tensor):
    """
    Slices a PyTorch sparse CSR tensor to create a new sparse tensor
    containing only the specified rows.

    This function is optimized for GPU performance and correctly handles
    empty slices or rows with no non-zero elements. It reconstructs the
    sparse CSR format (crow_indices, col_indices, values) for the
    subset of rows.

    :param csr_tensor:
        The input sparse CSR tensor to be sliced.
    :type csr_tensor: torch.Tensor
    :param row_indices:
        A 1D tensor of row indices to be selected from the `csr_tensor`.
        Must be on the same device as `csr_tensor`.
    :type row_indices: torch.Tensor
    
    :returns:
        A new sparse CSR tensor containing only the rows specified by
        `row_indices`. The new tensor is on the same device as the input.
    :rtype: torch.Tensor
    """
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


def e_step_cpu(X, alpha, exp_E_log_beta, max_iter_d, conv_threshold=0.0001):
    modalities = list(X.keys())
    D = X[modalities[0]].shape[0]
    N = {m: X[m].shape[1] for m in modalities}
    K = exp_E_log_beta[modalities[0]].shape[0]

    new_lambda = dict()
    for m in modalities:
        new_lambda[m] = np.zeros((K, N[m]))

    phi_norm = dict()
    for m in modalities:
        phi_norm[m] = np.zeros(N[m])
    gamma = np.ones((D, K))

    for d in range(D):
        gamma_d = np.ones(K)
        exp_E_log_theta_d = dirichlet_exp_E_log_prior(gamma_d)

        idx_d = dict()
        counts_d = dict()
        exp_E_log_beta_d = dict()
        for m in modalities:
            idx_d[m] = X[m][d].nonzero()[1]
            counts_d[m] = X[m][d][:, idx_d[m]].toarray().flatten()
            exp_E_log_beta_d[m] = exp_E_log_beta[m][:, idx_d[m]]

        for _ in range(max_iter_d):
            prev_gamma = gamma_d

            for m in modalities:
                phi_norm[m] = np.dot(exp_E_log_theta_d, exp_E_log_beta_d[m]) + 1e-100
            gamma_d = np.ones(K) * alpha
            for m in modalities:
                gamma_d += exp_E_log_theta_d * np.dot(counts_d[m] / phi_norm[m], exp_E_log_beta_d[m].T)

            exp_E_log_theta_d = dirichlet_exp_E_log_prior(gamma_d)
            meanchange = np.mean(abs(gamma_d - prev_gamma))
            if (meanchange < conv_threshold):
                break

        gamma[d, :] = gamma_d

        for m in modalities:
            phi_norm[m] = np.dot(exp_E_log_theta_d, exp_E_log_beta_d[m]) + 1e-100
            new_lambda[m][:, idx_d[m]] += np.outer(exp_E_log_theta_d, counts_d[m] / phi_norm[m])

    return gamma, new_lambda

@torch.compile 
def e_step_cuda(
    X_batch: dict[str, torch.Tensor],
    alpha: float,
    exp_E_log_beta: dict[str, torch.Tensor],
    max_iter_d: int,
    conv_threshold: float = 0.0001,
):
    modalities = list(X_batch.keys())

    first_mod = modalities[0]
    
    device = X_batch[first_mod].device
    D_batch = X_batch[first_mod].shape[0]
    K = exp_E_log_beta[first_mod].shape[0]
    gamma = torch.full((D_batch, K), alpha, device=device)
    prev_gamma = torch.empty_like(gamma)
    for iter_num in range(max_iter_d):
        prev_gamma.copy_(gamma)
        exp_E_log_theta = dirichlet_exp_E_log_prior(gamma)
        gamma.fill_(alpha)

        for m in modalities:
            Xm_batch = X_batch[m]
            nnz = Xm_batch.values().shape[0]

            if nnz > 0:
                crow_indices = Xm_batch.crow_indices()
                col_indices = Xm_batch.col_indices()
                values = Xm_batch.values().to(torch.float32)

                docs = torch.repeat_interleave(torch.arange(D_batch, device=device), crow_indices[1:] - crow_indices[:-1])
                words = col_indices
                
                exp_E_log_beta_m = exp_E_log_beta[m]
                theta_gathered = exp_E_log_theta[docs]
                
                beta_gathered = exp_E_log_beta_m[:, words].T

                phi_elements = theta_gathered * beta_gathered + 1e-10

                phi_norm = torch.sum(phi_elements, dim=1, keepdim=True)
                phi_normalized = phi_elements / phi_norm

                gamma.scatter_add_(0, docs[:, None].expand(-1, K), values[:, None] * phi_normalized)
        
        meanchange = torch.mean(torch.abs(gamma - prev_gamma))
        if meanchange < conv_threshold:
            break
            
    exp_E_log_theta_final = dirichlet_exp_E_log_prior(gamma)
    suff_stats_lambda = {m: torch.zeros_like(exp_E_log_beta[m], device=device) for m in modalities}

    for m in modalities:
        Xm_batch = X_batch[m]
        nnz = Xm_batch.values().shape[0]

        if nnz > 0:
            crow_indices = Xm_batch.crow_indices()
            col_indices = Xm_batch.col_indices()
            values = Xm_batch.values().to(torch.float32)
            docs = torch.repeat_interleave(torch.arange(D_batch, device=device), crow_indices[1:] - crow_indices[:-1])
            words = col_indices

            exp_E_log_beta_m = exp_E_log_beta[m]
            theta_gathered = exp_E_log_theta_final[docs]
            beta_gathered = exp_E_log_beta_m[:, words].T
            phi_elements = theta_gathered * beta_gathered + 1e-10
            phi_norm = torch.sum(phi_elements, dim=1, keepdim=True)

            phi_norm = phi_norm + 1e-10
            phi_normalized = phi_elements / phi_norm

            suff_stats_lambda[m].scatter_add_(1, words[None, :].expand(K, -1), (values[:, None] * phi_normalized).T)
    return gamma, suff_stats_lambda
