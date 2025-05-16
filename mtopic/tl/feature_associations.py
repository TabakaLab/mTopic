import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm


class SingleFeatureModel(nn.Module):
    def __init__(self, P_g, temperature=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(P_g) * 0.1)
        self.temperature = temperature

    def forward(self):
        return torch.softmax(self.logits / self.temperature, dim=0)
        

def loss_function(A_g, B_g, X_g, lambda_reg=0.1, lambda_entropy=0.01, lambda_spread=0.05):
    reconstructed_A_g = torch.matmul(B_g, X_g) + 1e-6
    kl_loss = torch.sum(A_g * (torch.log(A_g + 1e-6) - torch.log(reconstructed_A_g)))
    sparsity_loss = lambda_reg * torch.sum(torch.pow(X_g + 1e-6, 0.1))
    sparsity_loss += lambda_reg * torch.sum(torch.log(X_g + 1e-6))
    entropy_loss = -lambda_entropy * torch.sum(X_g * torch.log(X_g + 1e-6))
    spread_loss = lambda_spread * torch.sum(X_g * torch.log(X_g + 1e-6))

    return kl_loss + sparsity_loss + entropy_loss + spread_loss


def is_row_normalized(matrix: torch.Tensor, tol: float = 1e-3) -> bool:
    """
    Check if a tensor is row-normalized (rows sum to 1 within a tolerance).
    
    :param matrix: 2D tensor to check.
    :param tol: Tolerance for row sum deviation from 1.0.
    :return: True if all rows are normalized, False otherwise.
    """
    row_sums = matrix.sum(dim=1)
    
    return torch.all(torch.abs(row_sums - 1.0) < tol)


def feature_associations(A: torch.Tensor,
                         A_var: pd.DataFrame,
                         B: torch.Tensor, 
                         B_var: pd.DataFrame,
                         mask: torch.Tensor = None, 
                         n_epochs = 10000, 
                         lambda_reg = 1e-4,
                         lambda_entropy = 1e-3,
                         lambda_spread = 0.05,
                         temperature = 0.2,
                         normalize = True,
                         seed = 1898,
                         n_threads = 10):
    """
    Cross-modality feature associations.

    Learns a feature-level probabilistic mapping from one modality to another
    using KL divergence minimization with regularization. The optimization is
    done independently for each feature in `A`.

    The result is a sparse matrix showing how features in modality B (columns) 
    associate with features in modality A (rows). The model applies softmax-based
    weighting with optional entropy and sparsity regularization.

    :param A: Topic-feature matrix of modality A (shape: `n_topics x n_features_A`).
    :type A: torch.Tensor

    :param A_var: `.var` DataFrame from modality A, used for column names.
    :type A_var: pandas.DataFrame

    :param B: Topic-feature matrix of modality B (shape: `n_topics x n_features_B`).
    :type B: torch.Tensor

    :param B_var: `.var` DataFrame from modality B, used for row names.
    :type B_var: pandas.DataFrame

    :param mask: Optional boolean mask (shape: `n_features_B x n_features_A`) specifying which feature pairs to consider. (default: `None`)
    :type mask: torch.Tensor or None

    :param n_epochs: Number of optimization steps per target feature. (default: `10000`)
    :type n_epochs: int

    :param lambda_reg: Regularization coefficient for sparsity. (default: `1e-4`)
    :type lambda_reg: float

    :param lambda_entropy: Regularization coefficient for entropy. (default: `1e-3`)
    :type lambda_entropy: float

    :param lambda_spread: Regularization coefficient for weight spread. (default: `0.05`)
    :type lambda_spread: float

    :param temperature: Softmax temperature for controlling assignment sharpness. (default: `0.2`)
    :type temperature: float

    :param normalize: Whether to check and normalize `A` and `B` to ensure row sums equal 1. (default: `True`)
    :type normalize: bool

    :param seed: Random seed for reproducibility. (default: `1898`)
    :type seed: int

    :param n_threads: Number of threads to use for Torch. (default: `10`)
    :type n_threads: int

    :returns: DataFrame of shape `(n_features_B, n_features_A)` with association weights.
    :rtype: pandas.DataFrame

    :example:

        .. code-block:: python

            # A, B = torch.Tensor (topics x features) with topic loadings
            # A_var, B_var = .var dataframes from each modality
            # mask = (optional) a prior-defined feature-feature mask

            df = feature_associations(A, A_var, B, B_var, mask=mask, normalize=True)
            df.head()  # View the learned feature associations
    """

    assert(isinstance(A, torch.Tensor))
    assert(isinstance(B, torch.Tensor))
    assert(A.shape[0] == B.shape[0])
    assert(A.shape[1] == A_var.shape[0])
    assert(B.shape[1] == B_var.shape[0])
    
    if mask != None:
        assert(mask.shape[0] == B.shape[1])
        assert(mask.shape[1] == A.shape[1])

    if normalize:
        if not is_row_normalized(A):
            A = A / A.sum(dim=1, keepdim=True)
        if not is_row_normalized(B):
            B = B / B.sum(dim=1, keepdim=True)

    torch.manual_seed(seed)
    torch.set_num_threads(n_threads)

    T, G = A.shape
    T, P = B.shape
    
    X_final = torch.zeros(P, G)
    
    for g in tqdm(range(G)):
        relevant_features = mask[:, g].bool() if mask != None else torch.ones(P, dtype=bool)
        P_g = relevant_features.sum().item()
        
        if P_g == 0:
            continue

        A_g = A[:, g]
        B_g = B[:, relevant_features]
        
        model = SingleFeatureModel(P_g, temperature=temperature)
        optimizer = torch.optim.Adam([model.logits], lr=1e-4)
    
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            X_g = model()
            loss = loss_function(A_g, B_g, X_g, lambda_reg, lambda_entropy, lambda_spread)
            loss.backward()
            optimizer.step()
    
        X_final[relevant_features, g] = model().detach()
    
    X_np = X_final.numpy()
    
    return pd.DataFrame(X_np, index=B_var.index, columns=A_var.index)
