import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
import warnings
from . import tfidf, clr, scale_counts
from ..tl import MTM, sMTM


def _preprocess(mdata):
    for mod in mdata.mod:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
            if mod == "prot":
                clr(mdata, mod=mod)
            else:
                tfidf(mdata, mod=mod)
    scale_counts(mdata)


def _build_model(mdata, K, n_jobs, is_spatial, spatial_key):
    if is_spatial:
        return sMTM(mdata, n_topics=K, n_jobs=n_jobs, verbose=False, spatial_key=spatial_key)
    return MTM(mdata, n_topics=K, n_jobs=n_jobs, verbose=False)


def _extract_theta(model, mdata_test, K):
    gamma = pd.DataFrame(model.gamma)
    theta = gamma.div(gamma.sum(axis=1), axis=0)
    theta.columns = [f"T{i + 1}" for i in range(K)]
    theta.index = mdata_test.obs.index
    return theta


def _extract_beta(lambda_train, mdata_ref, K):
    beta = {}
    for mod, lam in lambda_train.items():
        df = pd.DataFrame(lam)
        df = df.div(df.sum(axis=1), axis=0)
        df.columns = mdata_ref.mod[mod].var.index
        df.index = [f"T{i + 1}" for i in range(K)]
        beta[mod] = df
    return beta


def _log_likelihood(X_test, theta, beta, eps=1e-12):
    theta_np = theta.to_numpy()
    ll_total = 0.0
    ll_per_mod = {}

    for mod, B_df in beta.items():
        B_df = B_df.loc[theta.columns]
        X = X_test[mod]

        if hasattr(X, "columns"):
            B_df = B_df[X.columns]
            X_mat = X.to_numpy()
        else:
            X_mat = X

        B = B_df.to_numpy()
        X_mat = sp.csr_matrix(X_mat) if not sp.issparse(X_mat) else X_mat.tocsr()

        ll = 0.0
        for d in range(X_mat.shape[0]):
            start, end = X_mat.indptr[d], X_mat.indptr[d + 1]
            idx = X_mat.indices[start:end]
            cnt = X_mat.data[start:end]
            if len(idx) == 0:
                continue
            p = theta_np[d] @ B[:, idx]
            ll += np.sum(cnt * np.log(p + eps))

        ll_per_mod[mod] = ll
        ll_total += ll

    return ll_total, ll_per_mod


def estimate_topic_number(
    mdata,
    Ks=[5, 20, 40, 60, 80, 100],
    *,
    n_folds=5,
    n_iter=50,
    n_jobs=10,
    is_spatial=False,
    spatial_key="coords",
    seed=2291
):
    """
    K-fold cross-validation for MTM and sMTM via held-out log-likelihood
    to estimate the optimal number of topics.

    For every combination of fold × K, cells are split into train / test subsets,
    the model is trained on the training set, lambda is transferred to predict
    theta on the test set, and the Poisson log-likelihood is evaluated on raw
    test counts. Per-fold results are averaged across folds and appended as
    summary rows (``CV == "total"``).

    Results are stored in ``mdata.uns["CV_results"]`` and fold assignments
    in ``mdata.obs["CV_split"]``. Results are also written to
    ``<output_prefix>.csv``.

    :param mdata:
        Pre-loaded multimodal dataset with raw counts.
    :type mdata: muon.MuData
    :param Ks:
        Topic numbers to evaluate. There need to be at least 6 values. 
        Default is ``[5, 20, 40, 60, 80, 100]``.
    :type Ks: array-like of int, optional
    :param n_folds:
        Number of CV folds. Default is ``5``.
    :type n_folds: int, optional
    :param n_iter:
        VI iterations per model fit. Default is ``50``.
    :type n_iter: int, optional
    :param n_jobs:
        Parallelism for the model constructor. Default is ``10``.
    :type n_jobs: int, optional
    :param is_spatial:
        Use ``sMTM`` when ``True``, ``MTM`` when ``False``. Default is ``False``.
    :type is_spatial: bool, optional
    :param spatial_key:
        Key in ``mdata.obsm`` containing spatial coordinates, used when
        ``is_spatial=True``. Default is ``"coords"``.
    :type spatial_key: str, optional
    :param seed:
        RNG seed for fold assignment. Default is ``2291``.
    :type seed: int, optional

    :returns:
        ``None``. Results are stored in ``mdata.uns["CV_results"]`` and
        fold assignments in ``mdata.obs["CV_split"]``.
    :rtype: None
    """
    
    Ks = list(Ks)
    assert len(Ks) >= 6, "At least 6 K values are required for reliable fitting."

    modality_names = list(mdata.mod.keys())
    N = mdata.n_obs

    rng = np.random.default_rng(seed=seed)
    fold_indices = np.array_split(rng.permutation(N), n_folds)

    split_assignments = np.empty(N, dtype=int)
    for fold_id, idx in enumerate(fold_indices, start=1):
        split_assignments[idx] = fold_id
    mdata.obs["CV_split"] = split_assignments

    result_rows = []

    for fold_id, test_idx in enumerate(fold_indices, start=1):
        mask = np.zeros(N, dtype=bool)
        mask[test_idx] = True
        print(f"[CV {fold_id}/{n_folds}]")

        for K in Ks:
            print(f"  K={K}", end="")

            mdata_train = mdata[~mask].copy()
            mdata_test  = mdata[mask].copy()

            _preprocess(mdata_train)
            _preprocess(mdata_test)

            model_train = _build_model(mdata_train, K, n_jobs, is_spatial, spatial_key)
            model_test  = _build_model(mdata_test,  K, n_jobs, is_spatial, spatial_key)

            t0 = time.perf_counter()
            model_train.VI(n_iter=n_iter)
            train_time_sec = time.perf_counter() - t0

            lambda_train = {mod: model_train.lambda_[mod].copy()
                            for mod in model_train.modalities}
            model_test._predict_theta(lambda_train)

            theta_test = _extract_theta(model_test, mdata_test, K)
            beta       = _extract_beta(lambda_train, mdata_test, K)
            ll_total, ll_per_mod = _log_likelihood(model_test.X, theta_test, beta)

            result_rows.append({
                "K": K, "CV": fold_id,
                "train_time_sec": train_time_sec,
                "total": ll_total,
                **{mod: ll_per_mod[mod] for mod in modality_names},
            })
            print(f"\tLL={ll_total:.2f}\tt={train_time_sec:.1f}s")

    results_df = pd.DataFrame(result_rows)

    total_rows = [{
        "K": K, "CV": "total",
        "train_time_sec": results_df[results_df["K"] == K]["train_time_sec"].sum(),
        "total":          results_df[results_df["K"] == K]["total"].mean(),
        **{mod: results_df[results_df["K"] == K][mod].mean() for mod in modality_names},
    } for K in Ks]

    final_df = pd.concat([results_df, pd.DataFrame(total_rows)], ignore_index=True)
    mdata.uns["CV_results"] = final_df
