import numpy as np
import scipy.sparse as sp
import muon as mu

mu.set_options(pull_on_update=False)


def _log1p_inplace_sparse(X):
    """
    In-place log1p on a sparse matrix data array. Returns X for convenience.
    """
    if X.nnz:
        X.data = np.log1p(X.data)
    return X


def clr(mdata,
        mod,
        copy=False,
        from_layer=None,
        create_counts_layer=True,
        counts_layer="counts",
        pseudocount=1e-6):
    """
    Perform Centered Log-Ratio (CLR) normalization on a modality in a MuData object.

    This function applies CLR normalization to the specified modality within a `MuData` object. CLR normalization
    is widely used for compositional data, such as single-cell protein counts. It calculates a per-cell scaling
    factor derived from the mean of `log1p(counts)` across features, rescales counts by this factor, and applies a
    log transformation.

    By default, this function creates `layers["counts"]` (if it does not exist) to preserve raw counts, and then
    overwrites the modality `.X` matrix with CLR-transformed values.

    :param mdata:
        A `MuData` object containing multimodal single-cell data.
    :type mdata: muon.MuData
    :param mod:
        The modality to which CLR normalization will be applied (e.g., 'prot').
    :type mod: str
    :param copy:
        If True, creates a copy of the `MuData` object, and the normalization is applied to the copy.
        If False, the normalization is performed in-place on the original object. Default is False.
    :type copy: bool, optional
    :param from_layer:
        Optional layer name used as input for CLR. If None, uses `.X` as input. Default is None.
        Note: regardless of `from_layer`, the CLR-transformed matrix is written to `.X`.
    :type from_layer: str, optional
    :param create_counts_layer:
        If True and `layers[counts_layer]` does not exist, store a copy of the current `.X` matrix in
        `layers[counts_layer]` before overwriting `.X`. Default is True.
    :type create_counts_layer: bool, optional
    :param counts_layer:
        Name of the layer used to store raw counts. Default is `"counts"`.
    :type counts_layer: str, optional
    :param pseudocount:
        Small value added to the per-cell scaling factor to avoid division by zero. Default is `1e-6`.
    :type pseudocount: float, optional

    :returns:
        If `copy` is True, returns a new `MuData` object with CLR-transformed `.X`.
        If `copy` is False, returns None and modifies the object in-place.
    :rtype: muon.MuData or None

    :example:

        .. code-block:: python

            import mtopic

            # Load a MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # CLR-transform protein counts in-place (stores raw counts in layers["counts"])
            mtopic.pp.clr(mdata, mod="prot")

            # CLR-transform using a layer as input, but still write output to .X
            mtopic.pp.clr(mdata, mod="prot", from_layer="counts")

            # Return a copy
            mdata_clr = mtopic.pp.clr(mdata, mod="prot", copy=True)
    """
    assert isinstance(mdata, mu.MuData)

    if copy:
        mdata = mdata.copy()

    adata = mdata.mod[mod]

    if create_counts_layer and counts_layer not in adata.layers:
        adata.layers[counts_layer] = adata.X.copy()

    if from_layer is None:
        X_in = adata.X
    else:
        if from_layer not in adata.layers:
            raise KeyError(f"Layer '{from_layer}' not found in mdata.mod['{mod}'].layers.")
        X_in = adata.layers[from_layer]

    n_features = X_in.shape[1]

    if sp.issparse(X_in):
        X_log = X_in.copy().tocsc()
        _log1p_inplace_sparse(X_log)
        mean_log1p = (X_log.sum(axis=1).A1) / float(n_features)
    else:
        mean_log1p = np.log1p(np.asarray(X_in)).sum(axis=1) / float(n_features)

    row_sum = np.expm1(mean_log1p) + float(pseudocount)
    inv_gm = 1.0 / row_sum

    if sp.issparse(X_in):
        X_scaled = X_in.tocsc(copy=True).multiply(inv_gm[:, None]).tocsr()
        _log1p_inplace_sparse(X_scaled)
        adata.X = X_scaled
    else:
        X_scaled = np.asarray(X_in) * inv_gm[:, None]
        adata.X = np.log1p(X_scaled)

    if copy:
        return mdata