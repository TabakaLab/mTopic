import muon as mu
from sklearn.feature_extraction.text import TfidfTransformer
import anndata as ad

mu.set_options(pull_on_update=False)


def tfidf(mdata,
          mod,
          copy=False,
          from_layer=None,
          create_counts_layer=True,
          counts_layer="counts"):
    """
    Apply Term Frequency-Inverse Document Frequency (TF-IDF) transformation to a specific modality in a MuData object.

    This function performs a TF-IDF transformation on the input matrix of the specified modality within the provided
    `MuData` object. The TF-IDF transformation adjusts raw counts by considering both term frequency (TF) and inverse
    document frequency (IDF). This enhances interpretability by down-weighting common features and emphasizing rare ones.

    By default, this function creates `layers["counts"]` (if it does not exist) to preserve raw counts, and then
    overwrites the modality `.X` matrix with TF-IDF-transformed values.

    :param mdata:
        A `MuData` object containing multiple modalities, each with an `.X` matrix to be transformed.
    :type mdata: muon.MuData
    :param mod:
        The modality to apply the TF-IDF transformation to (e.g., 'rna', 'atac').
    :type mod: str
    :param copy:
        If True, the TF-IDF transformation is applied to a copy of the `MuData` object, leaving the original unchanged.
        If False, the transformation is applied in-place. Default is False.
    :type copy: bool, optional
    :param from_layer:
        Optional layer name used as input for TF-IDF. If None, uses `.X` as input. Default is None.
        Note: regardless of `from_layer`, the TF-IDF-transformed matrix is written to `.X`.
    :type from_layer: str, optional
    :param create_counts_layer:
        If True and `layers[counts_layer]` does not exist, store a copy of the current `.X` matrix in
        `layers[counts_layer]` before overwriting `.X`. Default is True.
    :type create_counts_layer: bool, optional
    :param counts_layer:
        Name of the layer used to store raw counts. Default is `"counts"`.
    :type counts_layer: str, optional

    :returns:
        If `copy` is True, returns a new `MuData` object with TF-IDF-transformed data in `.X`.
        If `copy` is False, the transformation is applied in-place, and None is returned.
    :rtype: muon.MuData or None

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Apply TF-IDF in-place (stores raw counts in layers["counts"])
            mtopic.pp.tfidf(mdata, mod="atac")

            # Apply TF-IDF using a specific layer as input, but still write output to .X
            mtopic.pp.tfidf(mdata, mod="atac", from_layer="counts")

            # Return a copy
            mdata_tfidf = mtopic.pp.tfidf(mdata, mod="atac", copy=True)
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

    transformer = TfidfTransformer(norm=None, smooth_idf=False)
    adata.X = transformer.fit_transform(X_in)

    if copy:
        return mdata
