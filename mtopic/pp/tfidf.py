import muon
from sklearn.feature_extraction.text import TfidfTransformer
import anndata as ad


muon.set_options(pull_on_update=False)


def tfidf(mdata,
          mod, 
          copy=False):
    """
    Apply Term Frequency-Inverse Document Frequency (TF-IDF) transformation to a specific modality in a MuData object.

    This function performs a TF-IDF transformation on the `.X` matrix of the specified modality within the provided 
    `MuData` object. The TF-IDF transformation adjusts raw counts by considering both term frequency (TF) and inverse 
    document frequency (IDF). This enhances interpretability by down-weighting common features and emphasizing rare ones. 
    The transformation is applied in-place by default or to a copy if specified.

    :param mdata: 
        A `MuData` object containing multiple modalities, each with an `.X` matrix to be transformed.
    :type mdata: muon.MuData
    :param mod: 
        The modality to apply the TF-IDF transformation to (e.g., 'rna', 'protein').
    :type mod: str
    :param copy: 
        If True, the TF-IDF transformation is applied to a copy of the `MuData` object, leaving the original unchanged. 
        If False, the transformation is applied in-place. Default is False.
    :type copy: bool, optional

    :returns: 
        If `copy` is True, returns a new `MuData` object with the TF-IDF-transformed data. 
        If `copy` is False, the transformation is applied in-place, and None is returned.
    :rtype: muon.MuData or None

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Apply TF-IDF transformation in-place for the 'rna' modality
            mtopic.pp.tfidf(mdata, mod='rna')

            # Apply TF-IDF transformation for the 'rna' modality and return a copy
            transformed_mdata = mtopic.pp.tfidf(mdata, mod='rna', copy=True)

    :notes:
        - The `TfidfTransformer` from `sklearn` is used with `norm=None` and `smooth_idf=False`, meaning the output will 
          not be normalized, and inverse document frequency will not be smoothed.
        - This function assumes the `.X` matrix of the specified modality is compatible with `TfidfTransformer`.
        - Ideal for preprocessing single-cell data with modalities like RNA or ATAC.

    """
    
    assert isinstance(mdata, muon.MuData)
    
    if copy:
        mdata = mdata.copy()

    if not isinstance(mdata.mod[mod].raw, ad._core.raw.Raw):
        mdata.mod[mod].raw = mdata.mod[mod].copy()

    transformer = TfidfTransformer(norm=None, smooth_idf=False)
    mdata[mod].X = transformer.fit_transform(mdata[mod].X)

    if copy:
        return mdata
