import numpy as np
from scipy.sparse import csr_matrix
import muon as mu
import anndata as ad
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='anndata._core.storage')


mu.set_options(pull_on_update=False)


def clr(mdata,
        mod,
        copy=False):
    """
    Perform Centered Log-Ratio (CLR) normalization on a modality in a MuData object.

    This function applies CLR normalization to the specified modality within a `MuData` object. CLR normalization 
    is widely used for compositional data, such as single-cell protein counts. It calculates the geometric mean of 
    counts for each cell, normalizes each feature by dividing its count by the geometric mean, and applies a log 
    transformation. This normalization helps mitigate the effects of varying library sizes and technical biases.

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

    :returns: 
        If `copy` is True, returns a new `MuData` object with CLR-normalized data. 
        If `copy` is False, returns None, and the normalization is applied in-place.
    :rtype: muon.MuData or None

    :raises AssertionError: 
        If the input `mdata` is not an instance of `muon.MuData`.

    :example:

        .. code-block:: python
        
            import mtopic

            # Load a MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Apply CLR normalization in-place to the 'prot' modality
            mtopic.pp.clr(mdata, mod='prot')

            # Apply CLR normalization to the 'prot' modality and return a copy
            normalized_mdata = mtopic.pp.clr(mdata, mod='prot', copy=True)

    :notes:
        - The CLR normalization follows these steps:
          1. Compute the geometric mean of counts for each cell (row) across all features (columns).
          2. Divide each feature count by the geometric mean of its respective cell.
          3. Apply a log1p transformation (log(x + 1)) to the normalized data.
        - A small pseudocount (1e-6) is added to the geometric mean to avoid division by zero errors.
    """

    assert isinstance(mdata, mu.MuData)
    
    if copy:
        mdata = mdata.copy()

    if not isinstance(mdata.mod[mod].raw, ad._core.raw.Raw):
        mdata.mod[mod].raw = mdata.mod[mod].copy()

    mdata[mod].X = mdata[mod].X.tocsc()
    row_sum = np.expm1(mdata[mod].X.log1p().sum(axis=1).A1 / mdata[mod].X.shape[1])
    
    pseudocount = 1e-6
    row_sum += pseudocount

    inverse_geometric_mean = 1 / row_sum

    mdata[mod].X = mdata[mod].X.multiply(inverse_geometric_mean[:, np.newaxis])
    mdata[mod].X = mdata[mod].X.log1p()
    mdata[mod].X = mdata[mod].X.tocsr()

    if copy:
        return mdata
