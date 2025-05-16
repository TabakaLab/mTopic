import muon
import anndata as ad


muon.set_options(pull_on_update=False)


def scale_counts(mdata,
                 counts_per_cell=10000, 
                 copy=False):
    """
    Scale count matrices to normalize the total sum of counts across modalities.

    This function normalizes the total sum of counts in the `.X` matrix of each modality within the provided `MuData` object. 
    Each modality's counts are scaled so that their total sum matches a specified target value (`counts_per_cell * n_obs`). 
    This ensures consistency in total counts across different modalities, which is crucial for comparative and integrative 
    analyses in single-cell multimodal datasets.

    :param mdata: 
        A `MuData` object containing multiple modalities, each with an `.X` attribute representing counts data.
    :type mdata: muon.MuData
    :param counts_per_cell: 
        The target average counts per cell. The total sum of counts in each modality is scaled to 
        `counts_per_cell * n_obs`, where `n_obs` is the number of cells. Default is 10,000.
    :type counts_per_cell: int, optional
    :param copy: 
        If True, a copy of the `MuData` object is created, and the scaling is applied to the copy. 
        If False, the scaling is performed in-place on the original object. Default is False.
    :type copy: bool, optional

    :returns: 
        If `copy` is True, returns a new `MuData` object with scaled counts data. 
        If `copy` is False, returns None and applies scaling directly to the input `MuData` object.
    :rtype: muon.MuData or None

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Scale counts in-place
            mtopic.pp.scale_counts(mdata)

            # Scale counts and return a copy
            scaled_mdata = mtopic.pp.scale_counts(mdata, copy=True)

    :notes:
        - This function is especially useful when working with multimodal datasets where each modality may have 
          different total counts, making comparisons challenging.
    """

    assert isinstance(mdata, muon.MuData)
    
    if copy:
        mdata = mdata.copy()

    for mod in mdata.mod:
        if not isinstance(mdata.mod[mod].raw, ad._core.raw.Raw):
            mdata.mod[mod].raw = mdata.mod[mod].copy()
        
        mdata[mod].X = mdata[mod].X * (mdata.n_obs * counts_per_cell / mdata[mod].X.sum())

    if copy:
        return mdata
