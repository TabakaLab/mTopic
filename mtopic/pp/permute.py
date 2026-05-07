import muon
import numpy
import scipy
import scanpy


muon.set_options(pull_on_update=False)


def permute(mdata,
            subset: int = None,
            seed=2291,
            copy=False):
    """
    Randomly permute the count matrices in a MuData object.

    This function permutes each modality's `.X` matrix along the observation 
    (cell/spot) axis, independently per modality. It is used to generate a 
    control dataset for detecting pervasive features — features that dominate 
    feature–topic distributions despite randomized input. Basic filtering is 
    applied to retain only non-empty cells and features.

    :param mdata: 
        A `MuData` object containing multiple modalities, each with an `.X` attribute representing the counts to be permuted.
    :type mdata: muon.MuData
    :param subset: 
        The number of cells (observations) to randomly subset before performing permutation. If None, all cells are used. 
        Default is None.
    :type subset: int, optional
    :param seed: 
        Seed for the random number generator to ensure reproducibility of the permutation. Default is 2291.
    :type seed: int, optional
    :param copy: 
        If True, creates a copy of the `MuData` object and performs permutation on the copy. If False, the operation 
        is performed in-place. Default is False.
    :type copy: bool, optional

    :returns: 
        If `copy` is True, returns a new `MuData` object with permuted data. If `copy` is False, returns None 
        and applies permutation directly to the input `MuData` object.
    :rtype: muon.MuData or None

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Permute data in-place
            mtopic.pp.permute(mdata)

            # Permute data and return a copy
            permuted_mdata = mtopic.pp.permute(mdata, copy=True)
    """
    
    assert isinstance(mdata, muon.MuData)

    rng = numpy.random.default_rng(seed=seed)
    
    if copy:
        mdata = mdata.copy()

    if subset is not None:
        mdata = mdata[rng.choice(mdata.n_obs, subset, replace=False)]
    
    for mod in mdata.mod:
        X = rng.permutation(mdata[mod].X.toarray(), axis=0)
        X = scipy.sparse.csr_matrix(X)
        X.eliminate_zeros()

        mdata.mod[mod].X = X
        scanpy.pp.filter_cells(mdata[mod], min_counts=1)
        scanpy.pp.filter_genes(mdata[mod], min_counts=1)

    muon.pp.intersect_obs(mdata)
    mdata.update()

    if copy:
        return mdata
