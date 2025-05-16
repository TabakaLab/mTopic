import muon
import numpy
import scipy
import scanpy


muon.set_options(pull_on_update=False)


def _permute_csr(matrix, 
                 rng):
    """
    Randomly permute the row indices of each column in a sparse matrix.

    This function permutes the row indices of non-zero entries within each column of a sparse matrix in CSR or CSC format. 
    It ensures that the column structure of the matrix is preserved while shuffling the data within columns. The result is 
    returned as a sparse matrix in CSR format.

    :param matrix: 
        The sparse matrix (in CSR or CSC format) whose row indices are to be permuted.
    :type matrix: scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
    :param rng: 
        A random number generator used to perform the permutations.
    :type rng: numpy.random.Generator

    :returns: 
        A permuted sparse matrix in CSR format.
    :rtype: scipy.sparse.csr_matrix

    :raises ValueError: 
        If the input matrix is not in CSR or CSC sparse format.

    :example:

        .. code-block:: python

            import numpy as np
            from scipy.sparse import csr_matrix
            from mtopic.pp import _permute_csr

            # Create example sparse matrix
            matrix = csr_matrix([[0, 1, 0], [3, 0, 4], [0, 5, 6]])

            # Initialize random number generator
            rng = np.random.default_rng(seed=42)

            # Permute the matrix
            permuted_matrix = _permute_csr(matrix, rng)
    """
    if not scipy.sparse.isspmatrix_csr(matrix) and not scipy.sparse.isspmatrix_csc(matrix):
        raise ValueError("Input matrix must be in CSR or CSC sparse format.")

    matrix = matrix.tocsc() 

    permuted_matrix = scipy.sparse.lil_matrix(matrix.shape)

    for col in range(matrix.shape[1]):
        start_idx, end_idx = matrix.indptr[col], matrix.indptr[col + 1]
        if start_idx < end_idx:
            row_indices = matrix.indices[start_idx:end_idx]
            data = matrix.data[start_idx:end_idx]
            permuted_indices = rng.permutation(row_indices)
            permuted_matrix[permuted_indices, col] = data

    permuted_matrix = permuted_matrix.tocsr()

    return permuted_matrix.astype(numpy.float32)


def permute(mdata,
            subset: int = None,
            seed=2291,
            copy=False, 
            sparse_mode=False):
    """
    Randomly permute the count matrices in a MuData object.

    This function randomly permutes the counts within each column of the `.X` matrices in each modality of a 
    `MuData` object. Permutations are performed either on dense matrices or sparse matrices based on the 
    `sparse_mode` parameter. Basic filtering is applied to retain only non-empty cells and features.

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
    :param sparse_mode: 
        If False, performs permutation on dense matrices (after converting to dense). Default is False.
    :type sparse_mode: bool, optional

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

    :notes:
        - This function permutes the counts within each column of the `.X` matrices for all modalities in the `MuData` object.
        - Filtering is applied using `scanpy.pp.filter_cells()` and `scanpy.pp.filter_genes()` to remove empty cells and features.
        - The function uses `muon.pp.intersect_obs()` to ensure consistency across modalities after filtering.
        - The `mdata.update()` method is called to synchronize the `MuData` object after modifications.
    """
    
    assert isinstance(mdata, muon.MuData)

    rng = numpy.random.default_rng(seed=seed)
    
    if copy:
        mdata = mdata.copy()

    if subset is not None:
        mdata = mdata[rng.choice(mdata.n_obs, subset, replace=False)]
    
    for mod in mdata.mod:
        if sparse_mode:
            X = _permute_csr(mdata[mod].X, rng)
        else:
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
