import torch


def row_normalize(matrix):
    return matrix / matrix.sum(dim=1, keepdim=True)
    

def feature_associations_data(mdata, 
                              mod_list, 
                              normalize=True):
    """
    Prepare topic signatures for training feature associations.

    Extracts and optionally row-normalizes signature matrices from multiple modalities 
    in a MuData object. Each signature matrix is expected to be stored in 
    `varm['signatures']` of the respective modality. Matrices are returned in a 
    dictionary keyed by modality name.

    :param mdata: A MuData object containing modalities with signature matrices in `varm['signatures']`.
    :type mdata: muon.MuData

    :param mod_list: List of modality names to extract from `mdata`.
    :type mod_list: list[str]

    :param normalize: If `True`, rows of the signature matrices are normalized to sum to 1. (default: `True`)
    :type normalize: bool

    :returns: A dictionary where keys are modality names and values are tuples `(M, M_var)`, with:
              - `M`: torch.FloatTensor of shape `(n_topics, n_features)`, the signature matrix.
              - `M_var`: The corresponding `var` DataFrame from the modality.
    :rtype: dict[str, tuple[torch.Tensor, pandas.DataFrame]]

    :example:

        .. code-block:: python

            import muon as mu
            import torch

            mdata = mu.read('multiome_data.h5mu')

            # Extract signatures from RNA and ATAC modalities
            associations_input = feature_associations_data(mdata, ['rna', 'atac'])

            A, A_var = associations_input['rna']
            B, B_var = associations_input['atac']
    """

    output = dict()

    for mod in mod_list:
        M = torch.tensor(mdata[mod].varm['signatures'].T.values, dtype=torch.float32)
        
        if normalize:
            M = row_normalize(M) 
    
        M_var = mdata[mod].var

        output[mod] = (M, M_var)

    return output
