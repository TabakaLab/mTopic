import muon as mu
from mtopic.read.h5mu import h5mu


mu.set_options(pull_on_update=False)


def filter_var_list(path, 
                    var):
    """
    Retain a specific list of features in a MuData object.

    This function retains only the specified list of features (e.g., genes, proteins) in a `MuData` object 
    and removes all other features. It is designed to streamline downstream analysis by focusing on a 
    predefined subset of relevant features.

    :param path: 
        The file path to the `.h5mu` file containing the `MuData` object to be processed.
    :type path: str
    :param var: 
        A list of feature names to be retained in the `MuData` object.
    :type var: list of str

    :returns: 
        A `MuData` object containing only the specified features.
    :rtype: muon.MuData

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object and specify features to retain
            path = "path/to/file.h5mu"
            features_to_keep = ['gene1', 'gene2', 'gene3']

            # Retain specified features
            filtered_mdata = mtopic.pp.filter_var_list(path, var=features_to_keep)

    :notes:
        - The function loads the `MuData` object from the specified path using `mtopic.read.h5mu` and 
          retains only the features specified in the `var` list using `mu.pp.filter_var`.
        - After filtering, the `mdata.update()` method ensures consistency across the multimodal data structure.
        - Ensure the feature names in `var` match the names in the dataset to avoid errors.
    """
    
    mdata = h5mu(path)
    mu.pp.filter_var(mdata, var=var)
    mdata.update()

    return mdata
