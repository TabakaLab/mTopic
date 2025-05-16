import muon


muon.set_options(pull_on_update=False)


def h5mu(path):
    """
    Load a MuData object from an `.h5mu` file.

    This function reads a MuData object from the specified `.h5mu` file using the `muon` library. 
    The `.h5mu` format is specifically designed for storing multimodal single-cell data, enabling 
    efficient handling of large datasets across multiple modalities in a unified structure.

    :param path: The file path to the `.h5mu` file containing the MuData object.
    :type path: str

    :returns: 
        A MuData object loaded from the specified file.
    :rtype: muon.MuData

    :example:

        .. code-block:: python

            import mtopic

            # Load the MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

    :notes:
        - The `.h5mu` format is optimized for multimodal data, such as single-cell RNA-seq combined with 
          protein expression or ATAC-seq. It supports integrated analyses and visualizations.
        - Verify the file path to avoid errors during loading.
    """
    
    return muon.read_h5mu(path)
