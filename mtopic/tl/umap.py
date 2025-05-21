from umap import UMAP
import pandas as pd


def umap(mdata,
         x='topics', 
         umap='umap', 
         n_components=2, 
         min_dist=0.1, 
         n_neighbors=20, 
         seed=2291,
         n_jobs=10):
    """
    Perform UMAP dimensionality reduction on topic distributions.

    This function applies Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality 
    of topic distributions stored in the `obsm` attribute of a `MuData` object. The reduced dimensions are stored 
    in the `obsm` attribute under a specified name.

    :param mdata: 
        A `MuData` object containing the topic distributions in the `obsm` attribute.
    :type mdata: muon.MuData
    :param x: 
        The key in the `obsm` attribute of `mdata` that holds the topic distributions to be used for UMAP. 
        Default is 'topics'.
    :type x: str, optional
    :param umap: 
        The key under which the UMAP results will be stored in the `obsm` attribute of `mdata`. 
        Default is 'umap'.
    :type umap: str, optional
    :param n_components: 
        The number of dimensions for the UMAP embedding. Default is 2.
    :type n_components: int, optional
    :param min_dist: 
        The minimum distance between points in the UMAP embedding. Controls the balance between local 
        and global structure. Default is 0.1.
    :type min_dist: float, optional
    :param n_neighbors: 
        The number of nearest neighbors to consider when computing the UMAP embedding. Default is 20.
    :type n_neighbors: int, optional
    :param seed: 
        Random seed for reproducibility. Ensures consistent embeddings across runs. Default is 2291.
    :type seed: int, optional
    :param n_jobs: 
        Number of CPU cores to use for parallel computation. If set to -1, all available cores are used. Default is 10.
    :type n_jobs: int, optional

    :returns: 
        None

    :updates:
        - `mdata.obsm[umap]`: A DataFrame containing the UMAP coordinates for each sample, with dimensions 
          specified by `n_components`.

    :example:

        .. code-block:: python
    
            import mtopic
            
            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")
            
            # Compute UMAP embedding for topic distributions
            mtopic.pp.umap(mdata, x='topics', n_components=2)
    """

    red = UMAP(n_components=n_components,
               min_dist=min_dist,
               n_neighbors=n_neighbors,
               random_state=seed, 
               n_jobs=n_jobs)

    mdata.obsm[umap] = pd.DataFrame(red.fit_transform(mdata.obsm[x]),
                                    index=mdata.obs.index,
                                    columns=[f'umap_{i+1}' for i in range(n_components)])
