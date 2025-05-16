import pandas as pd


def export_params(model,
                  mdata, 
                  prefix=None,
                  filter_topics=True,
                  filter_threshold=0.01, 
                  normalize=True):
    """
    Export model parameters (gamma and lambda) to a MuData object and filter insignificant topics.

    This function exports topic proportions (gamma) and topic-feature distributions (lambda) from the given model 
    into the `obsm` and `varm` attributes of a `MuData` object. Users can apply optional filtering to remove 
    topics with proportions below a specified threshold, which helps focus on meaningful patterns.

    :param model: 
        An instance of `mtopic.tl.MTM` or `mtopic.tl.spatialMTM` containing the parameters (`gamma` and `lambda`) 
        to be exported.
    :type model: mtopic.tl.MTM or mtopic.tl.spatialMTM
    :param mdata: 
        A `MuData` object to which the parameters will be exported.
    :type mdata: muon.MuData
    :param prefix: 
        A string prefix for keys under which the parameters will be stored in `MuData`. 
        If None, no prefix is added. Default is None.
    :type prefix: str, optional
    :param filter_topics: 
        Whether to filter topics based on their maximum normalized proportion across cells. If True, only topics 
        with a maximum proportion above `filter_threshold` are retained. Default is True.
    :type filter_topics: bool, optional
    :param filter_threshold: 
        The threshold for filtering topics when `filter_topics` is True. Topics with a maximum normalized 
        proportion below this threshold are removed. Default is 0.01.
    :type filter_threshold: float, optional
    :param normalize: 
        Whether to normalize the topic proportions (`gamma`). If True, normalizes rows of `gamma` so that 
        each row sums to 1. Default is True.
    :type normalize: bool, optional

    :returns: 
        None

    :updates:
        - `mdata.obsm['{prefix}_topics']`: A DataFrame containing topic proportions for each sample, optionally filtered.
        - `mdata[modality].varm['{prefix}_signatures']`: A DataFrame containing topic-feature distributions for 
          each modality, filtered to include only selected topics.

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Initialize MTM model
            model = mtopic.tl.MTM(mdata, n_topics=20)

            # Fit the model using Variational Inference
            model.VI(n_iter=20)

            # Export model parameters to MuData object
            mtopic.pp.export_params(model, mdata)

            # Access exported parameters
            print(mdata.obsm['topics'])  # Topic proportions
            print(mdata['rna'].varm['signatures'])  # Topic-feature distributions for 'rna' modality

    :notes:
        - **Gamma (Topic Proportions)**: Stored in `obsm` as `{prefix}_topics`. Each row corresponds to a sample, 
          and each column represents a topic.
        - **Lambda (Topic-Feature Distributions)**: Stored in `varm` for each modality as `{prefix}_signatures`. 
          Each column corresponds to a topic, and each row represents a feature.
        - If `prefix` is None, parameters are stored with keys `'topics'` and `'signatures'`.
    """

    gamma = pd.DataFrame(model.gamma / model.gamma.sum(axis=1, keepdims=True) if normalize else model.gamma,
                         index=model.barcodes, 
                         columns=['topic_{}'.format(i+1) for i in range(model.n_topics)]).loc[mdata.obs.index]

    if filter_topics:
        filtered_topics = gamma.loc[:, gamma.max() > filter_threshold].columns.tolist()
        gamma = gamma.loc[:, filtered_topics]
        gamma.div(gamma.sum(axis=1), axis=0)
        
    else:
        filtered_topics = gamma.columns.tolist()

    topics_key = '_'.join([prefix, 'topics']) if isinstance(prefix, str) else 'topics'
    mdata.obsm[topics_key] = gamma

    signatures_key = '_'.join([prefix, 'signatures']) if isinstance(prefix, str) else 'signatures'
    for m in model.modalities:
        mdata[m].varm[signatures_key] = pd.DataFrame(model.lambda_[m], 
                                                     index=['topic_{}'.format(i+1) for i in range(model.n_topics)],
                                                     columns=model.features[m]).T.loc[mdata[m].var.index, filtered_topics]
