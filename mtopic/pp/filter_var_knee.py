import muon as mu
import kneed
import numpy as np
import pandas as pd
from mtopic.read.h5mu import h5mu


mu.set_options(pull_on_update=False)


def filter_var_knee(path, 
                    model, 
                    knee_sensitivity=5):
    """
    Filter overrepresented features from a MuData object using a knee detection algorithm.

    This function identifies and removes overrepresented features (e.g., genes, proteins) across all topics in each 
    modality of a `MuData` object using a knee detection algorithm. Overrepresented features, which are beyond a 
    significant drop-off point (knee point) in their cumulative feature score, are filtered out to improve data quality 
    and downstream analysis.

    :param path: 
        The file path to the `.h5mu` file containing the `MuData` object to be processed.
    :type path: str
    :param model: 
        An instance of a topic model containing the topic-feature distributions (e.g., `lambda_` matrix for each modality).
    :type model: mtopic.tl.MTM or mtopic.tl.sMTM
    :param knee_sensitivity: 
        Sensitivity for the knee detection algorithm. Higher values make the algorithm more conservative in identifying 
        overrepresented features. It can be a single integer (global for all modalities) or a dictionary specifying 
        sensitivity per modality. Default is 5.
    :type knee_sensitivity: int or dict, optional

    :returns: 
        A `MuData` object with overrepresented features removed.
    :rtype: muon.MuData

    :raises FileNotFoundError: 
        If the specified `.h5mu` file does not exist or is inaccessible.
    :raises ValueError: 
        If `knee_sensitivity` is invalid or features cannot be identified for filtering.

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object and model
            mdata = mtopic.read.h5mu("path/to/file.h5mu")
            model = mtopic.tl.MTM(mdata, n_topics=20)

            # Filter overrepresented features
            filtered_mdata = mtopic.pp.filter_var_knee("path/to/file.h5mu", model)

    :notes:
        - **Feature Identification**:
          Overrepresented features are identified by calculating their cumulative feature score across all topics in a modality. 
          The knee detection algorithm (`kneed`) detects the knee point, beyond which features are considered overrepresented.
        - **Knee Sensitivity**:
          The `knee_sensitivity` parameter can be set globally for all modalities or specified individually for each modality 
          as a dictionary. This allows flexibility based on the characteristics of each modality.
        - **Data Consistency**:
          After filtering, the `mdata.update()` method ensures consistency across the multimodal data structure.
        - **Applicability**:
          This approach is ideal for filtering features that dominate topic distributions, which may obscure meaningful patterns.
    """
    
    # Load MuData object
    mdata = h5mu(path)

    # Convert knee_sensitivity to a dictionary if it is not one already
    if not isinstance(knee_sensitivity, dict):
        knee_sensitivity = {m: knee_sensitivity for m in mdata.mod}

    # Initialize an empty list to collect features to be filtered out
    var = []

    # Loop through each modality in the MuData object
    for m in mdata.mod:
        if knee_sensitivity[m] > 0:
            # Calculate cumulative feature score across topics
            sums = pd.DataFrame(model.lambda_[m], columns=model.features[m])
            sums = sums.div(sums.sum(axis=1), axis=0)
            sums = sums.sum(axis=0).sort_values(ascending=False)
            sums = sums / np.max(sums)

            # Apply the knee detection algorithm to find the point beyond which features are overrepresented
            kneedle = kneed.KneeLocator(
                range(len(sums)), 
                sums.values, 
                curve='convex', 
                direction='decreasing', 
                S=knee_sensitivity[m]
            )

            # Collect features beyond the knee point for filtering
            if kneedle.knee is not None:
                var += sums.iloc[kneedle.knee:].index.tolist()
        else:
            # If knee sensitivity is not positive, add all features for removal
            var += list(model.features[m])

    # Filter the overrepresented features from the MuData object
    mu.pp.filter_var(mdata, var=var)
    mdata.update()

    return mdata
