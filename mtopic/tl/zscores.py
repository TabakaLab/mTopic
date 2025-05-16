import scanpy as sc
import muon as mu
import numpy as np
import pandas as pd


mu.set_options(pull_on_update=False)


def _zscores_mod(mdata,
                 mod, 
                 signatures,
                 raw_data_path,
                 n_top, thr, 
                 out_key):
    norm_counts = mu.read_h5mu(raw_data_path)[mod]
    sc.pp.normalize_total(norm_counts, target_sum=10000)
    sc.pp.log1p(norm_counts)

    row_means = np.asarray(norm_counts.X.mean(axis=1)).flatten()
    row_stds = np.asarray([np.std(norm_counts.X.getrow(i).toarray()) for i in range(mdata.n_obs)])
    row_stds[row_stds == 0] = 1
    
    scores = list()
    for i, t in enumerate(mdata[mod].varm[signatures].columns):
        top = mdata[mod].varm[signatures][t].sort_values(ascending=True).iloc[-n_top:].index.tolist()
        s = pd.DataFrame(norm_counts[:, top].X.toarray(), index=mdata.obs.index, columns=top)
        s = s.subtract(row_means, axis=0)
        s = s.divide(row_stds, axis=0)
        s = np.minimum(s, thr)
        s = np.maximum(s, -thr)
        scores.append(np.mean(s, axis=1))
        
    mdata[mod].obsm[out_key] = pd.DataFrame(np.asarray(scores).T, index=mdata.obs.index, columns=mdata[mod].varm[signatures].columns)
    

def zscores(mdata, 
            raw_data_path, 
            signatures='signatures',
            mod=None, 
            n_top=10, 
            thr=5, 
            out_key='zscores'):
    """
    Compute z-scores for the top features in each topic within a MuData object.

    This function calculates z-scores for the top features associated with each topic in the specified 
    modality or across all modalities of a `MuData` object. Z-scores are computed using normalized and 
    log-transformed raw count data, allowing for a standardized comparison of feature expression levels 
    relative to their mean and standard deviation across all cells. Computed z-scores are capped within a 
    specified threshold range to limit extreme values.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data.
    :type mdata: muon.MuData
    :param raw_data_path: 
        Path to the `.h5mu` file containing the raw count data for normalization and z-score computation.
    :type raw_data_path: str
    :param signatures: 
        Key in the `varm` attribute of each modality representing the topic signatures to compute z-scores for. 
        Default is 'signatures'.
    :type signatures: str, optional
    :param mod: 
        Specific modality to compute z-scores for. If None, z-scores are computed for all modalities. Default is None.
    :type mod: str, optional
    :param n_top: 
        Number of top features to select for each topic based on their importance in the topic signature. 
        Default is 10.
    :type n_top: int, optional
    :param thr: 
        Threshold to cap the computed z-scores. Z-scores will be limited to the range [-thr, thr]. Default is 5.
    :type thr: float, optional
    :param out_key: 
        Key under which the computed z-scores will be stored in the `obsm` attribute of each modality. 
        Default is 'zscores'.
    :type out_key: str, optional

    :returns: 
        None

    :updates:
        - `mdata[mod].obsm[out_key]`: 
          A DataFrame containing the z-scores for the top features of each topic in the specified modality 
          or all modalities if `mod` is None.

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Compute z-scores for the top 10 features in each topic for all modalities
            mtopic.pp.zscores(
                mdata, 
                signatures='signatures', 
                raw_data_path="path/to/raw/data.h5mu"
            )

            # Compute z-scores for a specific modality ('rna')
            mtopic.pp.zscores(
                mdata, 
                signatures='signatures', 
                raw_data_path="path/to/raw/data.h5mu", 
                mod='rna'
            )

    :notes:
        - **Z-Score Calculation**: 
          Z-scores are computed as `(x - mean) / std`, where `x` is the log-transformed expression value of a feature, 
          `mean` is the mean across all cells, and `std` is the standard deviation across all cells.
        - **Feature Selection**: 
          The top `n_top` features for each topic are selected based on their importance in the topic signatures 
          (highest weights).
        - **Thresholding**: 
          Extreme z-scores are capped to the range [-thr, thr] to mitigate the impact of outliers.
        - The raw count data for normalization is loaded from `raw_data_path`.
    """

    if mod is None:
        for m in mdata.mod:
            _zscores_mod(mdata, 
                         signatures=signatures, 
                         raw_data_path=raw_data_path, 
                         mod=m, 
                         n_top=n_top, 
                         thr=thr,
                         out_key=out_key)
    else:
        _zscores_mod(mdata, 
                     signatures=signatures, 
                     raw_data_path=raw_data_path, 
                     mod=mod, 
                     n_top=n_top, 
                     thr=thr,
                     out_key=out_key)
