import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ._utils import optimal_grid, savefig


def feature_activity(mdata, 
                     x, 
                     features,
                     cmap='gnuplot', 
                     marker='.', 
                     s=10,
                     p_top=99,
                     fontsize=10,
                     figsize=None, 
                     transparent=False, 
                     save=None):    
    """
    Visualize the distribution of specified features in a MuData object.

    This function plots the spatial or embedding-based distribution of specified features (e.g., genes or proteins) 
    across a sample dataset. Each feature is visualized individually, highlighting regions with high or low activity 
    levels. This allows users to identify spatial patterns or clusters of cells associated with specific features.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data, including spatial coordinates and expression matrices.
    :type mdata: muon.MuData
    :param x: 
        The key in `obsm` of `mdata` representing the spatial coordinates or embeddings to use for plotting.
    :type x: str
    :param features: 
        A list of features (e.g., genes or proteins) to visualize.
    :type features: list
    :param cmap: 
        The colormap to use for visualizing feature activity. Default is 'gnuplot'.
    :type cmap: str, optional
    :param marker: 
        Marker style for scatter plots. Default is '.'.
    :type marker: str, optional
    :param s: 
        Marker size in the scatter plots. Default is 10.
    :type s: int, optional
    :param p_top: 
        Percentile threshold to highlight top feature activity values. Points above this percentile are displayed 
        prominently. Default is 99.
    :type p_top: float, optional
    :param fontsize: 
        Font size for plot titles and colorbar ticks. Default is 10.
    :type fontsize: int, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. If None, size is automatically determined based 
        on the number of features. Default is None.
    :type figsize: tuple, optional
    :param transparent: 
        If True, saves the figure with a transparent background. Default is False.
    :type transparent: bool, optional
    :param save: 
        Path to save the figure. If None, the figure is displayed but not saved. Default is None.
    :type save: str, optional

    :returns: 
        None

    :example:

        .. code-block:: python

            import mtopic

            # Load MuData object
            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            # Specify features to visualize
            features = ['GeneA', 'GeneB', 'ProteinX']

            # Plot spatial distribution of selected features
            mtopic.pl.feature_activity(
                mdata, 
                x='coords', 
                features=features, 
                cmap='viridis', 
                save='feature_activity.png'
            )
    """
    
    n_features = len(features)
    nrow, ncol = optimal_grid(n_features)

    figsize = (ncol * 2, nrow * 2) if figsize == None else figsize
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(nrow, ncol, figure=fig)
    
    for i, f in enumerate(features):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])

        for m in mdata.mod:
            if mdata[:, f][m].shape[1] > 0:
                counts = mdata[:, f][m].X.toarray().flatten()

        vmax = np.percentile(counts, 99.9)
        thr = np.percentile(counts, p_top)
        mask = counts >= thr

        ax.scatter(x=mdata.obsm[x].values[~mask, 0],
                   y=mdata.obsm[x].values[~mask, 1],
                   edgecolor='none', 
                   c=counts[~mask],
                   cmap=cmap, 
                   s=s, 
                   vmin=0, 
                   vmax=vmax, 
                   marker=marker)
        
        p = ax.scatter(x=mdata.obsm[x].values[mask, 0],
                       y=mdata.obsm[x].values[mask, 1],
                       edgecolor='none', 
                       c=counts[mask],
                       cmap=cmap, 
                       s=s, 
                       vmin=0, 
                       vmax=vmax, 
                       marker=marker)
        
        ax.set(aspect='equal', title=f)
        ax.title.set_size(fontsize)
        ax.axis('off')
        cb = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    savefig(save, transparent)
