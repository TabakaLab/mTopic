import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ._utils import optimal_grid, savefig


def topics(mdata, 
           x, 
           topics='topics',
           cmap='gnuplot', 
           marker='.', 
           s=10, 
           vmax=None, 
           fontsize=10, 
           figsize=None, 
           transparent=False,
           save=None):
    """
    Visualize topic distributions on spatial coordinates or embedding.

    This function generates scatter plots to visualize the topic distributions across cells/spots. 
    Each topic is displayed in a separate subplot, showing how the topic is spatially distributed or distributed 
    within a specified embedding. The plots are arranged in a grid for easy comparison.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data, with topic distributions stored in `obsm`.
    :type mdata: muon.MuData
    :param x: 
        The key in `obsm` of `mdata` representing the spatial coordinates or embeddings to use for plotting (e.g., UMAP, PCA, or spatial coordinates).
    :type x: str
    :param topics: 
        The key in `obsm` of `mdata` representing the topic distributions to plot. Default is 'topics'.
    :type topics: str, optional
    :param cmap: 
        Colormap to use for visualizing topic distributions. Default is 'gnuplot'.
    :type cmap: str, optional
    :param marker: 
        Marker style for scatter plots. Default is '.'.
    :type marker: str, optional
    :param s: 
        Marker size in scatter plots. Default is 10.
    :type s: int, optional
    :param vmax: 
        Maximum value for the color scale. If None, it is set to the 99.9th percentile of the data for each topic. Default is None.
    :type vmax: float, optional
    :param fontsize: 
        Font size for plot titles and colorbar ticks. Default is 10.
    :type fontsize: int, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. If None, the size is automatically determined based on the number of topics. Default is None.
    :type figsize: tuple, optional
    :param transparent: 
        Whether to make the figure background transparent. Useful for overlays in presentations. Default is False.
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

            # Plot spatial distribution of topics using UMAP coordinates
            mtopic.pl.topics(
                mdata, 
                x='umap', 
                topics='topics', 
                cmap='viridis', 
                marker='o', 
                s=20, 
                fontsize=12
            )

            # Save the figure to a file
            mtopic.pl.topics(
                mdata, 
                x='umap', 
                topics='topics', 
                save='topics_distribution.pdf'
            )
    """
    
    n_topics = mdata.obsm[topics].shape[1]
    nrow, ncol = optimal_grid(n_topics)

    figsize = (ncol * 2, nrow * 2) if figsize == None else figsize
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(nrow, ncol, figure=fig)
    
    for t in range(n_topics):
        ax = fig.add_subplot(gs[t // ncol, t % ncol])
        
        vmax_t = np.percentile(mdata.obsm[topics].iloc[:, t], 99.9) if vmax == None else vmax
        thr = np.percentile(mdata.obsm[topics].iloc[:, t], 99)
        mask = mdata.obsm[topics].iloc[:, t] >= thr

        ax.scatter(x=mdata.obsm[x].values[~mask, 0],
                   y=mdata.obsm[x].values[~mask, 1],
                   edgecolor='none', 
                   c=mdata.obsm[topics].values[~mask, t],
                   cmap=cmap, 
                   s=s, 
                   vmin=0, 
                   vmax=vmax_t, 
                   marker=marker)
        
        p = ax.scatter(x=mdata.obsm[x].values[mask, 0],
                       y=mdata.obsm[x].values[mask, 1],
                       edgecolor='none', 
                       c=mdata.obsm[topics].values[mask, t],
                       cmap=cmap, 
                       s=s, 
                       vmin=0, 
                       vmax=vmax_t, 
                       marker=marker)
        
        ax.set(aspect='equal', title=mdata.obsm[topics].columns[t])
        ax.title.set_size(fontsize)
        ax.axis('off')
        cb = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    savefig(save, transparent)
