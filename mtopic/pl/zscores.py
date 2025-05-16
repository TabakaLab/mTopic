import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from ._utils import optimal_grid, savefig

def zscores(mdata,
            mod, 
            x, 
            zscores='zscores',
            cmap=None, 
            marker='.',
            s=10, 
            fontsize=10,
            figsize=None, 
            transparent=False,
            save=None):
    """
    Visualize the spatial or embedding-based distribution of z-scores for topics in a specified modality.

    This function generates scatter plots showing the distribution of z-scores for each topic in a specified modality 
    of a `MuData` object. Z-scores are used to highlight spatial or embedding-based patterns of feature expression 
    across samples, revealing significant variations relative to the global mean.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data with z-scores stored in the `obsm` attribute.
    :type mdata: muon.MuData
    :param mod: 
        The modality to visualize z-scores for (e.g., 'rna', 'protein').
    :type mod: str
    :param x: 
        The key in `obsm` of `mdata` representing the spatial coordinates or embeddings to use for plotting (e.g., 'coords', 'umap').
    :type x: str
    :param zscores: 
        The key in `obsm` of the specified modality representing the z-scores to plot. Default is 'zscores'.
    :type zscores: str, optional
    :param cmap: 
        The colormap to use for visualizing z-scores. If None, a custom colormap is applied. Default is None.
    :type cmap: matplotlib.colors.Colormap or None, optional
    :param marker: 
        Marker style for scatter plots. Default is '.'.
    :type marker: str, optional
    :param s: 
        Marker size for scatter plots. Default is 10.
    :type s: int, optional
    :param fontsize: 
        Font size for plot titles and colorbar ticks. Default is 10.
    :type fontsize: int, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. If None, the size is automatically determined 
        based on the number of topics. Default is None.
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

            # Visualize z-scores for topics in the 'rna' modality using spatial coordinates
            mtopic.pl.zscores(
                mdata, 
                mod='rna', 
                x='coords', 
                zscores='zscores', 
                cmap='coolwarm', 
                marker='o', 
                s=15
            )

            # Save the figure to a file
            mtopic.pl.zscores(
                mdata, 
                mod='rna', 
                x='umap', 
                save='zscores_distribution.pdf'
            )
    """

    cmap = cmap if cmap != None else LinearSegmentedColormap.from_list('custom', list(zip([0, 0.1, 0.5, 0.9, 1],
                                                                ['#000052', '#0000ff', '#e3e3e3', '#ff0000', '#520000'])), N=256)
    
    n_topics = mdata[mod].obsm[zscores].shape[1]
    nrow, ncol = optimal_grid(n_topics)

    figsize = (ncol * 2, nrow * 2) if figsize == None else figsize
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(nrow, ncol, figure=fig)
    
    for t in range(n_topics):
        ax = fig.add_subplot(gs[t // ncol, t % ncol])
        
        vmax = np.percentile(mdata[mod].obsm[zscores].iloc[:, t], 99.9)
        thr = np.percentile(mdata[mod].obsm[zscores].iloc[:, t], 99)
        mask = mdata[mod].obsm[zscores].iloc[:, t] > thr

        ax.scatter(x=mdata.obsm[x].values[~mask, 0],
                   y=mdata.obsm[x].values[~mask, 1],
                   edgecolor='none', 
                   c=mdata[mod].obsm[zscores].values[~mask, t],
                   cmap=cmap, 
                   s=s, 
                   vmin=-vmax, 
                   vmax=vmax, 
                   marker=marker)
        
        p = ax.scatter(x=mdata.obsm[x].values[mask, 0],
                       y=mdata.obsm[x].values[mask, 1],
                       edgecolor='none', 
                       c=mdata[mod].obsm[zscores].values[mask, t],
                       cmap=cmap, 
                       s=s, 
                       vmin=-vmax, 
                       vmax=vmax, 
                       marker=marker)
        
        ax.set(aspect='equal', title=mdata[mod].obsm[zscores].columns[t])
        ax.title.set_size(fontsize)
        ax.axis('off')
        cb = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    savefig(save, transparent)
