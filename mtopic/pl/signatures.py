import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ._utils import optimal_grid, savefig

def signatures(mdata,
               mod, 
               signatures='signatures', 
               n_top=30, 
               cmap='viridis', 
               figsize=None,
               fontsize=8,
               transparent=False,
               save=None):
    """
    Visualize the top features (signatures) for each topic in a specified modality of a MuData object.

    This function generates bar plots displaying the top `n_top` features associated with each topic 
    in a specified modality of a `MuData` object. Features are ranked by their importance in the topic, 
    providing insight into the key contributors for each topic. The plots are arranged in a grid layout for easy comparison.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data with topic-feature distributions stored in `varm`.
    :type mdata: muon.MuData
    :param mod: 
        The modality to visualize topic signatures for (e.g., 'rna', 'protein').
    :type mod: str
    :param signatures: 
        Key in the `varm` attribute of the specified modality representing the topic-feature distributions to plot. 
        Default is 'signatures'.
    :type signatures: str, optional
    :param n_top: 
        Number of top features to display for each topic. Default is 30.
    :type n_top: int, optional
    :param cmap: 
        Colormap to use for visualizing feature importance. Default is 'viridis'.
    :type cmap: str, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. If None, the size is automatically determined 
        based on the number of topics and the number of top features. Default is None.
    :type figsize: tuple, optional
    :param fontsize: 
        Font size for plot titles and labels. Default is 8.
    :type fontsize: int, optional
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

            # Visualize the top 30 features for each topic in the 'rna' modality
            mtopic.pl.signatures(
                mdata, 
                mod='rna', 
                signatures='signatures', 
                n_top=30, 
                cmap='plasma', 
                fontsize=10
            )

            # Save the figure to a file
            mtopic.pl.signatures(
                mdata, 
                mod='rna', 
                save='topic_signatures.pdf'
            )
    """
    
    n_topics = mdata[mod].varm[signatures].shape[1]
    nrow, ncol = optimal_grid(n_topics)
    
    figsize = (ncol*3, nrow*2*(n_top/15)) if figsize == None else figsize
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(nrow, ncol, figure=fig)
    
    for i, t in enumerate(mdata[mod].varm[signatures].columns):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        
        top = mdata[mod].varm[signatures].iloc[:, i].sort_values(ascending=True).iloc[-n_top:]
        cmin, cmax = np.min(top.values), np.max(top.values)
        colors = [plt.get_cmap(cmap)((v - cmin) / (cmax - cmin)) for v in top.values]
        barh = ax.barh(y=range(n_top), width=top.values, height=0.7, color=colors)
        ax.set(yticks=[], ylim=[-1, n_top], xlim=[0, np.max(top.values)*1.5], title=t)
        ax.bar_label(barh, top.index, padding=5, color='black', fontsize=fontsize)
        ax.spines[['right']].set_visible(False)
        ax.title.set_size(fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    savefig(save, transparent)
