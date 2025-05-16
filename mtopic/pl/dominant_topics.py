import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import rgb2hex
from ._utils import savefig


def _generate_hex_colors(n, 
                         seed=2291):
    """
    Generate a list of unique hex color codes.

    This function generates a list of `n` unique hex color codes using a random number generator with a specified seed. 

    :param n: The number of hex colors to generate.
    :type n: int
    :param seed: The seed for the random number generator. Default is 2291.
    :type seed: int, optional

    :returns: A list of unique hex color strings.
    :rtype: list
    """
    
    rng = np.random.default_rng(seed=seed)
    colors = set()
    while len(colors) < n:
        color = "#{:06x}".format(rng.integers(0, 0xFFFFFF))
        colors.add(color)
    
    return list(colors)
    

def dominant_topics(mdata,
                    x, 
                    topics='topics',
                    palette=None, 
                    marker='.',
                    s=20, 
                    fontsize=10,
                    markerscale=2, 
                    figsize=(7, 5), 
                    transparent=False, 
                    save=None):
    """
    Visualize the dominant topic for each sample in a MuData object.

    This function creates a scatter plot where each point represents a sample, colored according to the 
    dominant topic (i.e., the topic with the highest proportion) for that sample. The plot provides an 
    intuitive overview of how topics are distributed spatially or in a given embedding. A legend maps 
    colors to topics for easy interpretation.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data with topic proportions stored in `obsm`.
    :type mdata: muon.MuData
    :param x: 
        The key in `obsm` of `mdata` representing the spatial coordinates or embeddings to use for plotting 
        (e.g., 'coords', 'umap').
    :type x: str
    :param topics: 
        The key in `obsm` of `mdata` representing the topic proportions. Default is 'topics'.
    :type topics: str, optional
    :param palette: 
        A dictionary mapping topics to specific colors. If None, a default palette of unique hex colors is generated. 
        Default is None.
    :type palette: dict, optional
    :param marker: 
        Marker style for the scatter plot. Default is '.'.
    :type marker: str, optional
    :param s: 
        Marker size in the scatter plot. Default is 20.
    :type s: int, optional
    :param fontsize: 
        Font size for legend labels. Default is 10.
    :type fontsize: int, optional
    :param markerscale: 
        Scale of markers in the legend relative to their size in the scatter plot. Default is 2.
    :type markerscale: float, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. Default is (7, 5).
    :type figsize: tuple, optional
    :param transparent: 
        Whether to save the figure with a transparent background. Useful for overlays or presentations. Default is False.
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

            # Plot dominant topics for all samples
            mtopic.pl.dominant_topics(
                mdata, 
                x='umap', 
                topics='topics', 
                marker='o', 
                s=30, 
                fontsize=12, 
                markerscale=3
            )

            # Save the figure to a file
            mtopic.pl.dominant_topics(
                mdata, 
                x='coords', 
                save='dominant_topics.pdf'
            )
    """
    
    n_topics = mdata.obsm[topics].shape[1]

    if palette is None:
        generated_numbers = _generate_hex_colors(n_topics)
        palette = {t: generated_numbers[i] for i, t in enumerate(mdata.obsm[topics].columns)}

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
    ax = fig.add_subplot(gs[0, 0])
    
    for i, t in enumerate(mdata.obsm[topics].columns):
        mask = np.argmax(mdata.obsm[topics], axis=1) == i
        ax.scatter(x=mdata.obsm[x].values[mask, 0], 
                   y=mdata.obsm[x].values[mask, 1], 
                   edgecolor='none', marker=marker,
                   c=palette[t], label=t, s=s)
        
    ax.set_aspect('equal')
    ax.axis('off')
    h, l = ax.get_legend_handles_labels()
    
    ax = fig.add_subplot(gs[0, 1])
    ax.legend(h, l, frameon=False, loc='right', fontsize=fontsize, markerscale=markerscale, ncol=np.ceil(n_topics/20))
    ax.axis('off')

    savefig(save, transparent)
