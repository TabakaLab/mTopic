import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from ._utils import savefig
    

def dominant_topics(mdata,
                    x, 
                    topics='topics',
                    palette=None,
                    annotation=None,
                    title=None,
                    marker='.',
                    s=20, 
                    fontsize=10,
                    markerscale=1,
                    legend=True,
                    legend_ncol=1,
                    figsize=(7, 5),
                    random_state=2291,
                    transparent=False, 
                    save=None):
    """
    Visualize the dominant topic for each cell/spot in a MuData object.

    This function creates a scatter plot where each point represents a cell/spot, colored according to the 
    dominant topic (i.e., the topic with the highest probability) for that sample. The plot provides an 
    intuitive overview of how topics are distributed spatially or in a given embedding. A legend maps 
    colors to topics for easy interpretation.

    :param mdata: 
        A `MuData` object containing multimodal single-cell data with topic distributions stored in `obsm`.
    :type mdata: muon.MuData
    :param x: 
        The key in `obsm` of `mdata` representing the spatial coordinates or embeddings to use for plotting 
        (e.g., 'coords', 'umap').
    :type x: str
    :param topics: 
        The key in `obsm` of `mdata` representing the topic distribution. Default is 'topics'.
    :type topics: str, optional
    :param palette: 
        A dictionary mapping topics to specific colors. If None, a default palette of unique hex colors is generated. 
        Default is None.
    :type palette: dict, optional
    :param annotation:
        Dictionary mapping topic names to display labels shown in the legend
        (e.g. ``{'topic_1': 'Inhibitory neurons-3', ...}``). If ``None`` or a topic is not found,
        the raw topic name is used. Default is ``None``.
    :type annotation: dict, optional
    :param title:
        Title of the plot. If None, no title is shown. Default is None.
    :type title: str, optional
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
        Scale of markers in the legend relative to their size in the scatter plot. Default is 1.
    :type markerscale: float, optional
    :param legend:
        Whether to display the legend. Default is ``True``.
    :type legend: bool, optional
    :param legend_ncol:
        Number of columns in the legend. Default is ``1``.
    :type legend_ncol: int, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. Default is (7, 5).
    :type figsize: tuple, optional
    :param random_state:
        Random seed for shuffling point plotting order, so no topic systematically
        occludes another. If None, a different random order is used each call.
        Default is 2291.
    :type random_state: int, optional
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

    if palette is None:
        topics_list = mdata.obsm[topics].columns
        n = len(topics_list)

        if n <= 20:
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i / 20) for i in range(n)]
        elif n <= 40:
            cmaps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b')]
            colors = [cmaps[i // 20](i % 20 / 20) for i in range(n)]
        elif n <= 60:
            cmaps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b'), plt.get_cmap('tab20c')]
            colors = [cmaps[i // 20](i % 20 / 20) for i in range(n)]
        else:
            cmap = plt.get_cmap('hsv')
            colors = [cmap(i / n) for i in range(n)]

        palette = {t: to_hex(c) for t, c in zip(topics_list, colors)}

    fig = plt.figure(constrained_layout=True, figsize=figsize)

    if legend:
        gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
    else:
        gs = GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])

    topics_list = mdata.obsm[topics].columns
    dominant = np.argmax(mdata.obsm[topics].values, axis=1)
    point_colors = np.array([palette[topics_list[i]] for i in dominant])
    all_x = mdata.obsm[x].values[:, 0]
    all_y = mdata.obsm[x].values[:, 1]

    if random_state is not None:
        np.random.seed(random_state)
    shuffled_idx = np.random.permutation(len(all_x))

    ax.scatter(x=all_x[shuffled_idx],
               y=all_y[shuffled_idx],
               c=point_colors[shuffled_idx],
               edgecolor='none', marker=marker, s=s)

    ax.set_aspect('equal')
    ax.axis('off')

    if title is not None:
        ax.set_title(title)

    if legend:
        if annotation:
            sorted_topics = sorted(
                palette.keys(),
                key=lambda t: annotation[t] if t in annotation else t
            )
        else:
            sorted_topics = list(mdata.obsm[topics].columns)

        legend_elements = [
            Line2D([0], [0], marker=marker, color='none', markeredgewidth=0,
                   markerfacecolor=palette[t],
                   markersize=np.sqrt(s),
                   label=annotation[t] if annotation and t in annotation else t)
            for t in sorted_topics
        ]

        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.legend(handles=legend_elements, frameon=False, loc='right',
                      fontsize=fontsize, markerscale=markerscale,
                      ncol=legend_ncol)
        ax_leg.axis('off')

    savefig(save, transparent)