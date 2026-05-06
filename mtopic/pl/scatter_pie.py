import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from ._utils import savefig


def scatter_pie(mdata, 
                topics='topics',
                x='coords', 
                radius=0.005, 
                xrange=[0, 1], 
                yrange=[0, 1], 
                figsize=(10, 10), 
                palette=None,
                annotation=None,
                title=None,
                legend=True,
                legend_ncol=1,
                legend_markersize=10,
                fontsize=10,
                transparent=False, 
                save=None):
    """
    Create a scatter plot with pie charts representing topic distributions at each cell/spot coordinate.

    This function visualizes topic distributions for each sample in a dataset using pie charts positioned
    at their corresponding spatial or embedding coordinates. Each pie chart represents the distribution
    of topics for a single cell/spot, and a legend provides the color mapping for each topic.

    :param mdata:
        A `MuData` object containing multimodal single-cell data, including topic distributions and coordinates.
    :type mdata: muon.MuData
    :param topics:
        Key in `mdata.obsm` for the topic distribution matrix. Default is ``'topics'``.
    :type topics: str, optional
    :param x:
        Key in `mdata.obsm` for a pandas DataFrame of spatial or embedding coordinates. Default is ``'coords'``.
    :type x: str, optional
    :param radius:
        Radius of each pie chart in data coordinates. Default is ``0.005``.
    :type radius: float, optional
    :param xrange:
        Range ``[min, max]`` of x-coordinates to display. Default is ``[0, 1]``.
    :type xrange: list, optional
    :param yrange:
        Range ``[min, max]`` of y-coordinates to display. Default is ``[0, 1]``.
    :type yrange: list, optional
    :param figsize:
        Figure size ``(width, height)`` in inches. Default is ``(10, 10)``.
    :type figsize: tuple, optional
    :param palette:
        Dictionary mapping topic names to hex color strings (e.g. ``{'topic_1': '#ffbcdd', ...}``).
        If ``None``, colors are generated automatically from matplotlib colormaps. Default is ``None``.
    :type palette: dict, optional
    :param annotation:
        Dictionary mapping topic names to display labels shown in the legend
        (e.g. ``{'topic_1': 'Inhibitory neurons-3', ...}``). If ``None`` or a topic is not found,
        the raw topic name is used. Default is ``None``.
    :type annotation: dict, optional
    :param title:
        Title of the plot. If None, no title is shown. Default is None.
    :type title: str, optional
    :param legend:
        Whether to display the legend. Default is ``True``.
    :type legend: bool, optional
    :param legend_ncol:
        Number of columns in the legend. Default is ``1``.
    :type legend_ncol: int, optional
    :param legend_markersize:
        Size of the circle markers in the legend. Default is ``10``.
    :type legend_markersize: int, optional
    :param fontsize:
        Font size for legend labels. Default is ``10``.
    :type fontsize: int, optional
    :param transparent:
        Whether to save the figure with a transparent background. Default is ``False``.
    :type transparent: bool, optional
    :param save:
        File path to save the figure. If ``None``, the figure is displayed but not saved. Default is ``None``.
    :type save: str, optional

    :returns:
        None

    :example:

        .. code-block:: python

            import mtopic

            mdata = mtopic.read.h5mu("path/to/file.h5mu")

            mtopic.pl.scatter_pie(
                mdata,
                topics='topics',
                x='coords',
                radius=0.01,
                palette=P22ATAC_TOPIC_COLOR,
                annotation=P22ATAC_TOPIC_CELLTYPE,
                save='scatter_pie.png'
            )

    :notes:
        - **Coordinates** are normalised to ``[0, 1]`` before plotting.
        - **Range filters**: use ``xrange`` and ``yrange`` to zoom into a specific region.
        - **Performance**: pie charts are rendered as ``PatchCollection`` objects (one per topic)
          rather than individual ``ax.pie()`` calls, making the function efficient for large datasets.
    """

    topics_df = mdata.obsm[topics].copy()
    coords = mdata.obsm[x].copy()

    coords -= coords.min()
    coords /= coords.max().max()

    mask = (
        (coords.iloc[:, 0] >= xrange[0]) & (coords.iloc[:, 0] <= xrange[1]) &
        (coords.iloc[:, 1] >= yrange[0]) & (coords.iloc[:, 1] <= yrange[1])
    )

    topics_df = topics_df[mask]
    coords = coords[mask]

    x_vals = coords.iloc[:, 0].values
    y_vals = coords.iloc[:, 1].values

    if palette is None:
        topics_list = topics_df.columns
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

    proportions = topics_df.values
    cumsum      = np.cumsum(proportions, axis=1) * 360
    starts      = np.hstack([np.zeros((len(topics_df), 1)), cumsum[:, :-1]])

    fig = plt.figure(constrained_layout=True, figsize=figsize)

    if legend:
        gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
    else:
        gs = GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])

    for k, topic in enumerate(topics_df.columns):
        patches = [
            Wedge((x_vals[i], y_vals[i]), radius,
                  90 - cumsum[i, k],
                  90 - starts[i, k])
            for i in range(len(topics_df))
        ]
        col = PatchCollection(patches, facecolor=palette[topic], edgecolor='none')
        ax.add_collection(col)

    ax.set(aspect='equal',
           xlim=[xrange[0] - 0.01, xrange[1] + 0.01],
           ylim=[yrange[0] - 0.01, yrange[1] + 0.01])
    ax.axis('off')

    if title is not None:
        ax.set_title(title)

    if legend:
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis('off')

        if annotation:
            sorted_topics = sorted(
                palette.keys(),
                key=lambda t: annotation[t] if t in annotation else t
            )
        else:
            sorted_topics = list(mdata.obsm[topics].columns)

        legend_elements = [
            Line2D([0], [0], marker='o', color='none', markeredgewidth=0,
                   markerfacecolor=palette[t],
                   markersize=legend_markersize,
                   label=annotation[t] if annotation and t in annotation else t)
            for t in sorted_topics
        ]
        ax_leg.legend(handles=legend_elements, loc='center', frameon=False,
                      ncol=legend_ncol, fontsize=fontsize)

    savefig(save, transparent)