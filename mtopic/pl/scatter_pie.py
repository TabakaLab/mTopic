import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from matplotlib.lines import Line2D
from ._utils import savefig


def scatter_pie(mdata, 
                topics='topics',
                x='coords', 
                radius=0.005, 
                xrange=[0, 1], 
                yrange=[0, 1], 
                figsize=(10, 10), 
                palette=None, 
                legend_markersize=10,
                legend_ncol=1,
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
        The key in `obsm` of `mdata` representing the topic distributions. Default is 'topics'.
    :type topics: str, optional
    :param x: 
        The key in `obsm` of `mdata` representing the spatial coordinates or embeddings for plotting that is a pandas.DataFrame. Default is 'coords'.
    :type x: str, optional
    :param radius: 
        The radius of the pie charts. Default is 0.005.
    :type radius: float, optional
    :param xrange: 
        The range of x-coordinates to display in the plot. Default is [0, 1].
    :type xrange: list, optional
    :param yrange: 
        The range of y-coordinates to display in the plot. Default is [0, 1].
    :type yrange: list, optional
    :param figsize: 
        The size of the figure (width, height) in inches. Default is (10, 10).
    :type figsize: tuple, optional
    :param palette: 
        A dictionary mapping topics to colors. If None, a default palette is generated. Default is None.
    :type palette: dict, optional
    :param legend_markersize: 
        The size of the markers in the legend. Default is 10.
    :type legend_markersize: int, optional
    :param legend_ncol: 
        The number of columns in the legend. Default is 1.
    :type legend_ncol: int, optional
    :param transparent: 
        Whether to save the figure with a transparent background. Default is False.
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

            # Plot scatter pie charts of topic distributions
            mtopic.pl.scatter_pie(
                mdata, 
                topics='topics', 
                x='coords', 
                radius=0.01, 
                palette=None, 
                save='scatter_pie.png'
            )

    :notes:
        - **Range Filters**:
          The `xrange` and `yrange` parameters allow zooming into specific regions of the plot.
    """
    
    topics = mdata.obsm[topics]
    coords = mdata.obsm[x]

    coords -= coords.min()
    coords /= coords.max().max()

    mask = (coords.iloc[:, 0] >= xrange[0]) & (coords.iloc[:, 0] <= xrange[1]) & (coords.iloc[:, 1] >= yrange[0]) & (coords.iloc[:, 1] <= yrange[1])

    topics = topics[mask]
    coords = coords[mask]

    x = coords.iloc[:, 0]
    y = coords.iloc[:, 1]

    if palette is None:
        palette = dict()
        for i, t in enumerate(topics.columns):
            if i < 20:
                palette[t] = plt.get_cmap('tab20')(i/20)
            elif i < 40:
                palette[t] = plt.get_cmap('tab20b')((i-20)/20)
            else:
                palette[t] = plt.get_cmap('tab20c')((i-40)/20)

    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
    
    ax = fig.add_subplot(gs[0, 0])

    colors_ordered = [palette[t] for t in topics.columns]

    for i in tqdm(range(topics.shape[0])):
        proportions = topics.iloc[i]
        center_x, center_y = x.iloc[i], y.iloc[i]
        pie = ax.pie(proportions, radius=radius, center=(center_x, center_y), labels=None, colors=colors_ordered)
    
    ax.set(aspect='equal', xlim=[xrange[0]-0.01, xrange[1]+0.01], ylim=[yrange[0]-0.01, yrange[1]+0.01])


    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')

    
    legend_elements = [Line2D([0], [0], marker='o', color='none', markeredgewidth=0,
                              markerfacecolor=palette[ctype], markersize=legend_markersize, label=ctype) for ctype in palette]

    ax.legend(handles=legend_elements, loc='center', frameon=False, ncol=legend_ncol)

    savefig(save, transparent)
