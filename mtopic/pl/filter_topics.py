import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
from ._utils import savefig


def filter_topics(model,
                  s=50,
                  figsize=(8, 6), 
                  fontsize=10, 
                  transparent=False, 
                  save=None):
    """
    Visualize the significance of topics based on their maximum proportion across cells.

    This function generates a scatter plot to show the maximum normalized proportion of each topic across all cells. 
    Topics with higher maximum proportions represent significant patterns in the dataset, while topics with low 
    maximum proportions might be less informative or represent noise. The plot includes a suggested threshold line 
    (default at y=0.01) to help identify insignificant topics for filtering in downstream analysis.

    :param model: 
        An instance of the topic model (e.g., `mtopic.tl.MTM` or `mtopic.tl.sMTM`) containing the topic proportions (`gamma`) to analyze.
    :type model: mtopic.tl.MTM or mtopic.tl.sMTM
    :param s: 
        Marker size for the scatter plot. Default is 50.
    :type s: int, optional
    :param figsize: 
        Tuple specifying the size of the figure (width, height) in inches. Default is (8, 6).
    :type figsize: tuple, optional
    :param fontsize: 
        Font size for plot labels and annotations. Default is 10.
    :type fontsize: int, optional
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

            # Assuming `model` is an instance of mtopic.tl.MTM or mtopic.tl.sMTM
            mtopic.pl.filter_topics(model, s=50, figsize=(8, 6), fontsize=10, save='filter_topics.png')
    """
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
    ax = fig.add_subplot(gs[0, 0])

    gamma_normalized = model.gamma / model.gamma.sum(axis=1)[:, np.newaxis]
    gamma_normalized = pd.DataFrame(gamma_normalized,
                                    index=model.barcodes, 
                                    columns=['topic_{}'.format(i+1) for i in range(model.n_topics)])
    max_proportions = gamma_normalized.max(axis=0).sort_values(ascending=False)

    x = [i for i in range(model.n_topics)]
    y = max_proportions.values
    ax.scatter(x=x, y=y, edgecolor='none', s=s, c='#990000')
    ax.set(ylim=[0, 1.02], xticks=[], ylabel='Maximum proportion of a topic across cells', xlabel='Topics', xlim=[-0.02*(model.n_topics-1), 1.02*(model.n_topics-1)])
    ax.grid(axis='y', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    for i, txt in enumerate(max_proportions.index):
        ax.annotate('{} : {}'.format(txt, np.round(y[i], 2)), (x[i], y[i] + 0.03 if y[i] < 0.5 else y[i] - 0.02), 
                    rotation=90, 
                    va='bottom' if y[i] < 0.5 else 'top', 
                    ha='center', 
                    fontsize=fontsize)

    ax.axhline(y=0.01, c='black')
    ax.annotate('Default topic filtering threshold: 0.01', (model.n_topics/2, 0.02), 
                va='bottom', 
                ha='center', 
                fontsize=fontsize)

    savefig(save, transparent)
