import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import kneed
from ._utils import savefig


def filter_var_knee(model,
                    mod,
                    knee_sensitivities=[1, 2, 5, 10],
                    s=20,
                    figsize=(8, 6), 
                    fontsize=10,
                    annotate_features=False, 
                    show_frac=1, 
                    log_scale=True, 
                    transparent=False,
                    save=None):
    """
    Detect and visualize overrepresented features using a knee detection algorithm.

    This function identifies and visualizes features (e.g., genes) that are overrepresented across all topics 
    in a specified modality of a topic model. By plotting the cumulative feature scores across topics, 
    it applies a knee detection algorithm to find points of significant activity drop, which can be used 
    to filter out less informative features in downstream analysis.

    Multiple knee points are calculated using varying sensitivities, providing insight into how the detection 
    threshold affects the selection of overrepresented features.

    :param model: 
        An instance of a topic model (e.g., `mtopic.tl.MTM` or `mtopic.tl.sMTM`) containing the topic-feature distributions.
    :type model: mtopic.tl.MTM or mtopic.tl.sMTM
    :param mod: 
        The modality to analyze (e.g., 'rna', 'protein').
    :type mod: str
    :param knee_sensitivities: 
        List of sensitivities for the knee detection algorithm. Higher sensitivity values detect more subtle changes. 
        Default is [1, 2, 5, 10].
    :type knee_sensitivities: list of int, optional
    :param s: 
        Marker size for the scatter plot. Default is 20.
    :type s: int, optional
    :param figsize: 
        Tuple specifying the size of the figure (width, height) in inches. Default is (8, 6).
    :type figsize: tuple, optional
    :param fontsize: 
        Font size for plot labels, annotations, and ticks. Default is 10.
    :type fontsize: int, optional
    :param annotate_features: 
        Whether to annotate feature names on the plot. If True, feature names will be displayed near their points. 
        Default is False.
    :type annotate_features: bool, optional
    :param show_frac: 
        Fraction of the top features to display based on their cumulative activity. Default is 1 (all features).
    :type show_frac: float, optional
    :param log_scale: 
        Whether to apply a log scale to the y-axis for feature activity. Default is True.
    :type log_scale: bool, optional
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

            # Assume `model` is an instance of mtopic.tl.MTM or mtopic.tl.sMTM
            mtopic.pl.filter_var_knee(
                model, 
                mod='rna', 
                knee_sensitivities=[1, 2, 5], 
                annotate_features=True, 
                save='knee_plot.pdf'
            )
    """

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 0.1])
    ax = fig.add_subplot(gs[0, 0])

    sums = pd.DataFrame(model.lambda_[mod], columns=model.features[mod])
    sums = sums.sum(axis=0).sort_values(ascending=False)
    
    n_show = int(len(sums) * show_frac)
    
    x = [i for i in range(n_show)]
    
    y = sums.values[:n_show]
    if log_scale:
        y = np.log1p(y)
    y = y / np.max(y)

    names = sums.index[:n_show]

    ylabel = 'Normalized cumulative feature activity across topics'
    if log_scale:
        ylabel = 'Normalized cumulative feature log-activity across topics'
    
    ax.scatter(x=x, y=y, edgecolor='none', s=s, c='#990000', zorder=1)
    ax.set(ylim=[0, 1.02], xticks=[], ylabel=ylabel, xlabel='{} features'.format(mod), 
           xlim=[-0.02*n_show, 1.02*(n_show-1)])
    ax.grid(axis='y', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    for i, S in enumerate(knee_sensitivities):
        knee = kneed.KneeLocator([i for i in range(len(sums))], sums.values, curve='convex', direction='decreasing', S=S).knee

        ax.axvline(x=knee, c='black', zorder=0)

        ax.annotate('S={}'.format(S), (knee + n_show * 0.005, 0.5+0.05*i), 
                    va='center', 
                    ha='left', 
                    fontsize=fontsize, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    if annotate_features:
        for i, txt in enumerate(names):
            ax.annotate(txt, (i, y[i] + 0.03 if y[i] < 0.5 else y[i] - 0.02), 
                        rotation=90, 
                        va='bottom' if y[i] < 0.5 else 'top',
                        ha='center', 
                        fontsize=fontsize)

    savefig(save, transparent)
