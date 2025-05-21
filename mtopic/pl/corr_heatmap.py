import numpy as np
import matplotlib.pyplot as plt
from ._utils import savefig


def corr_heatmap(arr1,
                 arr2,
                 label1=None,
                 label2=None,
                 cmap='bwr', 
                 fontsize=8,
                 figsize=(8, 6),
                 transparent=False, 
                 save=None):
    """
    Visualize the correlation matrix between two sets of features as a heatmap.

    This function computes and plots a correlation heatmap to visualize the relationships between two 
    sets of features. Each set is represented as a `pandas.DataFrame`, with columns as features. The color 
    intensity in the heatmap indicates the strength and direction of the correlation.

    :param arr1: 
        A pandas DataFrame representing the first set of features. Each column corresponds to a feature.
    :type arr1: pandas.DataFrame
    :param arr2: 
        A pandas DataFrame representing the second set of features. Each column corresponds to a feature.
    :type arr2: pandas.DataFrame
    :param label1: 
        Label for the y-axis, representing the features from `arr1`. If None, no label is set. Default is None.
    :type label1: str, optional
    :param label2: 
        Label for the x-axis, representing the features from `arr2`. If None, no label is set. Default is None.
    :type label2: str, optional
    :param cmap: 
        Colormap for the heatmap. Default is 'bwr' (blue-white-red colormap).
    :type cmap: str, optional
    :param fontsize: 
        Font size for axis labels and colorbar ticks. Default is 8.
    :type fontsize: int, optional
    :param figsize: 
        Tuple specifying the figure size (width, height) in inches. Default is (8, 6).
    :type figsize: tuple, optional
    :param transparent: 
        If True, saves the figure with a transparent background. Default is False.
    :type transparent: bool, optional
    :param save: 
        File path to save the figure. If None, the figure is displayed but not saved. Default is None.
    :type save: str, optional

    :returns: 
        None

    :example:

        .. code-block:: python

            import pandas as pd
            import mtopic

            # Create example datasets
            data1 = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            data2 = pd.DataFrame({'feature3': [7, 8, 9], 'feature4': [10, 11, 12]})

            # Plot correlation heatmap
            mtopic.pl.corr_heatmap(
                data1, 
                data2, 
                label1='Set 1', 
                label2='Set 2', 
                cmap='coolwarm', 
                fontsize=10, 
                save='correlation_heatmap.png'
            )
    """
    
    correlation_matrix = np.corrcoef(arr1.T, arr2.T)[:arr1.shape[1], arr1.shape[1]:]
    
    vmax = np.max(correlation_matrix[correlation_matrix > 0])

    plt.figure(figsize=figsize)
    plt.imshow(correlation_matrix, cmap=cmap, aspect='equal', vmin=-vmax, vmax=vmax)
    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize)

    plt.yticks(ticks=np.arange(arr1.shape[1]), labels=arr1.columns, fontsize=fontsize)
    plt.xticks(ticks=np.arange(arr2.shape[1]), labels=arr2.columns, rotation=90, fontsize=fontsize)

    if isinstance(label1, str):
        plt.ylabel(label1)
    if isinstance(label2, str):
        plt.xlabel(label2)
    
    savefig(save, transparent)
