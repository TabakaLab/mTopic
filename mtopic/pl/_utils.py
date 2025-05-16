import numpy as np
import matplotlib.pyplot as plt

def optimal_grid(n):
    """
    Determine the optimal grid size (rows and columns) for plotting a specified number of items.

    This function calculates the optimal number of rows and columns to display a given number of plots 
    (e.g., subplots in a grid). It aims to make the grid as square-like as possible, minimizing the difference 
    between the number of rows and columns.

    :param n: The total number of items to display in the grid.
    :type n: int

    :returns: A tuple containing the optimal number of rows and columns for the grid.
    :rtype: tuple (int, int)

    :example:

        .. code-block:: python

            from mtopic.utils import optimal_grid

            # Determine optimal grid size for 7 plots
            rows, cols = optimal_grid(7)
            print(f"Optimal grid size: {rows} rows x {cols} columns")

    :notes:
        - This function is useful for arranging subplots in a grid that is as square as possible.
        - It minimizes the difference between the number of rows and columns to achieve a visually balanced layout.
    """
    
    possible_rows = np.arange(1, int(np.sqrt(n)) + 2)
    possible_columns = np.ceil(n / possible_rows).astype(int)
    differences = np.abs(possible_columns - possible_rows)
    
    min_index = np.argmin(differences)
    best_rows = possible_rows[min_index]
    best_columns = possible_columns[min_index]
    
    return best_rows, best_columns


def savefig(save, 
            transparent, 
            dpi=300):
    """
    Save a figure to a file with optional transparency and resolution.

    This function saves the current matplotlib figure to a file with specified resolution (DPI) 
    and an option to make the background transparent. If a valid path is provided, the figure is saved 
    to that location; otherwise, the function does nothing.

    :param save: The file path where the figure should be saved. If None, the figure is not saved.
    :type save: str or None
    :param transparent: Whether to save the figure with a transparent background. Default is False.
    :type transparent: bool, optional
    :param dpi: The resolution of the saved figure in dots per inch (DPI). Default is 300.
    :type dpi: int, optional

    :returns: None

    :example:

        .. code-block:: python

            import matplotlib.pyplot as plt
            from mtopic.utils import savefig

            # Create a plot
            plt.plot([1, 2, 3], [4, 5, 6])

            # Save the plot to a file
            savefig('plot.png', transparent=True, dpi=300)

    :notes:
        - This function is designed to be used after creating a plot with matplotlib.
        - The `transparent` parameter is useful for saving plots with a transparent background, 
          which is helpful when overlaying plots or using them in presentations.
    """

    if isinstance(save, str):
        plt.savefig(save, dpi=dpi, transparent=transparent)
        plt.close()