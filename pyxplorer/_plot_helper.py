"""
helper functions for plot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



def get_lc_traj_singleColor(xs, ys, c, vmin, vmax, cmap, lw=0.8):
    """
    Get line collection object a trajectory with a single color based on a colormap defined by vmin ~ vmax
    For plotting, run:
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label="Colorbar label")

    Args:
        xs (np.array): array-like object of x-coordinates of the trajectory

    Returns:
        (obj): line collection object
    """
    # generate segments
    points = np.array([ xs , ys ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    plt_color = c * np.ones((len(xs), ))
    # create color bar
    norm = plt.Normalize( vmin, vmax )
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array( plt_color )
    lc.set_linewidth(lw)
    return lc