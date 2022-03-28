"""
helper functions for plot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



def get_circle_coordinates(radius, center=None, n=50):
    """Get x,y,z coordinates for circle
    
    Args:
        radius (float): radius
        center (list): x,y,z coordinates of center, if None set to [0.0, 0.0]
    """
    # check if center is provided
    if center is None:
        center = [0.0, 0.0]
    thetas = np.linspace(0, 2*np.pi, n)
    coord = np.zeros((2,n))
    for idx, theta in enumerate(thetas):
        coord[:,idx] = [center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta)]
    return coord



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




def set_equal_axis(ax, xlims, ylims, zlims, scale=1.0, dim3=True):
    """Helper function to set equal axis
    
    Args:
        ax (Axes3DSubplot): matplotlib 3D axis, created by `ax = fig.add_subplot(projection='3d')`
        xlims (list): 2-element list containing min and max value of x
        ylims (list): 2-element list containing min and max value of y
        zlims (list): 2-element list containing min and max value of z
        scale (float): scaling factor along x,y,z
        dim3 (bool): whether to also set z-limits (True for 3D plots)
    """
    # compute max required range
    max_range = np.array([max(xlims)-min(xlims), max(ylims)-min(ylims), max(zlims)-min(zlims)]).max() / 2.0
    # compute mid-point along each axis
    mid_x = (max(xlims) + min(xlims)) * 0.5
    mid_y = (max(ylims) + min(ylims)) * 0.5
    mid_z = (max(zlims) + min(zlims)) * 0.5
    # set limits to axis
    if dim3==True:
        ax.set_box_aspect((max_range, max_range, max_range))
    ax.set_xlim(mid_x - max_range*scale, mid_x + max_range*scale)
    ax.set_ylim(mid_y - max_range*scale, mid_y + max_range*scale)
    if dim3==True:
        ax.set_zlim(mid_z - max_range*scale, mid_z + max_range*scale)
    return