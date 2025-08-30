import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
import shapely
import seaborn as sns
import matplotlib
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from cmcrameri import cm

def plot_field(grid, array, ax, norm = None, set_clim = None, cmap = 'viridis'):
    values = grid.map_mean_of_patch_nodes_to_patch(array)

    coords = []
    for patch in range(grid.number_of_patches):
        nodes = []

        for node in grid.nodes_at_patch[patch]:
            nodes.append(
                [grid.node_x[node], grid.node_y[node]]
            )

        coords.append(nodes)

    hulls = [shapely.get_coordinates(shapely.Polygon(i).convex_hull) for i in coords]
    polys = [plt.Polygon(shp) for shp in hulls]

    if norm is None:
        norm = matplotlib.colors.Normalize(vmin = np.min(array), vmax = np.max(array))

    collection = matplotlib.collections.PatchCollection(polys, cmap=cmap, norm=norm, edgecolor='face')
    collection.set_array(values)

    if set_clim is not None:
        collection.set_clim(**set_clim)

    im = ax.add_collection(collection)
    ax.autoscale()

    return im

key = 'rolige-brae'

with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
    grid = pickle.load(g)

sns.set(font_scale = 1.75)

fig, ax = plt.subplots(4, 2, figsize = (24, 24))

for a in ax.flatten():
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x * 1e-3)))
    a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x * 1e-3)))
    a.set_xlabel('Grid x (km)')
    a.set_ylabel('Grid y (km)')

cmap = cm.batlow

ax[0, 0].set_title('(a) Ice thickness (m)')
im = plot_field(grid, grid.at_node['ice_thickness'], ax[0, 0], cmap = cmap)
plt.colorbar(im, ax = ax[0, 0])

ax[0, 1].set_title('(b) Bedrock elevation (m)')
im = plot_field(grid, grid.at_node['bed_elevation'], ax[0, 1], cmap = cmap)
plt.colorbar(im, ax = ax[0, 1])

ax[1, 0].set_title('(c) Hydraulic potential (MPa)')
im = plot_field(grid, grid.at_node['hydraulic_potential'] * 1e-6, ax[1, 0], cmap = cmap)
plt.colorbar(im, ax = ax[1, 0])

Pi = grid.at_node['ice_thickness'] * 917 * 9.81
ax[1, 1].set_title('(d) Water pressure (MPa)')
im = plot_field(grid, Pi * 1e-6 - grid.at_node['effective_pressure'] * 1e-6, ax[1, 1], cmap = cmap)
plt.colorbar(im, ax = ax[1, 1])

ax[2, 0].set_title('(e) Sliding velocity (m a$^{-1}$)')
im = plot_field(grid, np.abs(grid.at_node['sliding_velocity']) * 31556926, ax[2, 0], cmap = cmap)
plt.colorbar(im, ax = ax[2, 0])

ax[2, 1].set_title('(f) Basal melt rate (m a$^{-1}$)')
im = plot_field(grid, grid.at_node['basal_melt_rate'] * 31556926, ax[2, 1], cmap = cmap)
plt.colorbar(im, ax = ax[2, 1])

ax[3, 0].set_title('(g) Frozen fringe thickness (m)')
im = plot_field(grid, grid.at_node['fringe_thickness'], ax[3, 0], cmap = cmap, norm = LogNorm())
plt.colorbar(im, ax = ax[3, 0])

ax[3, 1].set_title('(h) Dispersed layer thickness (m)')
im = plot_field(grid, grid.at_node['dispersed_thickness'], ax[3, 1], cmap = cmap, norm = LogNorm())
plt.colorbar(im, ax = ax[3, 1])

plt.tight_layout()
plt.savefig('figures/example_model_run.png', dpi = 300)
