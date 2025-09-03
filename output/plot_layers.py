import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ird_model.utils.plotting import plot_triangle_mesh
from landlab_triangle import TriangleModelGrid

for file in os.listdir('ird_model/models/checkpoints/sediment'):
    with open(f'ird_model/models/checkpoints/sediment/{file}', 'rb') as f:
        grid = pickle.load(f)
    fringe = grid.at_node['fringe_thickness']
    dispersed = grid.at_node['dispersed_thickness']

    fringe = np.where(fringe > np.percentile(fringe, 99), np.percentile(fringe, 99), fringe)

    plot_triangle_mesh(grid, fringe, cmap = 'viridis')
    plt.show()
