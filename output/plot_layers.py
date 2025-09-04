import os
import pickle
import tomli
import numpy as np
import matplotlib.pyplot as plt

from ird_model.utils.static_grid import freeze_grid
from ird_model.utils.plotting import plot_triangle_mesh, plot_links
from landlab_triangle import TriangleModelGrid


def find_terminus(grid, config: dict):
    """Find the terminus of the glacier."""
    terminus = np.where(
        (grid.status_at_node != 0)
        & (grid.node_x > config['terminus.min_x'])
        & (grid.node_x < config['terminus.max_x'])
        & (grid.node_y > config['terminus.min_y'])
        & (grid.node_y < config['terminus.max_y']),
        1,
        0
    )
    
    # Get the indices of terminus nodes
    terminus_node_indices = np.where(terminus == 1)[0]
    
    # Get adjacent nodes for each terminus node
    adjacent_nodes = []
    for node_idx in terminus_node_indices:
        adjacent_nodes.extend(grid.adjacent_nodes_at_node[node_idx])
    
    # Convert to numpy array and get unique values
    adjacent_nodes = np.array(adjacent_nodes)
    adjacent_nodes = adjacent_nodes[adjacent_nodes != -1]  # Remove invalid nodes
    
    terminus_cells = np.unique(grid.cell_at_node[adjacent_nodes])
    terminus_cells = terminus_cells[terminus_cells != -1]
    return terminus, terminus_cells

for file in os.listdir('ird_model/models/checkpoints/sediment'):
    with open(f'ird_model/models/checkpoints/sediment/{file}', 'rb') as f:
        grid = pickle.load(f)
    
    with open(f'ird_model/models/inputs/config/{file.replace(".pickle", ".toml")}', 'rb') as f:
        config = tomli.load(f)

    config = config['fluxes']
    terminus = np.where(
        (grid.status_at_node != 0)
        & (grid.node_x > config['terminus.min_x'])
        & (grid.node_x < config['terminus.max_x'])
        & (grid.node_y > config['terminus.min_y'])
        & (grid.node_y < config['terminus.max_y']),
        1,
        0
    )
