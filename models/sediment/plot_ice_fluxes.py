import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from glacierbento.utils import plot_triangle_mesh


with open('./models/inputs/bounds.pickle', 'rb') as b:
    bounds = pickle.load(b)

regions = {
    'rolige-brae': 'CE',
    'sermeq-avannarleq': 'CW',
    'charcot-gletscher': 'CE',
    'sydbrae': 'CE',
    'kangiata-nunaata-sermia': 'SW',
    'eielson-gletsjer': 'CE',
    'narsap-sermia': 'SW',
    'kangilernata-sermia': 'CW',
    'dode-brae': 'CE',
    'daugaard-jensen-gletsjer': 'CE',
    'vestfjord-gletsjer': 'CE',
    'sermeq-kullajeq': 'CW',
    'bredegletsjer': 'CE',
    'magga-dan-gletsjer': 'CE',
    'graah-gletscher': 'CE',
    'akullersuup-sermia': 'SW',
    'eqip-sermia': 'CW',
    'kista-dan-gletsjer': 'CE'
}

def find_terminus(grid, bounds):
    terminus = np.where(
        (grid.status_at_node != 0)
        & (grid.node_x > bounds[0])
        & (grid.node_x < bounds[1])
        & (grid.node_y > bounds[2])
        & (grid.node_y < bounds[3]),
        1,
        0
    )
    return terminus

for key, _ in regions.items():

    with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
        grid = pickle.load(g)

    with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'rb') as h:
        results = pickle.load(h)
    
    if key == 'eielson-gletsjer':
        terminus_a = find_terminus(grid, bounds[key][0])
        terminus_b = find_terminus(grid, bounds[key][1])
        terminus = np.logical_or(terminus_a, terminus_b)
    else:    
        terminus = find_terminus(grid, bounds[key])

    adjacent_nodes = grid.adjacent_nodes_at_node[terminus == 1]
    terminus_cells = np.unique(grid.cell_at_node[adjacent_nodes])
    terminus_cells = terminus_cells[terminus_cells != -1]
    terminus_faces = grid.faces_at_cell[terminus_cells]
    outflow_faces = np.where(grid.status_at_link[grid.link_at_face[terminus_faces]] != 0, 1, 0)
    face_width = grid.length_of_face[terminus_faces] * outflow_faces
    cell_outflow_width = np.sum(face_width, axis = 1)

    velocity = np.abs(grid.at_node['sliding_velocity'])[grid.node_at_cell] * 31556926
    surface_velocity = np.sqrt(grid.at_node['vx']**2 + grid.at_node['vy']**2)[grid.node_at_cell] * 31556926
    terminus_velocity = surface_velocity[terminus_cells] # TODO surface vs. sliding velocity?

    ice_flux = grid.at_node['ice_thickness'] * np.abs(grid.at_node['sliding_velocity']) * 31556926 * 917
    plot_triangle_mesh(grid, ice_flux, cmap = 'viridis', show = False)
    plt.title(f'{key} ice flux (m^2 / yr)', fontsize = 18)
    plt.show()
