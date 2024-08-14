"""Generate grids for Greenland catchments."""

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import pickle
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt

from landlab import TriangleModelGrid
from landlab.plot import imshow_grid
from glacierbento.utils import freeze_grid


QUALITY = 30

mesh_params = {
    'rolige-brae': {'mesh_size': 1e4, 'buffer': 750, 'tolerance': 20},
    'sermeq-avannarleq': {'mesh_size': 750, 'buffer': 750, 'tolerance': 30},
    'charcot-gletscher': {'mesh_size': 1e4, 'buffer': 500, 'tolerance': 20},
    'sydbrae': {'mesh_size': 1e4, 'buffer': 500, 'tolerance': 20},
    'kangiata-nunaata-sermia': {'mesh_size': 5e3, 'buffer': 1000, 'tolerance': 20},
    'eielson-gletsjer': {'mesh_size': 3e3, 'buffer': 750, 'tolerance': 15},
    'narsap-sermia': {'mesh_size': 4e3, 'buffer': 500, 'tolerance': 20},
    'kangilernata-sermia': {'mesh_size': 5e3, 'buffer': 1200, 'tolerance': 30},
    'dode-brae': {'mesh_size': 1e4, 'buffer': 500, 'tolerance': 20},
    'daugaard-jensen-gletsjer': {'mesh_size': 5e3, 'buffer': 1700, 'tolerance': 20},
    'vestfjord-gletsjer': {'mesh_size': 5e3, 'buffer': 500, 'tolerance': 15},
    'sermeq-kullajeq': {'mesh_size': 2e3, 'buffer': 400, 'tolerance': 15},
    'bredegletsjer': {'mesh_size': 1e4, 'buffer': 500, 'tolerance': 20},
    'magga-dan-gletsjer': {'mesh_size': 8e3, 'buffer': 250, 'tolerance': 20},
    'graah-gletscher': {'mesh_size': 1e4, 'buffer': 500, 'tolerance': 20},
    'akullersuup-sermia': {'mesh_size': 8e3, 'buffer': 500, 'tolerance': 15},
    'eqip-sermia': {'mesh_size': 1e4, 'buffer': 250, 'tolerance': 15},
    'kista-dan-gletsjer': {'mesh_size': 1e4, 'buffer': 500, 'tolerance': 20}
}

grids = {}

for key, val in mesh_params.items():
    file = key + '.geojson'
    glacier = file.replace('.geojson', '').replace('-', ' ').title()
    print('Constructing mesh for ' + glacier + '...')

    with open('./models/inputs/catchments/' + file) as f:
        geoseries = gpd.read_file(f)

    buffer = mesh_params[key]['buffer']
    tolerance = mesh_params[key]['tolerance']

    smooth_polygon = (
        geoseries.loc[0, 'geometry'].buffer(buffer, join_style = 'round')
        .buffer(-2 * buffer, join_style = 'round')
        .buffer(buffer, join_style = 'round')
    )
    polygon = shapely.simplify(smooth_polygon, tolerance = tolerance)

    if polygon.geom_type == 'MultiPolygon':
        polygon = polygon.geoms[0]

    nodes_x = np.array(polygon.exterior.xy[0])
    nodes_y = np.array(polygon.exterior.xy[1])
    holes = polygon.interiors

    max_area = polygon.area // mesh_params[key]['mesh_size']
    triangle_opts = 'pDevjz' + 'q' + str(QUALITY) + 'a' + str(max_area)

    grid = TriangleModelGrid(
        (nodes_y, nodes_x), 
        holes = holes, 
        triangle_opts = triangle_opts,
        # reorient_links = True,
        # sorted = True
    )
    print('Mesh contains ' + str(grid.number_of_nodes) + ' nodes.')
    print('Maximum cell area is ' + str(np.max(grid.cell_area_at_node)) + ' m^2.')

    grids[key] = grid

with open('./models/inputs/grids/empty-grids.pickle', 'wb') as f:
    pickle.dump(grids, f)
