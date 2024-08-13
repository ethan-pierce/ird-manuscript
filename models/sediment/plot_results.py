import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd
import pickle
import shapely
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections
from matplotlib.colors import LogNorm, TwoSlopeNorm
from landlab.plot import imshow_grid
from glacierbento.utils import plot_triangle_mesh, plot_links, freeze_grid
from glacierbento.components import TVDAdvector

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

bounds = {
    'rolige-brae': [6.0932e5, 6.3e5, -2.035e6, -2.03e6],
    'sermeq-avannarleq': [-2.063e5, -1.9985e5, -2.175e6, -2.1718e6],
    'charcot-gletscher': [5.438e5, 5.453e5, -1.8834e6, -1.8814e6],
    'sydbrae': [6.917e5, 6.966e5, -2.05300e6, -2.0503e6],
    'kangiata-nunaata-sermia': [-2.322e5, -2.2387e5, -2.82e6, -2.81829e6],
    'eielson-gletsjer': [[5.9641e5, 5.9806e5, -1.9938e6, -1.99205e6], [6.0543e5, 6.0975e5, -1.9721e6, -1.9696e6]],
    'narsap-sermia': [-2.4817e5, -2.4449e5, -2.77970e6, -2.77615e6],
    'kangilernata-sermia': [-2.0785e5, -2.0381e5, -2.19282e6, -2.18831e6],
    'dode-brae': [5.84982e5, 5.87e5, -2.057326e6, -2.0553e6],
    'daugaard-jensen-gletsjer': [5.5648e5, 5.6091e5, -1.89625e6, -1.8912e6],
    'vestfjord-gletsjer': [5.8578e5, 5.8787e5, -2.06246e6, -2.06e6],
    'sermeq-kullajeq': [-1.99773e5, -1.98032e5, -2.18041e6, -2.17648e6],
    'bredegletsjer': [7.2777e5, 7.3120e5, -2.03134e6, -2.02869e6],
    'magga-dan-gletsjer': [6.65261e5, 6.6814e5, -2.08944e6, -2.08383e6],
    'graah-gletscher': [5.4845e5, 5.50237e5, -1.877166e6, -1.87488e6],
    'akullersuup-sermia': [-2.29522e5, -2.2673e5, -2.816803e6, -2.81362e6],
    'eqip-sermia': [-2.04326e5, -2.0160e5, -2.204225e6, -2.20054e6],
    'kista-dan-gletsjer': [6.6050e5, 6.6336e5, -2.08995e6, -2.0887e6]
}

discharge_gate = {
    'rolige-brae': 150,
    'sermeq-avannarleq': 167,
    'charcot-gletscher': 139,
    'sydbrae': 152,
    'kangiata-nunaata-sermia': 275,
    'eielson-gletsjer': 144,
    'narsap-sermia': 262,
    'kangilernata-sermia': 177,
    'dode-brae': 160, # proxy
    'daugaard-jensen-gletsjer': 140,
    'vestfjord-gletsjer': 154,
    'sermeq-kullajeq': 169,
    'bredegletsjer': 151,
    'magga-dan-gletsjer': 156,
    'graah-gletscher': 138,
    'akullersuup-sermia': 270,
    'eqip-sermia': 180,
    'kista-dan-gletsjer': 158
}
discharge_data = pd.read_csv('/home/egp/repos/local/ice-discharge/dataverse_files/gate_D.csv', header = 0)
discharge = {key: discharge_data[str(val)].iloc[2666:].mean() for key, val in discharge_gate.items()}


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

    collection = matplotlib.collections.PatchCollection(polys, cmap=cmap, norm=norm)
    collection.set_array(values)

    if set_clim is not None:
        collection.set_clim(**set_clim)

    im = ax.add_collection(collection)
    ax.autoscale()

    return im

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

# Plot individual catchments
# for key, _ in bounds.items():
#     # if regions[key] == 'CE':
#     if key == 'sermeq-avannarleq':

#         with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
#             grid = pickle.load(g)

#         with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'rb') as h:
#             results = pickle.load(h)

#         fig, ax = plt.subplots()

#         hf = results['fields'][-1]['fringe_thickness'].value
#         hd = results['fields'][-1]['dispersed_thickness'].value

#         im = plot_field(grid, hd, ax, set_clim = {'vmax': np.percentile(hd, 95)})

#         plt.colorbar(im)
#         plt.title(f'{key.replace("-", " ").title()} fringe thickness (m)', fontsize = 18)
#         plt.show()


# Plot regions
# fig, ax = plt.subplots()

# for key, _ in bounds.items():
#     if regions[key] == 'CE':

#         with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
#             grid = pickle.load(g)

#         with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'rb') as h:
#             results = pickle.load(h)

#         hf = results['fields'][-1]['fringe_thickness'].value
#         hd = results['fields'][-1]['dispersed_thickness'].value

#         im = plot_field(grid, hf, ax, norm = LogNorm(vmin = 1e-3, vmax = 10))

# plt.colorbar(im)
# plt.title('Kangertittivaq fringe thickness (m)', fontsize = 18)
# plt.show()

# Plot fluxes
fluxes_df = pd.DataFrame(columns = ['glacier', 'ice_flux', 'fringe_flux', 'dispersed_flux', 'total_flux'])
fluxes_df['glacier'] = regions.keys()
fluxes_df['ice_flux'] = [discharge[key] for key in regions.keys()]

for key, _ in regions.items():

    with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
        grid = pickle.load(g)

    with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'rb') as h:
        results = pickle.load(h)

    fig, ax = plt.subplots()
    im = plot_field(grid, results['fields'][-1]['fringe_thickness'].value, ax, norm = LogNorm(vmin = 1e-3, vmax = 10))
    plt.title(f'{key.replace("-", " ").title()} fringe thickness (m)', fontsize = 18)
    plt.colorbar(im)
    plt.show()
    continue

    advector = TVDAdvector(freeze_grid(grid), fields_to_advect = ['fringe_thickness', 'dispersed_thickness'])
    advector = advector.initialize(results['fields'][-1])

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

    ifluxes = []
    ffluxes = []
    dfluxes = []

    for i in results['fields']:
        hi = i['ice_thickness'].value[grid.node_at_cell][terminus_cells]
        iflux = np.sum((hi * terminus_velocity * 917 * cell_outflow_width))

        hf = i['fringe_thickness'].value[grid.node_at_cell][terminus_cells]
        hf_flux = np.sum((hf * terminus_velocity * 0.65 * 2700 * cell_outflow_width))

        hd = i['dispersed_thickness'].value[grid.node_at_cell][terminus_cells]
        hd_flux = np.sum((hd * terminus_velocity * 0.05 * 2700 * cell_outflow_width))

        ifluxes.append(iflux)
        ffluxes.append(hf_flux)
        dfluxes.append(hd_flux)

    fluxes_df.loc[fluxes_df['glacier'] == key, 'model_ice_flux'] = ifluxes[-1]
    fluxes_df.loc[fluxes_df['glacier'] == key, 'fringe_flux'] = ffluxes[-1]
    fluxes_df.loc[fluxes_df['glacier'] == key, 'dispersed_flux'] = dfluxes[-1]
    fluxes_df.loc[fluxes_df['glacier'] == key, 'total_flux'] = ffluxes[-1] + dfluxes[-1]

    fig, ax = plt.subplots(figsize = (12, 6))
    time = np.array(results['time']) / 31556926
    plt.plot(time, ffluxes, label = 'Frozen fringe')
    plt.plot(time, dfluxes, label = 'Dispersed layer')
    plt.xlabel('Year of simulation')
    plt.ylabel('Sediment load (kg a$^{-1}$)')
    plt.legend()
    plt.title(f'{key.replace("-", " ").title()} sediment load')
    plt.show()

# fluxes_df.to_csv('./models/sediment/outputs/fluxes.csv', index = False)

# plt.scatter(fluxes_df['ice_flux'], fluxes_df['total_flux'])
# plt.xlabel('Ice flux (m$^3$ s$^{-1}$)')
# plt.ylabel('Sediment flux (kg a$^{-1}$)')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
