import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
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
    'rolige-brae': [6.08e5, 6.3e5, -2.035e6, -2.03e6],
    'sermeq-avannarleq': [-2.063e5, -1.94e5, -2.175e6, -2.1695e6],
    'charcot-gletscher': [5.438e5, 5.453e5, -1.8834e6, -1.8814e6],
    'sydbrae': [6.917e5, 6.966e5, -2.05300e6, -2.0503e6],
    'kangiata-nunaata-sermia': [-2.322e5, -2.2387e5, -2.8211e6, -2.81829e6],
    'eielson-gletsjer': [[5.9432e5, 5.9806e5, -1.9938e6, -1.99107e6], [6.0470e5, 6.0975e5, -1.9721e6, -1.9684e6]],
    'narsap-sermia': [-2.4817e5, -2.4318e5, -2.78049e6, -2.77618e6],
    'kangilernata-sermia': [-2.0785e5, -2.0248e5, -2.19282e6, -2.187e6],
    'dode-brae': [5.84982e5, 5.87e5, -2.057326e6, -2.0553e6],
    'daugaard-jensen-gletsjer': [5.5327e5, 5.6091e5, -1.89852e6, -1.8912e6],
    'vestfjord-gletsjer': [5.849e5, 5.8915e5, -2.064e6, -2.0616e6],
    'sermeq-kullajeq': [-1.99773e5, -1.98032e5, -2.18107e6, -2.17603e6],
    'bredegletsjer': [7.2777e5, 7.3204e5, -2.03134e6, -2.02869e6],
    'magga-dan-gletsjer': [6.65261e5, 6.68950e5, -2.09014e6, -2.08383e6],
    'graah-gletscher': [5.48122e5, 5.50237e5, -1.877166e6, -1.874439e6],
    'akullersuup-sermia': [-2.29522e5, -2.26196e5, -2.816803e6, -2.813243e6],
    'eqip-sermia': [-2.04326e5, -2.01153e5, -2.204225e6, -2.200172e6],
    'kista-dan-gletsjer': [6.60337e5, 6.63701e5, -2.09062e6, -2.08841e6]
}

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
#     if key in ['charcot-gletscher', 'eielson-gletsjer', 'dode-brae', 'graah-gletscher']:

#         with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
#             grid = pickle.load(g)

#         with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'rb') as h:
#             results = pickle.load(h)

#         fig, ax = plt.subplots()

#         hf = results['fields'][-1]['fringe_thickness'].value
#         hd = results['fields'][-1]['dispersed_thickness'].value

#         im = plot_field(grid, hf, ax, norm = LogNorm(vmin = 1e-3, vmax = 10))

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
for key, _ in regions.items():

        with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'rb') as g:
            grid = pickle.load(g)

        with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'rb') as h:
            results = pickle.load(h)

        advector = TVDAdvector(freeze_grid(grid), fields_to_advect = ['fringe_thickness', 'dispersed_thickness'])
        advector = advector.initialize(results['fields'][-1])

        if key == 'eielson-gletsjer':
            terminus_a = find_terminus(grid, bounds[key][0])
            terminus_b = find_terminus(grid, bounds[key][1])
            terminus = np.logical_or(terminus_a, terminus_b)
        else:    
            terminus = find_terminus(grid, bounds[key])

        # adj_nodes = np.unique(grid.adjacent_nodes_at_node[terminus == 1])
        # adj_terminus = np.asarray([1 if i in adj_nodes else 0 for i in range(grid.number_of_nodes)])
        # cross_terminus_links = np.where(
        #     (terminus[grid.node_at_link_head] == 1) & (terminus[grid.node_at_link_tail] == 1), 1, 0
        # )

        terminus_links = grid.links_at_node[terminus == 1]
        flux_links = np.unique(terminus_links[terminus_links != -1])
        flux_links_bool = np.asarray([1 if i in flux_links else 0 for i in range(grid.number_of_links)])
        
        ffluxes = []
        dfluxes = []

        for i in results['fields']:
            hf = i['fringe_thickness'].value
            hf_links = grid.map_mean_of_link_nodes_to_link(hf)[flux_links]
            link_velocity = np.abs(advector._link_velocity)[flux_links]
            hf_flux = hf_links * link_velocity * 0.65 * 2700 * 31556926 * grid.length_of_face[grid.face_at_link[flux_links]]
            ffluxes.append(np.sum(hf_flux))

            hd = i['dispersed_thickness'].value
            hd_links = grid.map_mean_of_link_nodes_to_link(hd)[flux_links]
            link_velocity = np.abs(advector._link_velocity)[flux_links]
            hd_flux = hd_links * link_velocity * 0.05 * 2700 * 31556926 * grid.length_of_face[grid.face_at_link[flux_links]]
            dfluxes.append(np.sum(hd_flux))


        fig, ax = plt.subplots(figsize = (12, 6))
        time = np.array(results['time']) / 31556926
        plt.plot(time, ffluxes, label = 'Frozen fringe')
        plt.plot(time, dfluxes, label = 'Dispersed layer')
        plt.xlabel('Year of simulation')
        plt.ylabel('Sediment load (kg a$^{-1}$)')
        plt.legend()
        plt.title(f'{key.replace("-", " ").title()} sediment load')
        plt.show()