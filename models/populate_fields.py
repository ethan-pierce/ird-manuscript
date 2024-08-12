"""Populate initial conditions for Greenland models."""

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import pickle
import rasterio as rio
import xarray as xr
import rioxarray as rxr
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from landlab import TriangleModelGrid
from landlab.plot import imshow_grid
from glacierbento.utils import freeze_grid

SEC_PER_A = 31556926

with open('./models/inputs/grids/empty-grids.pickle', 'rb') as f:
    grids = pickle.load(f)

with open('./models/inputs/BedMachineGreenland-v5.nc', 'rb') as f:
    print('Adding data from BedMachine...')
    bedmachine = xr.open_dataset(f)
    crs = bedmachine.attrs['proj4'].split('=')[-1]
    nodata = bedmachine.attrs['no_data']

    bedmachine.rio.write_crs(crs, inplace = True)

    for key, grid in grids.items():
        clipped = bedmachine.rio.clip_box(
            minx = np.min(grid.node_x),
            maxx = np.max(grid.node_x),
            miny = np.min(grid.node_y),
            maxy = np.max(grid.node_y)
        )

        destination = np.vstack([grid.node_x, grid.node_y]).T

        ice_thickness = clipped['thickness']
        ice_thickness.rio.write_nodata(nodata, inplace = True)
        ice_stacked = ice_thickness.stack(z = ['x', 'y'])
        ice_coords = np.vstack([ice_stacked.coords['x'], ice_stacked.coords['y']]).T
        ice_values = ice_thickness.values.flatten(order = 'F')
        ice_interp = RBFInterpolator(ice_coords, ice_values, neighbors = 9)
        ice_interpolated = ice_interp(destination)
        grid.add_field('ice_thickness', ice_interpolated, at = 'node')

        bed_elevation = clipped['bed']
        bed_elevation.rio.write_nodata(nodata, inplace = True)
        bed_stacked = bed_elevation.stack(z = ['x', 'y'])
        bed_coords = np.vstack([bed_stacked.coords['x'], bed_stacked.coords['y']]).T
        bed_values = bed_elevation.values.flatten(order = 'F')
        bed_interp = RBFInterpolator(bed_coords, bed_values, neighbors = 9)
        bed_interpolated = bed_interp(destination)
        grid.add_field('bed_elevation', bed_interpolated, at = 'node')

        surface_elevation = clipped['surface']
        surface_elevation.rio.write_nodata(nodata, inplace = True)
        surface_stacked = surface_elevation.stack(z = ['x', 'y'])
        surface_coords = np.vstack([surface_stacked.coords['x'], surface_stacked.coords['y']]).T
        surface_values = gaussian_filter(surface_elevation.values, sigma = 7, truncate = 2).flatten(order = 'F')
        surface_interp = RBFInterpolator(surface_coords, surface_values, neighbors = 9)
        surface_interpolated = surface_interp(destination)
        grid.add_field('surface_elevation', surface_interpolated, at = 'node')

with open('./models/inputs/MEaSUREs_120m.nc', 'rb') as f:
    print('Adding data from MEaSUREs...')
    measures = xr.open_dataset(f)
    measures.rio.write_crs('epsg:3413', inplace = True)

    for key, grid in grids.items():
        clipped = measures.rio.clip_box(
            minx = np.min(grid.node_x),
            maxx = np.max(grid.node_x),
            miny = np.min(grid.node_y),
            maxy = np.max(grid.node_y)
        )

        destination = np.vstack([grid.node_x, grid.node_y]).T

        vx = clipped['vx']
        vx[:] *= 1 / SEC_PER_A
        vx.rio.write_nodata(np.nan, inplace = True)
        vx = vx.rio.interpolate_na(method = 'nearest')

        vx_stacked = vx.stack(z = ['x', 'y'])
        vx_coords = np.vstack([vx_stacked.coords['x'], vx_stacked.coords['y']]).T
        vx_values = vx.values.flatten(order = 'F')
        vx_interp = RBFInterpolator(vx_coords, vx_values, neighbors = 9)
        vx_interpolated = vx_interp(destination)
        grid.add_field('vx', vx_interpolated, at = 'node')

        vy = clipped['vy']
        vy[:] *= 1 / SEC_PER_A
        vy.rio.write_nodata(np.nan, inplace = True)
        vy = vy.rio.interpolate_na(method = 'nearest')

        vy_stacked = vy.stack(z = ['x', 'y'])
        vy_coords = np.vstack([vy_stacked.coords['x'], vy_stacked.coords['y']]).T
        vy_values = vy.values.flatten(order = 'F')
        vy_interp = RBFInterpolator(vy_coords, vy_values, neighbors = 9)
        vy_interpolated = vy_interp(destination)
        grid.add_field('vy', vy_interpolated, at = 'node')

with open('./models/inputs/basalmelt.nc', 'rb') as f:
    print('Adding data from Karlsson et al. (2021)...')
    basalmelt = xr.open_dataset(f)
    basalmelt.rio.write_crs('epsg:3413', inplace = True)

    for key, grid in grids.items():
        clipped = basalmelt.rio.clip_box(
            minx = np.min(grid.node_x),
            maxx = np.max(grid.node_x),
            miny = np.min(grid.node_y),
            maxy = np.max(grid.node_y)
        )

        destination = np.vstack([grid.node_x, grid.node_y]).T

        melt = clipped['totalmelt']
        melt[:] *= 1 / SEC_PER_A
        melt[:] = np.where(melt < 0, 0, melt)
        melt.rio.write_nodata(np.nan, inplace = True)
        melt = melt.rio.interpolate_na(method = 'nearest')

        melt_stacked = melt.stack(z = ['x', 'y'])
        melt_coords = np.vstack([melt_stacked.coords['x'], melt_stacked.coords['y']]).T
        melt_values = melt.values.flatten(order = 'F')
        melt_interp = RBFInterpolator(melt_coords, melt_values, neighbors = 9)
        melt_interpolated = melt_interp(destination)
        grid.add_field('basal_melt_rate', melt_interpolated, at = 'node')

with open('./models/inputs/grids/initial-conditions.pickle', 'wb') as f:
    pickle.dump(grids, f)