"""Construct a mesh and interpolate input fields."""

import numpy as np
import jax.numpy as jnp
import xarray as xr
import rioxarray as rxr
import pickle
import shapely
import geopandas as gpd
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from landlab_triangle import TriangleModelGrid

from ird_model.components import ShallowIceApproximation
from ird_model.utils.core import Field
from ird_model.utils.static_grid import freeze_grid

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

SEC_PER_A = 31556926

def generate_mesh(
    shapefile: str,
    buffer: float,
    tolerance: float,
    quality: int,
    mesh_size: int
):
    """Generate a mesh from a shapefile."""
    polygon = _prepare_polygon(shapefile, buffer, tolerance)
    nodes_x, nodes_y, holes = _extract_nodes(polygon)
    max_area = polygon.area // mesh_size
    mesh = _build_grid(nodes_x, nodes_y, holes, quality, max_area)
    return mesh

def interpolate_fields(
    grid: TriangleModelGrid, 
    bedmachine: xr.Dataset,
    measures: xr.Dataset,
    basalmelt: xr.Dataset,
    config: dict
):
    """Interpolate input data onto a mesh."""
    names = ['ice_thickness', 'bed_elevation', 'surface_elevation']
    sigma = config['surface.sigma']
    truncate = config['surface.truncate']
    coeff = config['sia.ice_flow_coefficient']
    glens_n = config['sia.glens_n']

    if 'sia.max_surface_slope' in config:
        max_surface_slope = config['sia.max_surface_slope']
    else:
        max_surface_slope = None

    if 'max_surface_velocity' in config:
        max_surface_velocity = config['max_surface_velocity']
    else:
        max_surface_velocity = None

    _add_from_bedmachine(grid, bedmachine, ['thickness', 'bed', 'surface'], names, sigma = sigma, truncate = truncate)
    _add_measures_velocity(grid, measures)
    _add_basalmelt_2021(grid, basalmelt)
    _add_SIA_velocity(grid, coeff, glens_n, max_surface_slope, max_surface_velocity)
    return grid

def save_grid(
    grid: TriangleModelGrid, 
    path: str
):
    """Save a grid to a file."""
    with open(path, 'wb') as f:
        pickle.dump(grid, f)

def _prepare_polygon(shapefile: str, buffer: float, tolerance: float):
    """Prepare a smooth polygon from a shapefile."""
    with open(shapefile, 'rb') as f:
        geoseries = gpd.read_file(f)
    
    smooth_polygon = (
        geoseries.loc[0, 'geometry'].buffer(buffer, join_style = 'round')
        .buffer(-2 * buffer, join_style = 'round')
        .buffer(buffer, join_style = 'round')
    )

    polygon = shapely.simplify(smooth_polygon, tolerance = tolerance)
    
    if polygon.geom_type == 'MultiPolygon':
        polygon = polygon.geoms[0]
    
    return polygon

def _extract_nodes(polygon):
    """Extract arrays of nodes and holes from a polygon."""
    nodes_x = np.array(polygon.exterior.xy[0])
    nodes_y = np.array(polygon.exterior.xy[1])
    holes = polygon.interiors
    return nodes_x, nodes_y, holes

def _build_grid(nodes_x: np.ndarray, nodes_y: np.ndarray, holes: np.ndarray, quality: int, max_area: float):
    """Make a new TriangleModelGrid."""
    triangle_opts = 'pDevjz' + 'q' + str(quality) + 'a' + str(max_area)
    grid = TriangleModelGrid(
        (nodes_y, nodes_x), 
        holes = holes, 
        triangle_opts = triangle_opts,
        reorient_links = True
    )
    return grid

def _add_bedmachine_field(grid: TriangleModelGrid, dataset: xr.Dataset, field: str, name: str, smooth: bool = False, sigma: int = 7, truncate: int = 2):
    """Add a field to a grid."""
    nodata = dataset.attrs['no_data']
    clipped = dataset.rio.clip_box(
        minx = np.min(grid.node_x),
        maxx = np.max(grid.node_x),
        miny = np.min(grid.node_y),
        maxy = np.max(grid.node_y)
    )
    destination = np.vstack([grid.node_x, grid.node_y]).T
    values = clipped[field]
    values.rio.write_nodata(nodata, inplace = True)
    values_stacked = values.stack(z = ['x', 'y'])
    coords = np.vstack([values_stacked.coords['x'], values_stacked.coords['y']]).T
    values = values.values.flatten(order = 'F')

    if smooth:
        values = gaussian_filter(values, sigma = sigma, truncate = truncate)

    interp = RBFInterpolator(coords, values, neighbors = 9)
    interpolated = interp(destination)
    grid.add_field(name, interpolated, at = 'node')

def _add_from_bedmachine(grid: TriangleModelGrid, bedmachine: xr.Dataset, fields: list[str], names: list[str], sigma: int = 7, truncate: int = 2):
    """Add fields from BedMachine to a grid."""
    crs = bedmachine.attrs['proj4'].split('=')[-1]
    bedmachine.rio.write_crs(crs, inplace = True)

    for i, field in enumerate(fields):
        if field == 'surface':
            smooth = True
        else:
            smooth = False

        _add_bedmachine_field(grid, bedmachine, field, names[i], smooth, sigma, truncate)

def _add_measures_velocity(grid: TriangleModelGrid, measures: xr.Dataset):
    """Add a field from MEaSUREs to a grid."""
    measures.rio.write_crs('epsg:3413', inplace = True)

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

def _add_basalmelt_2021(grid: TriangleModelGrid, basalmelt: xr.Dataset):
    """Add basal melt to a grid."""
    basalmelt.rio.write_crs('epsg:3413', inplace = True)

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

def _add_basalmelt_2022(grid: TriangleModelGrid, basalmelt: xr.Dataset):
    """Add basal melt to a grid - use the 2022 data."""
    basalmelt.rio.write_crs('epsg:3413', inplace = True)
    
    clipped = basalmelt.rio.clip_box(
        minx = np.min(grid.node_x),
        maxx = np.max(grid.node_x),
        miny = np.min(grid.node_y),
        maxy = np.max(grid.node_y)
    )
    destination = np.vstack([grid.node_x, grid.node_y]).T
    
    melt = clipped['gfmelt'] + clipped['fricmelt']
    melt[:] *= 1 / SEC_PER_A
    melt[:] = np.where(melt < 0, 0, melt)
    melt.rio.write_nodata(np.nan, inplace = True)
    melt.rio.write_crs('epsg:3413', inplace = True)
    melt = melt.rio.interpolate_na(method = 'nearest')
    
    melt_stacked = melt.stack(z = ['x', 'y'])
    melt_coords = np.vstack([melt_stacked.coords['x'], melt_stacked.coords['y']]).T
    melt_values = melt.values.flatten(order = 'F')
    melt_interp = RBFInterpolator(melt_coords, melt_values, neighbors = 9)
    melt_interpolated = melt_interp(destination)
    grid.add_field('basal_melt_rate', melt_interpolated, at = 'node')

def _add_SIA_velocity(grid: TriangleModelGrid, coeff: float, glens_n: int, max_surface_slope: float, max_surface_velocity: float):
    """Add SIA velocity to a grid."""
    frozen = freeze_grid(grid)
    sia = ShallowIceApproximation(frozen)

    sia.params['ice_flow_coefficient'] = coeff
    sia.params['glens_n'] = glens_n
    sia.params['max_surface_slope'] = max_surface_slope
    fields = {
        'ice_thickness': Field(grid.at_node['ice_thickness'][:], 'm', 'node'),
        'surface_elevation': Field(grid.at_node['surface_elevation'][:], 'm', 'node'),
        'bed_elevation': Field(grid.at_node['bed_elevation'][:], 'm', 'node')
    }
    sia_output = sia.run_one_step(0.0, fields)

    U_deformation = frozen.map_mean_of_links_to_node(jnp.abs(sia_output['deformation_velocity'].value))
    U_surface = jnp.asarray(np.sqrt(grid.at_node['vx'][:]**2 + grid.at_node['vy'][:]**2))
    if max_surface_velocity is not None:
        U_surface = jnp.where(U_surface * 31556926 > max_surface_velocity, max_surface_velocity / 31556926, U_surface)
    U_sliding = jnp.where(U_surface > U_deformation, U_surface - U_deformation, 0.0)
    grid.add_field('sliding_velocity', U_sliding, at = 'node')
