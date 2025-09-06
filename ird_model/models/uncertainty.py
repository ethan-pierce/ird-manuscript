"""Run each model over a range of uncertain parameters."""

import numpy as np
import pandas as pd
import xarray as xr
import time
import pickle
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, pareto

from ird_model.models.run_models import parse_args, load_config, load_stage_data
from ird_model.models.mesh import generate_mesh, interpolate_fields
from ird_model.models.hydrology import run_to_steady_state
from ird_model.models.sediment import run_sediment_transport
from ird_model.models.fluxes import calc_fluxes
from landlab_triangle import TriangleModelGrid

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc = mean, scale = sd)

distributions = {
    'fringe_till_porosity': get_truncated_normal(mean = 0.35, sd = 0.075, low = 0.2, upp = 0.5),
    'dispersed_concentration': pareto(b = 1.6, scale = 0.01),
    'fringe_till_grain_radius': get_truncated_normal(mean = 1.5e-4, sd = 5e-5, low = 5e-5, upp = 5e-4),
    'fringe_film_thickness': get_truncated_normal(mean = 1e-8, sd = 5e-8, low = 1e-9, upp = 1e-7),
    'critical_depth': get_truncated_normal(mean = 100, sd = 50, low = 10, upp = 200),
    'ice_flow_coefficient': get_truncated_normal(mean = 2.4e-24, sd = 5e-24, low = 3.5e-25, upp = 2.4e-24),
    'sheet_conductivity': get_truncated_normal(mean = 0.05, sd = 0.02, low = 0.01, upp = 0.1),
    'erosion_coefficient': get_truncated_normal(mean = 2.7e-7, sd = 5e-7, low = 1e-7, upp = 5e-7),
    'erosion_exponent': get_truncated_normal(mean = 2, sd = 0.25, low = 1.5, upp = 2.5),
}

def make_df(grid: TriangleModelGrid, config: dict):
    """Make a dataframe from the model output."""
    fringe_flux, dispersed_flux, ice_flux = calc_fluxes(grid, config)
    discharge_df = pd.read_csv('ird_model/models/inputs/gate_D.csv', header = 0)
    ice_discharge = discharge_df[str(config['fluxes']['gate'])].iloc[1744:2853].median()

    df = pd.DataFrame(
        {
            'glacier': config['name'],
            'region': config['region'],
            'area': np.sum(grid.cell_area_at_node),
            'ice_discharge': np.float64(ice_discharge),
            'model_ice_flux': np.float64(ice_flux),
            'fringe_flux': np.float64(fringe_flux),
            'dispersed_flux': np.float64(dispersed_flux),
            'ice_flow_coefficient': np.float64(config['inputs']['sia.ice_flow_coefficient']),
            'erosion_coefficient': np.float64(config['sediment']['erosion.coefficient']),
            'erosion_exponent': np.float64(config['sediment']['erosion.exponent']),
            'sheet_conductivity': np.float64(config['hydrology']['sheet_conductivity']),
            'fringe_porosity': np.float64(config['sediment']['fringe.till_porosity']),
            'fringe_grain_radius': np.float64(config['sediment']['fringe.till_grain_radius']),
            'fringe_film_thickness': np.float64(config['sediment']['fringe.film_thickness']),
            'dispersed_concentration': np.float64(config['fluxes']['dispersed.concentration']),
            'critical_depth': np.float64(config['sediment']['critical_depth'])
        },
        index = [0]
    )
    return df

def run_all_stages(config: dict):
    """Run all stages of the model."""
    grid = generate_mesh(
        shapefile = config['files']['path_to_shapefile'],
        buffer = config['grid']['buffer'],
        tolerance = config['grid']['tolerance'],
        quality = config['grid']['quality'],
        mesh_size = config['grid']['mesh_size']
    )
    grid = interpolate_fields(
        grid = grid,
        bedmachine = xr.open_dataset(config['files']['path_to_bedmachine']),
        measures = xr.open_dataset(config['files']['path_to_measures']),
        basalmelt = xr.open_dataset(config['files']['path_to_basalmelt']),
        config = config['inputs']
    )
    grid = run_to_steady_state(grid, config['hydrology'])
    grid = run_sediment_transport(grid, config['sediment'])
    df = make_df(grid, config)

    return df

def run_uncertainty(grid: TriangleModelGrid, config: dict):
    """Run each model over a range of uncertain parameters."""
    dfs = []
    errors = []
    
    n = 10
    for i in range(n):
        print(f'RUNNING MONTE CARLO ITERATION {i}')
        try:
            if i == 0:
                start_time = time.time()

            parameters = {key: val.rvs() for key, val in distributions.items()}

            config['sediment']['fringe.till_porosity'] = parameters['fringe_till_porosity']
            config['fluxes']['dispersed.concentration'] = parameters['dispersed_concentration']
            config['sediment']['fringe.till_grain_radius'] = parameters['fringe_till_grain_radius']
            config['sediment']['fringe.film_thickness'] = parameters['fringe_film_thickness']
            config['sediment']['critical_depth'] = parameters['critical_depth']
            config['inputs']['sia.ice_flow_coefficient'] = parameters['ice_flow_coefficient']
            config['hydrology']['sheet_conductivity'] = parameters['sheet_conductivity']
            config['sediment']['erosion.coefficient'] = parameters['erosion_coefficient']
            config['sediment']['erosion.exponent'] = parameters['erosion_exponent']

            df = run_all_stages(config)
            dfs.append(df)

            if i == 0:
                end_time = time.time()
                print(f'Time taken for 1 iteration: {end_time - start_time} seconds')
                print(f'Total time estimated: {(end_time - start_time) * n / 60 / 60} hours')

        except Exception as e:
            print(f'Error on iteration {i}: {e}')
            errors.append(parameters)
            continue

    return pd.concat(dfs), errors

if __name__ == "__main__":
    config_path, _ = parse_args()
    config = load_config(config_path)
    grid = load_stage_data('sediment', config)
    results, errors = run_uncertainty(grid, config)

    total_flux = (results['fringe_flux'] + results['dispersed_flux']) * 1e-9 # Mt/yr
    median = total_flux.median()
    pct25 = total_flux.quantile(0.25)
    pct75 = total_flux.quantile(0.75)
    print(f'Median: {median} Mt/yr')
    print(f'25th percentile: {pct25} Mt/yr')
    print(f'75th percentile: {pct75} Mt/yr')

    shortname = config["name"].lower().replace(" ", "-")
    results.to_csv(f'ird_model/models/checkpoints/uncertainty/{shortname}.csv', index = False)

    with open(f'ird_model/models/checkpoints/uncertainty/{shortname}-errors.pickle', 'wb') as f:
        pickle.dump(errors, f)