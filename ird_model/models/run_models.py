"""Main script for running all models with checkpointing support."""

import os
import argparse
import tomli
import pickle
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

from ird_model.models.mesh import generate_mesh, interpolate_fields
from ird_model.models.hydrology import run_to_steady_state
from ird_model.models.sediment import run_sediment_transport
from ird_model.models.fluxes import calc_fluxes

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description = "Run glacier models with parameters from a TOML file."
    )
    parser.add_argument(
        "config",
        type = str,
        help = "Path to TOML configuration file",
        nargs = 1
    )
    parser.add_argument(
        "--stages",
        type = str,
        nargs = "+",
        choices = ["grid", "hydrology", "sediment", "fluxes"],
        help = "Stages to run",
        default = ["grid", "hydrology", "sediment", "fluxes"]
    )
    args = parser.parse_args()
    return args.config[0], args.stages

def load_config(path):
    """Load configuration from TOML file."""
    with open(path, "rb") as f:
        return tomli.load(f)

def get_checkpoint_path(stage, config):
    """Get path for stage checkpoint."""
    return Path(f"ird_model/models/checkpoints/{stage}/{Path(config['name'].lower().replace(' ', '-'))}.pickle")

def load_stage_data(stage, config):
    """Load data from a stage's checkpoint."""
    checkpoint = get_checkpoint_path(stage, config)
    if checkpoint.exists():
        print(f"Loading checkpoint for {stage}")
        with open(checkpoint, "rb") as f:
            return pickle.load(f)
    return None

def run_stage(stage, config, prev_stage = None):
    """Run a model stage with checkpoint support."""
    os.makedirs(f"ird_model/models/checkpoints/{stage}", exist_ok = True)
    checkpoint = get_checkpoint_path(stage, config)

    # Load previous stage data if needed
    prev_data = None
    if prev_stage:
        prev_data = load_stage_data(prev_stage, config)
        if not prev_data:
            raise RuntimeError(f"{prev_stage} data required for {stage} stage")

    # Run the stage
    if stage == "grid":
        print("Generating mesh")
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
        data = grid
        print("Number of nodes: ", grid.number_of_nodes)
        print("Number of elements: ", grid.number_of_cells)

    elif stage == "hydrology":
        print("Running hydrology models")
        grid = prev_data
        data = run_to_steady_state(grid, config['hydrology'])

    elif stage == "sediment":
        print("Running sediment transport models")
        grid = prev_data
        data = run_sediment_transport(grid, config['sediment'])

    elif stage == "fluxes":
        fringe_flux, dispersed_flux = calc_fluxes(prev_data, config)
        discharge_df = pd.read_csv('ird_model/models/inputs/gate_D.csv', header = 0)
        ice_discharge = discharge_df[str(config['fluxes']['gate'])].iloc[1744:2853].mean()

        df = pd.DataFrame(
            {
                'glacier': config['name'],
                'region': config['region'],
                'area': np.sum(prev_data.cell_area_at_node),
                'ice_discharge': np.float64(ice_discharge),
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
        data = df
    
    # Save checkpoint
    with open(checkpoint, "wb") as f:
        pickle.dump(data, f)
    
    return data

if __name__ == "__main__":
    config_path, stages = parse_args()
    config = load_config(config_path)
    
    # Map of stages to their dependencies
    stage_deps = {
        "grid": None,
        "hydrology": "grid",
        "sediment": "hydrology",
        "fluxes": "sediment"
    }
    
    # Run requested stages in sequence
    for stage in stages:
        try:
            data = run_stage(stage, config, stage_deps[stage])
        except Exception as e:
            print(f"Error in {stage} stage: {str(e)}")
            raise
