"""Main script for running all models with checkpointing support."""

import os
import argparse
import tomli
import pickle
import xarray as xr
from pathlib import Path

from ird_model.models.mesh import generate_mesh, interpolate_fields, save_grid
from ird_model.models.hydrology import run_to_steady_state

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
        choices = ["grid", "hydrology", "sediment", "discharge"],
        help = "Stages to run",
        default = ["grid", "hydrology", "sediment", "discharge"]
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
            sigma = config['inputs']['surface.sigma'],
            truncate = config['inputs']['surface.truncate'],
            ice_flow_coefficient = config['inputs']['sia.ice_flow_coefficient'],
            glens_n = config['inputs']['sia.glens_n']
        )
        data = grid
        print("Number of nodes: ", grid.number_of_nodes)
        print("Number of elements: ", grid.number_of_cells)

    elif stage == "hydrology":
        print("Running hydrology models")
        grid = prev_data
        data = run_to_steady_state(grid, config['hydrology'])

    elif stage == "sediment":
        # TODO: Implement sediment model using prev_data (hydrology)
        data = {}
    elif stage == "discharge":
        # TODO: Implement discharge calculations using prev_data (sediment)
        data = {}
    
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
        "discharge": "sediment"
    }
    
    # Run requested stages in sequence
    for stage in stages:
        try:
            data = run_stage(stage, config, stage_deps[stage])
        except Exception as e:
            print(f"Error in {stage} stage: {str(e)}")
            raise
