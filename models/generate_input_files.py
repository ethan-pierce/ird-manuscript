#!/usr/bin/env python3
"""Generate input TOML files for each catchment."""

import os
import glob
import toml
from pathlib import Path

# Extract mesh parameters from generate_grids.py
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

# Extract discharge parameters from save_results_to_csv.py
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

bounds = {
    'rolige-brae': [6.0932e5, 6.3e5, -2.035e6, -2.03e6],
    'sermeq-avannarleq': [-2.063e5, -1.9985e5, -2.175e6, -2.1718e6],
    'charcot-gletscher': [5.35305e5, 5.36070e5, -1.88426e6, -1.88253e6],
    'sydbrae': [6.917e5, 6.966e5, -2.05300e6, -2.0503e6],
    'kangiata-nunaata-sermia': [-2.322e5, -2.2387e5, -2.82e6, -2.81829e6],
    'eielson-gletsjer': [6.0543e5, 6.0975e5, -1.9721e6, -1.9696e6],
    'narsap-sermia': [-2.4817e5, -2.4449e5, -2.77970e6, -2.77615e6],
    'kangilernata-sermia': [-2.0785e5, -2.0381e5, -2.19282e6, -2.18831e6],
    'dode-brae': [5.84982e5, 5.87e5, -2.057326e6, -2.0553e6],
    'daugaard-jensen-gletsjer': [5.5648e5, 5.6091e5, -1.89625e6, -1.8912e6],
    'vestfjord-gletsjer': [5.8578e5, 5.8787e5, -2.06246e6, -2.06e6], # TODO fix bounds
    'sermeq-kullajeq': [-1.99773e5, -1.98032e5, -2.18041e6, -2.17648e6],
    'bredegletsjer': [7.2777e5, 7.3120e5, -2.03134e6, -2.02869e6],
    'magga-dan-gletsjer': [6.65261e5, 6.6814e5, -2.08944e6, -2.08383e6],
    'graah-gletscher': [5.4728e5, 5.4974e5, -1.875739e6, -1.873994e6],
    'akullersuup-sermia': [-2.29522e5, -2.2673e5, -2.816803e6, -2.81362e6],
    'eqip-sermia': [-2.04326e5, -2.0160e5, -2.204225e6, -2.20054e6],
    'kista-dan-gletsjer': [6.6050e5, 6.6336e5, -2.08995e6, -2.0887e6]
}

# Region mapping
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

def create_toml_content(catchment_name):
    """Create TOML content for a specific catchment."""
    
    # Base template
    config = {
        "name": catchment_name.replace("-", " ").title(),
        "region": regions[catchment_name],
        "files": {
            "path_to_shapefile": f"data/shapefiles/{catchment_name}.geojson",
            "path_to_bedmachine": "data/BedMachineGreenland-v5.nc",
            "path_to_measures": "data/MEaSUREs_120m.nc",
            "path_to_basalmelt": "data/basalmelt.nc",
            "path_to_discharge": "data/gate_D.nc"
        },
        "grid": {
            "quality": 30,
            "mesh_size": mesh_params[catchment_name]["mesh_size"],
            "buffer": mesh_params[catchment_name]["buffer"],
            "tolerance": mesh_params[catchment_name]["tolerance"]
        },
        "inputs": {
            "surface.sigma": 7,
            "surface.truncate": 2,
            "sia.ice_flow_coefficient": 2.4e-24,
            "sia.glens_n": 3
        },
        "hydrology": {
            "sheet_flow_height.initial": 1e-3,
            "sheet_flow_height.minimum": 1e-3,
            "potential.initial": 1.0,
            "time_steps": 100,
            "dt": 3600,
            "max_diff.cutoff": 1000,
            "max_diff.percentile": 75,
            "cavity_spacing": 10,
            "bed_bump_height": 1.0,
            "sheet_conductivity": 0.05,
            "flow_exp_a": 3,
            "flow_exp_b": 2
        },
        "sediment": {
            "till_thickness.initial": 1e-3,
            "fringe_thickness.initial": 1e-3,
            "fringe_thickness.minimum": 1e-3,
            "dispersed_thickness.initial": 1e-3,
            "CFL": 0.2,
            "n_years": 400,
            "save_every": 1,
            "erosion.coefficient": 2.7e-7,
            "erosion.exponent": 2,
            "fringe.surface_energy": 0.034,
            "fringe.pore_throat_radius": 1e-6,
            "fringe.till_porosity": 0.35,
            "fringe.till_permeability": 4.1e-17,
            "fringe.till_grain_radius": 1.5e-4,
            "fringe.film_thickness": 1e-8,
            "fringe.alpha": 3.1,
            "fringe.beta": 0.53
        },
        "discharge": {
            "gate": discharge_gate[catchment_name],
            "terminus.min_x": bounds[catchment_name][0],
            "terminus.max_x": bounds[catchment_name][1],
            "terminus.min_y": bounds[catchment_name][2],
            "terminus.max_y": bounds[catchment_name][3],
            "fringe_thickness.cutoff": 99.5,
            "dispersed_thickness.cutoff": 99.5
        }
    }
    
    return config

def main():
    """Generate TOML files for all catchments."""
    
    # Get all geojson files in the catchments directory
    catchments_dir = Path("models/inputs/catchments")
    geojson_files = list(catchments_dir.glob("*.geojson"))
    
    # Create parameters directory if it doesn't exist
    parameters_dir = Path("models/inputs/parameters")
    parameters_dir.mkdir(exist_ok=True)
    
    print(f"Found {len(geojson_files)} catchment files")
    
    for geojson_file in geojson_files:
        catchment_name = geojson_file.stem  # Remove .geojson extension
        
        if catchment_name not in mesh_params:
            print(f"Warning: No mesh parameters found for {catchment_name}")
            continue
            
        if catchment_name not in discharge_gate:
            print(f"Warning: No discharge gate found for {catchment_name}")
            continue
            
        if catchment_name not in bounds:
            print(f"Warning: No bounds found for {catchment_name}")
            continue
        
        # Create TOML content
        config = create_toml_content(catchment_name)
        
        # Write to file
        output_file = parameters_dir / f"{catchment_name}.toml"
        with open(output_file, 'w') as f:
            toml.dump(config, f)
        
        print(f"Created {output_file}")
    
    print(f"\nGenerated {len(geojson_files)} TOML files in {parameters_dir}")

if __name__ == "__main__":
    main()
