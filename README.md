# Sediment Transport by Greenland's Icebergs

Source code and model output for the manuscript: **"Sediment transport by Greenland's icebergs"** currently under review.

This work is included as a chapter in the following PhD dissertation:  
**"From bedrock to bergs: sediment entrainment beneath glaciers and ice sheets"**

## Contact

**Ethan Pierce**  
Email: ethan.g.pierce@dartmouth.edu  
GitHub: [via github](https://github.com/ethan-pierce/ird-manuscript)

For questions about the research or code, feel free to reach out!

## Installation

### Option 1: Using Pixi (Recommended)

```bash
# Install dependencies with pixi
pixi install
```

### Option 2: Using pip

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Option 3: Using conda

```bash
# Create conda environment
conda env create -f environment.yml  # If available
# OR install manually:
conda install numpy scipy matplotlib pandas xarray geopandas landlab jax jaxlib

# Install the package
pip install -e .
```

### Download Required Data

After installation, download the required data files:

```bash
# Set your Earthdata credentials (replace with your actual credentials)
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"

# Download data files
bash download_data_files.sh
```

**Note:** The data files are large (>10GB total) and require NASA Earthdata credentials for download.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `ird_model/` | Main Python package containing all model components |
| `ird_model/components/` | Process models for erosion, hydrology, and sediment transport |
| `ird_model/models/` | Main model execution scripts and configuration |
| `ird_model/models/inputs/` | Model configuration files (TOML) and input data |
| `ird_model/models/checkpoints/` | Model checkpoint files (ignored by git) |
| `ird_model/utils/` | Utility functions for grid generation, numerics, and plotting |
| `data/` | Field measurements, catchment shapefiles, and ice discharge data |
| `data/shapefiles/` | GeoJSON shapefiles for glacier catchments |
| `output/` | Model output files and analysis scripts |
| `run_all_*.sh` | Shell scripts for running models across all catchments |
| `download_data_files.sh` | Script for downloading required data files |

## Output Data

The `output/` directory contains two key CSV files with model results:

### `catchment_sediment_flux_statistics.csv`
Summary statistics for sediment fluxes across all catchments, including:
- **Fringe layer fluxes**: Min, max, median, and IQR values
- **Dispersed layer fluxes**: Min, max, median, and IQR values  
- **Total sediment fluxes**: Combined fringe and dispersed fluxes
- **Catchment metadata**: Region and area information

### `comprehensive_catchment_export.csv`
Detailed catchment-by-catchment results including:
- **Ice discharge**: Annual ice discharge in kg/yr
- **Ice yield**: Ice discharge per unit area (kg m⁻² yr⁻¹)
- **Sediment yield**: Sediment flux per unit area (kg m⁻² yr⁻¹)
- **Total flux**: Total sediment flux in kg/yr
- **Gate information**: Discharge gate numbers and coordinates

These files contain the primary results used in the manuscript analysis and can be used to reproduce the figures and analysis.

## Citation

If you use this code in your research, please cite the associated manuscript:

*Citation pending, manuscript currently under peer review.*

## License

This project is licensed under the GPL v3 License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for bugs or feature requests.