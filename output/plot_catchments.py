"""Plot fringe or dispersed thickness for every catchment in different regions."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections
import matplotlib.patches
from pathlib import Path
import seaborn as sns
from matplotlib.colors import LogNorm
import cmcrameri.cm as cmc
import argparse
import geopandas as gpd

# Region mapping from the codebase
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

# Region names for plotting
region_names = {
    'SW': 'Nuup Kangerlua',
    'CW': 'Ikerasak', 
    'CE': 'Kangertittivaq'
}

def plot_field(grid, array, ax, norm=None, set_clim=None, cmap='viridis'):
    """Plot a field on a triangle mesh grid."""
    values = grid.map_mean_of_patch_nodes_to_patch(array)
    
    coords = []
    for patch in range(grid.number_of_patches):
        nodes = []
        for node in grid.nodes_at_patch[patch]:
            nodes.append([grid.node_x[node], grid.node_y[node]])
        coords.append(nodes)
    
    import shapely
    hulls = [shapely.get_coordinates(shapely.Polygon(i).convex_hull) for i in coords]
    polys = [plt.Polygon(shp) for shp in hulls]
    
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=np.min(array), vmax=np.max(array))
    
    collection = matplotlib.collections.PatchCollection(polys, cmap=cmap, norm=norm, edgecolors='none')
    collection.set_array(values)
    
    if set_clim is not None:
        collection.set_clim(**set_clim)
    
    im = ax.add_collection(collection)
    ax.autoscale()
    
    return im

def load_sediment_results(catchment_name):
    """Load sediment model results for a catchment."""
    checkpoint_path = Path(f"ird_model/models/checkpoints/sediment/{catchment_name}.pickle")
    
    if not checkpoint_path.exists():
        print(f"Warning: No checkpoint found for {catchment_name}")
        return None
    
    with open(checkpoint_path, 'rb') as f:
        grid = pickle.load(f)
    
    return grid

def plot_region_thickness(region_code, thickness_type='fringe'):
    """Plot thickness (fringe or dispersed) for all catchments in a region."""
    # Get catchments for this region
    region_catchments = [name for name, reg in regions.items() if reg == region_code]
    
    if not region_catchments:
        print(f"No catchments found for region {region_code}")
        return
    
    print(f"Plotting {len(region_catchments)} catchments for region {region_code} - {thickness_type} thickness")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each catchment
    for catchment in region_catchments:
        grid = load_sediment_results(catchment)
        if grid is None:
            continue
            
        # Get thickness data based on type
        if thickness_type == 'fringe':
            thickness_data = grid.at_node['fringe_thickness']
            field_name = 'fringe_thickness'
        elif thickness_type == 'dispersed':
            thickness_data = grid.at_node['dispersed_thickness']
            field_name = 'dispersed_thickness'
        else:
            raise ValueError(f"Unknown thickness type: {thickness_type}")
        
        # Apply percentile cutoff to avoid extreme outliers
        thickness_pct99 = np.percentile(thickness_data, 99)
        thickness_clipped = np.where(thickness_data > thickness_pct99, thickness_pct99, thickness_data)
        
        # Choose colormap based on thickness type
        # Using complementary colormaps that work well together in figures
        if thickness_type == 'fringe':
            cmap = cmc.batlow  # Cool blue-green
        elif thickness_type == 'dispersed':
            cmap = cmc.vik  # Warm yellow-orange-red
        else:
            cmap = cmc.batlow
        
        # Plot the field
        im = plot_field(
            grid, 
            thickness_clipped, 
            ax, 
            cmap=cmap, 
            norm=LogNorm(vmin=1e-3, vmax=5)
        )
        
        # Add catchment boundary outline
        try:
            shapefile_path = f"data/shapefiles/{catchment}.geojson"
            if Path(shapefile_path).exists():
                gdf = gpd.read_file(shapefile_path)
                gdf.boundary.plot(ax=ax, color='white', linewidth=0.5)
        except Exception as e:
            print(f"Warning: Could not load boundary for {catchment}: {e}")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{thickness_type.title()} thickness (m)', fontsize=14)
    
    # Set labels and title
    ax.set_xlabel('Grid x (m)', fontsize=12)
    ax.set_ylabel('Grid y (m)', fontsize=12)
    ax.set_title(f'{region_names[region_code]} - {thickness_type.title()} thickness', fontsize=16)
    
    # Add scale bar
    if region_code == 'CE':
        ax.plot([3e5, 4e5], [-2.15e6, -2.15e6], color='black', linewidth=2)
        ax.text(3.5e5, -2.14e6, '100 km', fontsize=12, ha='center', va='center', color='black')
    elif region_code == 'CW':
        ax.plot([1e5, 2e5], [-2.18e6, -2.18e6], color='black', linewidth=2)
        ax.text(1.5e5, -2.17e6, '100 km', fontsize=12, ha='center', va='center', color='black')
    elif region_code == 'SW':
        ax.plot([-2.5e5, -1.5e5], [-2.9e6, -2.9e6], color='black', linewidth=2)
        ax.text(-2e5, -2.89e6, '100 km', fontsize=12, ha='center', va='center', color='black')
    
    # Save the plot
    output_path = f"figures/{thickness_type}/{region_names[region_code]}-{thickness_type}-thickness.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved plot to {output_path}")

def main():
    """Main function to plot thickness for all regions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot fringe or dispersed thickness for catchments in different regions')
    parser.add_argument('--type', choices=['fringe', 'dispersed'], default='fringe',
                        help='Type of thickness to plot (default: fringe)')
    parser.add_argument('--regions', nargs='+', choices=['SW', 'CW', 'CE'], default=['SW', 'CW', 'CE'],
                        help='Regions to plot (default: all regions)')
    
    args = parser.parse_args()
    
    # Set up plotting style
    sns.set_theme(style="white")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette('flare'))
    
    # Plot for each specified region
    for region_code in args.regions:
        plot_region_thickness(region_code, args.type)

if __name__ == "__main__":
    main()