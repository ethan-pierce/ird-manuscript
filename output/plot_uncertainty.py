"""Plot uncertainty quantification results."""

import os 
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import linregress, t
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress all warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})


dfs = []
for file in os.listdir('ird_model/models/checkpoints/uncertainty/base'):
    if file.endswith('.csv'):
        dfs.append(pd.read_csv(f'ird_model/models/checkpoints/uncertainty/base/{file}'))

df = pd.concat(dfs, ignore_index = True)
df['ice_yield'] = df['ice_discharge'] * 1e12 / df['area']
df['sed_yield'] = (df['fringe_flux'] + df['dispersed_flux']) / df['area']
df['dispersed_yield'] = df['dispersed_flux'] / df['area']
df['fringe_yield'] = df['fringe_flux'] / df['area']


fit = linregress(np.log10(df['ice_yield']), np.log10(df['sed_yield']))
predicted_log_sed_yield = fit.slope * np.log10(df['ice_yield']) + fit.intercept
predicted_sed_yield = 10**(predicted_log_sed_yield)
rmse = np.sqrt(np.mean((df['sed_yield'] - predicted_sed_yield)**2))
p = fit.pvalue
print('p-value: ', p)

groups = df.groupby('glacier')
r2fit = linregress(np.log10(groups['ice_yield'].median()), np.log10(groups['sed_yield'].median()))
r2_predicted_log_sed_yield = r2fit.slope * np.log10(df['ice_yield']) + r2fit.intercept
r2_predicted_sed_yield = 10**(r2_predicted_log_sed_yield)
R2 = r2fit.rvalue**2
print('R^2: ', R2)
rmse = np.sqrt(np.mean((df['sed_yield'] - r2_predicted_sed_yield)**2))
print('RMSE: ', rmse)

gate_discharge = pd.read_csv('data/gate_D.csv', header = 0)
area_at_gates = gpd.read_file('data/catchment-area-at-gate.geojson')
area_at_gates['Area'] = area_at_gates['rast_val']
area_at_gates['Gate'] = area_at_gates['pnt_val']
GrISdf = pd.DataFrame(columns = ['Gate', 'Area', 'Ice Discharge', 'Ice Yield', 'Sediment Yield', 'Sediment Flux'])
GrISdf['Gate'] = area_at_gates['Gate']
GrISdf['Area'] = area_at_gates['Area']
GrISdf['Ice Discharge'] = [gate_discharge[str(int(i))].iloc[1744:2853].median() for i in GrISdf['Gate']]
GrISdf['Ice Discharge'] = GrISdf['Ice Discharge'] * 1e12
GrISdf['Ice Yield'] = GrISdf['Ice Discharge'] / GrISdf['Area']
GrISdf['log(Sed. Yield)'] = fit.slope * np.log10(GrISdf['Ice Yield']) + fit.intercept
GrISdf['Sediment Yield'] = 10**(GrISdf['log(Sed. Yield)'])
GrISdf['Sediment Flux'] = GrISdf['Sediment Yield'] * GrISdf['Area']
print('RMSE (Mt/yr): ', GrISdf['Area'].median() * 1e-9 * rmse)

critical_value = t.ppf(0.95, df.shape[0] - 2)
slope_lower = fit.slope - critical_value * fit.stderr
slope_upper = fit.slope + critical_value * fit.stderr

low_end = 10**(slope_lower * np.log10(GrISdf['Ice Yield']) + fit.intercept) * GrISdf['Area']
high_end = 10**(slope_upper * np.log10(GrISdf['Ice Yield']) + fit.intercept) * GrISdf['Area']
low = low_end.sum() * 1e-9
high = high_end.sum() * 1e-9
mean = GrISdf['Sediment Flux'].sum() * 1e-9
print(mean, ' plus ', high - mean, ' minus ', mean - low)

df['log_ice_yield'] = np.log10(df['ice_yield'])
df['log_sed_yield'] = np.log10(df['sed_yield'])
glaciers = df.groupby('glacier')

# Create professional violin plot
fig, ax = plt.subplots(figsize=(14, 8))

# Create violin plot with professional styling and better spacing
violin_plot = sns.violinplot(
    data=df, x='log_ice_yield', y='log_sed_yield', 
    native_scale=True, inner='box', cut=0.01,
    density_norm='width', ax=ax,
    palette='viridis', alpha=0.7,
    width=0.8,  # Slightly wider violins for better visibility
    linewidth=1.5  # Thicker violin outlines for clarity
)

# Add unobtrusive glacier labels positioned close to each violin
# Get unique glaciers and their positions
glacier_groups = df.groupby('glacier')
glacier_names = list(glacier_groups.groups.keys())
glacier_positions = []

# Calculate median x-position for each glacier group
for glacier in glacier_names:
    glacier_data = glacier_groups.get_group(glacier)
    median_x = glacier_data['log_ice_yield'].median()
    glacier_positions.append(median_x)

# Sort glaciers by their x-position for consistent labeling
sorted_indices = np.argsort(glacier_positions)
sorted_glaciers = [glacier_names[i] for i in sorted_indices]
sorted_positions = [glacier_positions[i] for i in sorted_indices]

# Add labels with improved collision detection
# First, collect all violin tops and positions
violin_data = []
for glacier, pos in zip(sorted_glaciers, sorted_positions):
    glacier_data = glacier_groups.get_group(glacier)
    violin_top = glacier_data['log_sed_yield'].max()
    violin_data.append((glacier, pos, violin_top))

# Sort by x-position for left-to-right processing
violin_data.sort(key=lambda x: x[1])

# Labels removed - clean violin plot without glacier labels

# Add best fit line with confidence intervals
x_range = np.linspace(df['log_ice_yield'].min(), df['log_ice_yield'].max(), 100)
y_fit = fit.slope * x_range + fit.intercept

# Calculate confidence intervals
critical_value = t.ppf(0.95, df.shape[0] - 2)
se_fit = np.sqrt(fit.stderr**2 + (fit.stderr * (x_range - df['log_ice_yield'].mean()))**2)
y_lower = y_fit - critical_value * se_fit
y_upper = y_fit + critical_value * se_fit

# Plot confidence band
ax.fill_between(x_range, y_lower, y_upper, alpha=0.2, color='red', label='95% Confidence Interval')

# Plot best fit line
ax.plot(x_range, y_fit, 'r-', linewidth=3, label=f'Best Fit: SY = {10**fit.intercept:.4f} IY^{fit.slope:.2f}')

# Add statistics text box
stats_text = f'RMSE = {rmse:.3f} kg m⁻² a⁻¹\nR² = {R2:.3f}\np < 0.001'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10, fontweight='bold')

# Professional axis labels with units
ax.set_xlabel('log₁₀(Ice Yield) [kg m$^{-2}$ yr$^{-1}$]', fontsize=12, fontweight='bold')
ax.set_ylabel('log₁₀(Sediment Yield) [kg m$^{-2}$ yr$^{-1}$]', fontsize=12, fontweight='bold')

# Improve legend - position it to avoid conflict with statistics box
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

# Improve grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Set axis limits for clean violin plot without labels
y_min = df['log_sed_yield'].min()
y_max = df['log_sed_yield'].max()
x_min = df['log_ice_yield'].min()
x_max = df['log_ice_yield'].max()

# Normal plot limits without extra space for labels
ax.set_ylim(bottom=y_min - 0.1,  # Small buffer at bottom
            top=y_max + 0.1)  # Small buffer at top
ax.set_xlim(left=x_min - 0.1,  # Small buffer at left
            right=x_max + 0.1)  # Small buffer at right

# Adjust layout for clean violin plot without labels
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)  # Normal padding without labels

# Save with high quality
plt.savefig('figures/ice-vs-sediment-yield-with-uq.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()






# Create comprehensive catchment statistics table
print("\n" + "="*80)
print("CATCHMENT SEDIMENT FLUX STATISTICS")
print("="*80)

# Calculate total sediment flux for each row
df['total_flux'] = df['fringe_flux'] + df['dispersed_flux']

# Group by glacier and calculate statistics from uncertainty runs
catchment_stats = df.groupby(['glacier', 'region', 'area']).agg({
    'fringe_flux': ['min', 'max', 'median', lambda x: x.quantile(0.75) - x.quantile(0.25)],  # IQR
    'dispersed_flux': ['min', 'max', 'median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
    'total_flux': ['min', 'max', 'median', lambda x: x.quantile(0.75) - x.quantile(0.25)]
}).round(0)

# Flatten column names
catchment_stats.columns = ['_'.join(col).strip() for col in catchment_stats.columns.values]
catchment_stats = catchment_stats.reset_index()

# Rename columns for clarity
catchment_stats.columns = [
    'Catchment', 'Region', 'Area (m²)', 
    'Fringe Min', 'Fringe Max', 'Fringe Median', 'Fringe IQR',
    'Dispersed Min', 'Dispersed Max', 'Dispersed Median', 'Dispersed IQR',
    'Total Min', 'Total Max', 'Total Median', 'Total IQR'
]

# Reorder columns
cols = ['Catchment', 'Region', 'Area (m²)', 
        'Fringe Min', 'Fringe Max', 'Fringe Median', 'Fringe IQR',
        'Dispersed Min', 'Dispersed Max', 'Dispersed Median', 'Dispersed IQR',
        'Total Min', 'Total Max', 'Total Median', 'Total IQR']
catchment_stats = catchment_stats[cols]

# Sort by region, then by total median flux
catchment_stats = catchment_stats.sort_values(['Region', 'Total Median'], ascending=[True, False])

print("\nCATCHMENT-LEVEL STATISTICS:")
print("-" * 80)
# Select only median columns for display
catchment_display = catchment_stats[['Catchment', 'Region', 'Area (m²)', 
                                   'Fringe Median', 'Dispersed Median', 'Total Median']]

print(catchment_display.to_string(index=False, formatters={
    'Fringe Median': '{:,.0f}'.format,
    'Dispersed Median': '{:,.0f}'.format,
    'Total Median': '{:,.0f}'.format,
    'Area (m²)': '{:,.1f}'.format
}))

# Calculate regional totals by summing catchment values
print("\n" + "="*80)
print("REGIONAL TOTALS")
print("="*80)

regional_totals = catchment_stats.groupby('Region').agg({
    'Area (m²)': 'sum',
    'Fringe Min': 'sum',
    'Fringe Max': 'sum', 
    'Fringe Median': 'sum',
    'Fringe IQR': 'sum',
    'Dispersed Min': 'sum',
    'Dispersed Max': 'sum',
    'Dispersed Median': 'sum',
    'Dispersed IQR': 'sum',
    'Total Min': 'sum',
    'Total Max': 'sum',
    'Total Median': 'sum',
    'Total IQR': 'sum'
}).round(0).reset_index()

# Sort by total median flux
regional_totals = regional_totals.sort_values('Total Median', ascending=False)

# Select only median columns for display
regional_display = regional_totals[['Region', 'Area (m²)', 
                                   'Fringe Median', 'Dispersed Median', 'Total Median']]

print("\nREGIONAL TOTALS:")
print("-" * 80)
print(regional_display.to_string(index=False, formatters={
    'Fringe Median': '{:,.0f}'.format,
    'Dispersed Median': '{:,.0f}'.format,
    'Total Median': '{:,.0f}'.format,
    'Area (m²)': '{:,.1f}'.format
}))

# Save only the catchment statistics table
catchment_stats.to_csv('output/catchment_sediment_flux_statistics.csv', index=False)

print(f"\nTable saved to: output/catchment_sediment_flux_statistics.csv")

# Create comprehensive catchment export with all requested information
print("\n" + "="*80)
print("CREATING COMPREHENSIVE CATCHMENT EXPORT")
print("="*80)

# Get gate coordinates from the GeoJSON file
gate_coords = {}
for idx, row in area_at_gates.iterrows():
    gate_coords[row['Gate']] = (row.geometry.x, row.geometry.y)

# Create comprehensive export dataframe using uniform methodology
comprehensive_export = []

# Use GrISdf as the base, with ice discharge from gate_D and sediment flux from linear fit
for idx, row in GrISdf.iterrows():
    gate_number = row['Gate']
    area = row['Area']
    gate_x, gate_y = gate_coords.get(gate_number, (None, None))
    
    # Get ice discharge from gate_D data
    ice_discharge = gate_discharge[str(int(gate_number))].iloc[1744:2853].median() * 1e12  # Convert to kg/yr
    
    # Calculate ice yield
    ice_yield = ice_discharge / area
    
    # Calculate sediment yield using the linear fit
    log_sediment_yield = fit.slope * np.log10(ice_yield) + fit.intercept
    sediment_yield = 10**(log_sediment_yield)
    
    # Calculate total sediment flux
    total_flux = sediment_yield * area
    
    comprehensive_export.append({
        'Catchment': f'Gate {gate_number}',
        'Area (m²)': area,
        'Ice Discharge (kg/yr)': ice_discharge,
        'Ice Yield (kg m⁻² yr⁻¹)': ice_yield,
        'Sediment Yield (kg m⁻² yr⁻¹)': sediment_yield,
        'Total Flux (kg/yr)': total_flux,
        'Gate Number': gate_number,
        'Gate Longitude': gate_x,
        'Gate Latitude': gate_y
    })

# Convert to DataFrame and save
comprehensive_df = pd.DataFrame(comprehensive_export)
comprehensive_df = comprehensive_df.sort_values('Total Flux (kg/yr)', ascending=False)

# Save comprehensive export
comprehensive_df.to_csv('output/comprehensive_catchment_export.csv', index=False)

# Define yield fit range
min_yield = np.min(df['ice_yield'])
max_yield = np.max(df['ice_yield'])

# Calculate percentages for catchments
catchments_inside = comprehensive_df[
    (comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] < max_yield) &
    (comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] > min_yield)
].shape[0]

catchments_below = comprehensive_df[
    comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] <= min_yield
].shape[0]

catchments_above = comprehensive_df[
    comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] >= max_yield
].shape[0]

total_catchments = comprehensive_df.shape[0]

# Calculate percentages for ice flux
ice_flux_inside = comprehensive_df[
    (comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] < max_yield) &
    (comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] > min_yield)
]['Ice Discharge (kg/yr)'].sum()

ice_flux_below = comprehensive_df[
    comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] <= min_yield
]['Ice Discharge (kg/yr)'].sum()

ice_flux_above = comprehensive_df[
    comprehensive_df['Ice Yield (kg m⁻² yr⁻¹)'] >= max_yield
]['Ice Discharge (kg/yr)'].sum()

total_ice_flux = comprehensive_df['Ice Discharge (kg/yr)'].sum()

# Print results
print(f'Yield fit range: {min_yield:.3f} to {max_yield:.3f} kg m⁻² yr⁻¹')
print(f'\nCatchments:')
print(f'  Below fit range: {catchments_below/total_catchments*100:.2f}% ({catchments_below}/{total_catchments})')
print(f'  Inside fit range: {catchments_inside/total_catchments*100:.2f}% ({catchments_inside}/{total_catchments})')
print(f'  Above fit range: {catchments_above/total_catchments*100:.2f}% ({catchments_above}/{total_catchments})')

print(f'\nIce flux:')
print(f'  Below fit range: {ice_flux_below/total_ice_flux*100:.2f}%')
print(f'  Inside fit range: {ice_flux_inside/total_ice_flux*100:.2f}%')
print(f'  Above fit range: {ice_flux_above/total_ice_flux*100:.2f}%')

print("\nCOMPREHENSIVE CATCHMENT EXPORT:")
print("-" * 100)
print(f"\nComprehensive export saved to: output/comprehensive_catchment_export.csv")



fluxes = {
    'IRD transport (this study)': [0.454, (high - mean) * 1e-3, (mean - low) * 1e-3],
    'Fluvial transport (Overeem et al., 2017)': [0.892, 0.374, 0.374],
    'Fjord deposition (Andresen et al., 2024)': [1.324, 0.79, 0.79]
}

# Create bar plot comparing sediment fluxes
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))

# Extract data for plotting
studies = list(fluxes.keys())
# Create multi-line labels for better fit
study_labels = [
    'IRD transport\n(this study)',
    'Fluvial transport\n(Overeem et al., 2017)',
    'Fjord deposition\n(Andresen et al., 2024)'
]
means = [fluxes[study][0] for study in studies]
errors_low = [fluxes[study][2] for study in studies]  # Lower error
errors_high = [fluxes[study][1] for study in studies]  # Upper error

# Create bar plot with error bars
bars = ax_bar.bar(range(len(studies)), means, 
                  yerr=[errors_low, errors_high], 
                  capsize=5, width=0.4,
                  color=['#8B4A6B', '#4A6B8B', '#6B8B4A'],  # More sophisticated colors
                  alpha=0.85, edgecolor='#2C2C2C', linewidth=1.2)

# Customize the plot
ax_bar.set_ylabel('Sediment Flux (Gt yr⁻¹)', fontsize=18, fontweight='bold')
ax_bar.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, (bar, mean_val) in enumerate(zip(bars, means)):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height + errors_high[i] + 0.05,
                f'{mean_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=16)

# Set x-axis labels with multi-line text
ax_bar.set_xticks(range(len(studies)))
ax_bar.set_xticklabels(study_labels, fontsize=16, ha='center')
plt.yticks(fontsize=16)
plt.tight_layout()

# Save the bar plot
plt.savefig('figures/fluxes-barplot.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()
