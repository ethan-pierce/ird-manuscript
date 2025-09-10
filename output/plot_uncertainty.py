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

# Labels removed for cleaner visualization

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

# Set y-axis limits for clean visualization without labels
y_min = df['log_sed_yield'].min()
y_max = df['log_sed_yield'].max()
ax.set_ylim(bottom=y_min - 0.1,  # Small buffer
            top=y_max + 0.1)  # Small buffer

# Adjust layout with extra padding for labels
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)

# Save with high quality
plt.savefig('figures/ice-vs-sediment-yield-with-uq.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()