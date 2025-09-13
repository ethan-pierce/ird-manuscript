"""Plot sediment and ice fluxes from model output."""

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import linregress
from scipy.stats import t

dfs = []
for file in os.listdir('ird_model/models/checkpoints/fluxes'):
    with open(f'ird_model/models/checkpoints/fluxes/{file}', 'rb') as f:
        data = pickle.load(f)
    dfs.append(data)

df = pd.concat(dfs, ignore_index = True)

ice_yield = (df['ice_discharge'] * 1e12 / df['area']).to_numpy()
sed_yield = ((df['fringe_flux'] + df['dispersed_flux']) / df['area']).to_numpy()

print(df['contributing_area'].sum() / df['area'].sum())
quit()

# sns.set_theme(style = 'darkgrid', font_scale = 1.5)
# fig, ax = plt.subplots(figsize = (12, 8))
# plt.scatter(df['max_sliding_velocity'] * 31556926, sed_yield)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Max Sliding Velocity (m/yr)')
# plt.ylabel('Sediment Yield (kg m$^{-2}$ yr$^{-1}$)')
# plt.show()
# quit()

# iy_fit = ice_yield[np.log10(sed_yield) > -2]
# sy_fit = sed_yield[np.log10(sed_yield) > -2]
fit = linregress(np.log10(ice_yield), np.log10(sed_yield))
predicted_log_sed_yield = fit.slope * np.log10(ice_yield) + fit.intercept
predicted_sed_yield = 10**(predicted_log_sed_yield)
rmse = np.sqrt(np.mean((sed_yield - predicted_sed_yield)**2))
print('RMSE: ', rmse * df['area'].mean() * 1e-9)
R2 = fit.rvalue**2
print('R^2: ', R2)
p = fit.pvalue
print('p-value: ', p)

dof = df.shape[0]
t_crit = t.ppf(0.8, dof - 2)
slope_lower = fit.slope - t_crit * fit.stderr
slope_upper = fit.slope + t_crit * fit.stderr
print('Slope: ', fit.slope)

fig, ax = plt.subplots(figsize = (25, 10))
plt.scatter(np.log10(ice_yield), np.log10(sed_yield), s = 100, alpha = 0.5)
plt.plot(np.log10(ice_yield), predicted_log_sed_yield, color = 'black')
for i, glacier in enumerate(df['glacier']):
    plt.annotate(glacier, (np.log10(ice_yield[i]), np.log10(sed_yield[i])), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.show()

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

low_end = 10**(slope_lower * np.log10(GrISdf['Ice Yield']) + fit.intercept) * GrISdf['Area']
high_end = 10**(slope_upper * np.log10(GrISdf['Ice Yield']) + fit.intercept) * GrISdf['Area']
low = low_end.sum() * 1e-9
high = high_end.sum() * 1e-9
mean = GrISdf['Sediment Flux'].sum() * 1e-9
print(mean, ' plus ', high - mean, ' minus ', mean - low)
