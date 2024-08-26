import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress

sns.set_theme(style="whitegrid")

#########################
# Continent-wide fluxes #
#########################
sns.set(font_scale = 1.25)
Q = pd.DataFrame(columns = ['Source', 'Flux', 'Uncertainty'])
Q['Source'] = ['Fjord accumulation', 'Suspended sediment', 'Ice-rafted debris']
Q['Flux'] = [1.324, 0.892, 0.416]
Q['Uncertainty'] = [0.79, 0.374, 0.254] # TODO check UQ

fig, ax = plt.subplots(figsize = (6, 8))
sns.barplot(data = Q, x = 'Source', y = 'Flux', errorbar = 'ci', palette = 'viridis', ax = ax, width = 0.5)

for i in range(Q.shape[0]):
    plt.scatter(i, Q['Flux'].iloc[i], color = 'black')
    plt.plot([i, i], [Q['Flux'].iloc[i] - Q['Uncertainty'].iloc[i], Q['Flux'].iloc[i] + Q['Uncertainty'].iloc[i]], color = 'black')

plt.xlabel('')
ax.set_xticklabels(['Fjords', 'Plumes', 'Icebergs'])
plt.ylabel('Flux (Gt a$^{-1}$)')
plt.tight_layout()
plt.savefig('figures/integrated-fluxes-barplot.png', dpi = 300)
plt.show()
quit()




df = pd.read_csv('models/sediment/outputs/fluxes.csv')
df['ice_flux'] = df['ice_flux'] * 1e12 * 917
df['model_ice_flux'] = df['model_ice_flux'] * 917
df['ice_yield'] = df['ice_flux'] / df['catchment_area']
df['model_ice_yield'] = df['model_ice_flux'] / df['catchment_area']

# If Cf = 0.65 and Cd = 0.05
df['fringe_load'] = df['fringe_flux'] * 0.65 * 2700
df['dispersed_load'] = df['dispersed_flux'] * 0.05 * 2700
df['total_load'] = df['fringe_load'] + df['dispersed_load']

df['sediment_yield'] = df['total_load'] / df['catchment_area']

df['source_area_ratio'] = df['source_area'] / df['catchment_area']
df['transport_ratio'] = df['total_load'] / df['erosion_rate']

df['specific_erosion'] = df['erosion_rate'] / 2700 / df['catchment_area'] * 1e3
df['specific_sed_yield'] = df['total_load'] / 2700 / df['catchment_area'] * 1e3

test_fit = linregress(np.log10(df['ice_yield']), np.log10(df['sediment_yield']))
flux_correction = linregress(df['ice_yield'], df['model_ice_yield'])

##########################
# Ice vs. Sediment yield #
##########################
df['log_ice_yield'] = np.log10(df['model_ice_yield'])
df['log_sed_yield'] = np.log10(df['sediment_yield'])

first_fit = linregress(df['log_ice_yield'], df['log_sed_yield'])

df2 = df[df['log_ice_yield'] > 1]
second_fit = linregress(df2['log_ice_yield'], df2['log_sed_yield'])

df3 = df[(df['log_sed_yield'] > -1) & (df['log_ice_yield'] > 1)]
third_fit = linregress(df3['log_ice_yield'], df3['log_sed_yield'])
fit = third_fit
print(fit)

#############
# Upscaling #
#############
discharge = pd.read_csv('models/inputs/gate_D.csv', header = 0)
# discharge = {key: discharge_data[str(val)].iloc[308:2314].mean() for key, val in discharge_gate.items()}

areas = gpd.read_file('models/inputs/catchment-area-at-gate.geojson')
areas['Area'] = areas['rast_val']
areas['Gate'] = areas['pnt_val']
print(areas.head)

GrISdf = pd.DataFrame(columns = ['Gate', 'Area', 'Ice Discharge', 'Ice Yield', 'Sediment Yield', 'Sediment Flux'])
GrISdf['Gate'] = areas['Gate']
GrISdf['Area'] = areas['Area']
GrISdf['x'] = areas['geometry'].x
GrISdf['y'] = areas['geometry'].y
GrISdf['Ice Discharge'] = [discharge[str(int(i))].iloc[308:2314].mean() for i in GrISdf['Gate']]
GrISdf['Ice Discharge'] = GrISdf['Ice Discharge'] * 1e12 * 917
GrISdf['Corrected Discharge'] = flux_correction.slope * GrISdf['Ice Discharge'] + flux_correction.intercept
GrISdf['Ice Yield'] = GrISdf['Corrected Discharge'] / GrISdf['Area']
GrISdf['log(Sed. Yield)'] = fit.slope * np.log10(GrISdf['Ice Yield']) + fit.intercept
GrISdf['Sediment Yield'] = 10**(GrISdf['log(Sed. Yield)'])
GrISdf['Sediment Flux'] = GrISdf['Sediment Yield'] * GrISdf['Area']

print('Total discharge: ', np.round(GrISdf['Sediment Flux'].sum() * 1e-9, 2), 'Mt / yr')

GrISdf['Ice-rafted sediment flux'] = GrISdf['Sediment Flux'] * 1e-9
GrISdf.to_csv('models/sediment/outputs/GrIS-fluxes.csv')


import statsmodels.api as sm
lr = sm.OLS(df3['log_sed_yield'], sm.add_constant(df3['log_ice_yield'])).fit()
ci = lr.conf_int(alpha = 0.6)
print(lr.summary())
print(ci)

lower = np.sum(10**(ci[0][1] * np.log10(GrISdf['Ice Yield']) + ci[0][0]) * GrISdf['Area']) * 1e-9
upper = np.sum(10**(ci[1][1] * np.log10(GrISdf['Ice Yield']) + ci[1][0]) * GrISdf['Area']) * 1e-9
mean = np.sum(10**(lr.params['log_ice_yield'] * np.log10(GrISdf['Ice Yield']) + lr.params['const']) * GrISdf['Area']) * 1e-9

print(lower, mean, upper)


quit()

# df['Group'] = 0
# for idx, row in df.iterrows():
#     if row['log_ice_yield'] < 1:
#         df.at[idx, 'Group'] = 2
#     elif row['log_sed_yield'] < -1:
#         df.at[idx, 'Group'] = 1
#     else:
#         df.at[idx, 'Group'] = 0

# df['Region'] = df['region'].replace({
#     'SW': 'Nuup Kangerlua',
#     'CW': 'Ikerasak',
#     'CE': 'Kangertittivaq'
# })

# fig, ax = plt.subplots(figsize = (12, 6))

# sns.scatterplot(
#     ax = ax, data = df[df['log_ice_yield'] > 1], x = 'log_ice_yield', y = 'log_sed_yield', 
#     style = 'Group', hue = 'Region', s = 100,
#     palette = sns.color_palette(['blue', 'magenta', 'red'])
# )
# sns.regplot(
#     ax = ax, data = df3, x = 'log_ice_yield', y = 'log_sed_yield',
#     color = 'black', line_kws = {'linestyle': ':', 'linewidth': 2}, scatter = False
# )

# label = 'log(sediment yield) = {:.2f}log(ice yield) + {:.2f}'.format(third_fit.slope, third_fit.intercept)
# ax.text(1.5, 0.75, label, fontsize = 12)
# ax.text(1.5, 0.5, 'R$^2$ = {:.2f}'.format(third_fit.rvalue ** 2), fontsize = 12)

# h,l = ax.get_legend_handles_labels()
# l[4] = 'Category'
# l[5] = 'Fringe reaches terminus'
# l[6] = 'Upstream deposition only'
# ax.legend(h[:7], l[:7])

# ax.set_xlabel('Log$_{10}$(Ice yield) (kg m$^{-2}$ a$^{-1}$)')
# ax.set_ylabel('Log$_{10}$(Sediment yield) (kg m$^{-2}$ a$^{-1}$)')

# plt.tight_layout()
# plt.savefig('figures/ice-vs-sed-yield.png', dpi = 300)
# quit()

#########################
# Total sediment fluxes #
#########################
# sns.set(font_scale = 1.25)

# df['log_fringe_flux'] = np.log10(df['fringe_flux'])
# df['log_dispersed_flux'] = np.log10(df['dispersed_flux'])
# df['log_total_flux'] = np.log10(df['fringe_flux'] + df['dispersed_flux'])
# df['total_flux'] = df['fringe_flux'] + df['dispersed_flux']
# df = df.sort_values('total_flux', ascending = False)

# df['Glacier'] = [i.replace('-', ' ').title() for i in df['glacier']]

# fig, ax = plt.subplots(figsize = (10, 14))

# sns.set_color_codes('pastel')
# sns.barplot(
#     ax = ax, data = df, x = 'total_flux', y = 'Glacier', 
#     label = 'Dispersed load', color = 'lightcoral', width = 0.8
# )
# sns.set_color_codes('muted')
# sns.barplot(
#     ax = ax, data = df, x = 'fringe_flux', y = 'Glacier', 
#     label = 'Fringe load', color = 'darkred', width = 0.8
# )

# sns.despine(left = True, bottom = True)
# ax.set_xlabel('Sediment flux (m$^3$ a$^{-1}$)')
# ax.set_ylabel('')
# ax.legend(ncol = 2, loc = 'lower right')

# plt.tight_layout()
# plt.savefig('figures/total-sediment-fluxes.png', dpi = 300)

# fig, ax = plt.subplots(figsize = (10, 14))
# sns.set_color_codes('pastel')
# sns.barplot(
#     ax = ax, data = df, x = 'specific_erosion', y = 'Glacier', 
#     label = 'Erosion', color = 'lightblue'
# )
# sns.set_color_codes('muted')
# sns.barplot(
#     ax = ax, data = df, x = 'specific_sed_yield', y = 'Glacier', 
#     label = 'Transport', color = 'darkblue'
# )
# sns.despine(left = True, bottom = True)
# ax.set_xlabel('Catchment-averaged rate (mm a$^{-1}$)')
# ax.set_ylabel('')
# ax.legend(ncol = 2, loc = 'lower right')
# plt.tight_layout()
# plt.savefig('figures/specific-yield.png', dpi = 300)

#####################
# Linear regression #
#####################
# target_col = df['total_load']
# target = (target_col - target_col.min()) / (target_col.max() - target_col.min())

# data = df[[
#     'model_ice_flux', 'ice_flux', 'model_ice_yield', 'ice_yield', 'source_area', 'catchment_area',
#     'terminus_width', 'terminus_height', 'mean_pressure', 'mean_terminus_pressure', 
#     'mean_velocity', 'max_velocity', 'mean_terminus_velocity', 'max_terminus_velocity', 
#     'erosion_rate', 'total_basal_melt', 'max_basal_melt', 'transport_ratio'
# ]]

# predictors = pd.DataFrame()
# for idx, col in data.items():
#     normalized = (col - col.min()) / (col.max() - col.min())
#     predictors[idx] = normalized

# regression = LinearRegression().fit(predictors, target)

# print('R2 = ', regression.score(predictors, target))

# predict = regression.predict(predictors)
# plt.scatter(target, predict)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.show()

# coeffs = pd.DataFrame()
# coeffs['Variable'] = data.columns
# coeffs['Coefficient'] = regression.coef_
# coeffs['Absolute Value'] = np.abs(regression.coef_)

# print(coeffs.sort_values(by='Coefficient', ascending=False))

# correlation_matrix = data.corr()
# plt.figure(figsize=(12, 12))
# plt.imshow(correlation_matrix, cmap='coolwarm')
# plt.colorbar()
# plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
# plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
# plt.tight_layout()
# plt.show()