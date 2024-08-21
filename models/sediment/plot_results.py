import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import linregress

sns.set_theme(style="whitegrid")

df = pd.read_csv('models/sediment/outputs/fluxes.csv')
df['ice_flux'] = df['ice_flux'] * 1e12 * 917
df['model_ice_flux'] = df['model_ice_flux'] * 917
df['ice_yield'] = df['ice_flux'] / df['catchment_area']
df['model_ice_yield'] = df['model_ice_flux'] / df['catchment_area']

# If Cf = 0.65 and Cd = 0.05
df['fringe_load'] = df['fringe_flux'] * 0.65 * 2700
df['dispersed_load'] = df['dispersed_flux'] * 0.1 * 2700
df['total_load'] = df['fringe_load'] + df['dispersed_load']
df['sediment_yield'] = df['total_load'] / df['catchment_area']

df['source_area_ratio'] = df['source_area'] / df['catchment_area']
df['transport_ratio'] = df['total_load'] / df['erosion_rate']

df['specific_erosion'] = df['erosion_rate'] / 2700 / df['catchment_area'] * 1e3


##########################
# Ice vs. Sediment yield #
##########################
df['log_ice_yield'] = np.log10(df['model_ice_yield'])
df['log_sed_yield'] = np.log10(df['sediment_yield'])
df = df[df['log_ice_yield'] > 1]
df = df[df['log_sed_yield'] > -1]
slope, intercept, r_value, p_value, std_err = linregress(df['log_ice_yield'], df['log_sed_yield'])
print(slope, intercept)
print('R = ', r_value)
print('R^2 = ', r_value**2)
fig = sns.regplot(data = df, x = 'log_ice_yield', y = 'log_sed_yield')
plt.show()
quit()

########################
# Transport efficiency #
########################
# ratios, names = zip(*sorted(zip(df['transport_ratio'], df['glacier'])))
# fig, ax = plt.subplots()
# ax.plot(range(len(names)), ratios)
# ax.set_xticks(range(len(names)))
# ax.set_xticklabels([i.replace('-', ' ').title() for i in names], rotation = 90)
# ax.set_ylabel('Sediment transport / Erosion')
# plt.tight_layout()
# plt.show()

#####################
# Linear regression #
#####################
target_col = df['total_load']
target = (target_col - target_col.min()) / (target_col.max() - target_col.min())

data = df[[
    'model_ice_flux', 'ice_flux', 'model_ice_yield', 'ice_yield', 'source_area', 'catchment_area',
    'terminus_width', 'terminus_height', 'mean_pressure', 'mean_terminus_pressure', 
    'mean_velocity', 'max_velocity', 'mean_terminus_velocity', 'max_terminus_velocity', 
    'erosion_rate', 'total_basal_melt', 'max_basal_melt', 'transport_ratio'
]]

predictors = pd.DataFrame()
for idx, col in data.items():
    normalized = (col - col.min()) / (col.max() - col.min())
    predictors[idx] = normalized

regression = LinearRegression().fit(predictors, target)

print('R2 = ', regression.score(predictors, target))

predict = regression.predict(predictors)
plt.scatter(target, predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

coeffs = pd.DataFrame()
coeffs['Variable'] = data.columns
coeffs['Coefficient'] = regression.coef_
coeffs['Absolute Value'] = np.abs(regression.coef_)

print(coeffs.sort_values(by='Coefficient', ascending=False))

correlation_matrix = data.corr()
plt.figure(figsize=(12, 12))
plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
plt.show()