import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df = pd.read_csv('models/sediment/outputs/fluxes.csv')
df['ice_flux'] = df['ice_flux'] * 1e12
df['simple_ice_flux'] = df['terminus_width'] * df['terminus_height'] * df['mean_terminus_velocity'] * 917

# If Cf = 0.65 and Cd = 0.05
df['fringe_load'] = df['fringe_flux'] * 0.65 * 2700
df['dispersed_load'] = df['dispersed_flux'] * 0.05 * 2700
df['total_load'] = df['fringe_load'] + df['dispersed_load']

print(df)


simple_fringe = df['fringe_height_terminus'] * df['terminus_width'] * df['mean_terminus_velocity'] * 0.65 * 2700
simple_dispersed = df['dispersed_height_terminus'] * df['terminus_width'] * df['mean_terminus_velocity'] * 0.05 * 2700
simple_total = simple_fringe + simple_dispersed

plt.scatter(df['source_area'] / df['catchment_area'], simple_total)
plt.show()