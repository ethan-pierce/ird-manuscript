"""Plot sediment and ice fluxes from model output."""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dfs = []
for file in os.listdir('ird_model/models/checkpoints/fluxes'):
    with open(f'ird_model/models/checkpoints/fluxes/{file}', 'rb') as f:
        data = pickle.load(f)
    dfs.append(data)

df = pd.concat(dfs)
ice_yield = (df['ice_discharge'] * 1e12 / df['area']).to_numpy()
sed_yield = ((df['fringe_flux'] + df['dispersed_flux']) / df['area']).to_numpy()

fig, ax = plt.subplots(figsize = (10, 6))
plt.scatter(np.log10(ice_yield), np.log10(sed_yield), s = 100, alpha = 0.5)
plt.xlabel('Ice yield (kg$^3$ a$^{-1}$)')
plt.ylabel('Sediment yield (m$^3$ a$^{-1}$)')
plt.show()
