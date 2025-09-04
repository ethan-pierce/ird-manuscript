"""Plot sediment and ice fluxes from model output."""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import linregress

dfs = []
for file in os.listdir('ird_model/models/checkpoints/fluxes'):
    with open(f'ird_model/models/checkpoints/fluxes/{file}', 'rb') as f:
        data = pickle.load(f)
    dfs.append(data)

df = pd.concat(dfs)
ice_yield = (df['ice_discharge'] * 1e12 / df['area']).to_numpy()
sed_yield = ((df['fringe_flux'] + df['dispersed_flux']) / df['area']).to_numpy()

result = linregress(np.log10(ice_yield), np.log10(sed_yield))
print(result)
