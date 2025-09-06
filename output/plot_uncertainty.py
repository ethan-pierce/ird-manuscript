import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ird_model/models/checkpoints/uncertainty/eqip-sermia.csv')

ice_yield = df['ice_discharge'] * 1e12 / df['area']
sed_yield = (df['fringe_flux'] + df['dispersed_flux']) / df['area']

plt.boxplot(np.log10(sed_yield), bootstrap = 1000, positions = [np.log10(ice_yield[0])])
plt.show()

# plt.scatter(df['erosion_exponent'], np.log10(sed_yield))
# plt.show()