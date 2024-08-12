"""Plot rafted sediment concentrations from field data."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_sw = pd.read_csv('NuupKangerlua_ssc_results.csv')
df_sw = df_sw[df_sw['SSC_mass'] > 0]

df_cw = pd.read_csv('SAMPLEWEIGHTS_EQ22icebergsamples.csv', header = 5, names = ['Sample', 'GRL', 'Field weight', 'Lab weight', '% loss', 'Tape weight', 'Empty weight', 'Sed. weight', 'SSC_mass'])
df_cw = df_cw[(df_cw['SSC_mass'] > 0) & (df_cw['SSC_mass'] != 1.0)]
df_cw['SSC_mass'] = df_cw['SSC_mass'] * 100

sco18 = pd.read_csv('East_Greenland_2018_ssc.csv', nrows = 26)
sco18.loc['sd W%', 27] = 40
sco18['SSC_mass'] = sco18['sd W%'].astype('float64')
sco18 = sco18[['SSC_mass']]
# sco19 = pd.read_csv('East_Greenland_2019_ssc.csv', nrows = 76)
# sco19['SSC_mass'] = ((sco19['Sed. G'] / sco19['Sample g']) * 100).astype('float64')
# sco19 = sco19[['SSC_mass']]
# df_ce = pd.merge(sco18, sco19, how = 'outer')
df_ce = sco18

fig, ax1 = plt.subplots()
df = pd.merge(df_sw, df_cw, how = 'outer').merge(df_ce, how = 'outer')
sns.kdeplot(data = df, x = "SSC_mass", ax = ax1, bw_adjust = 0.75)
ax1.set_xlim((df["SSC_mass"].min() - 1, df["SSC_mass"].max() + 1))
ax2 = ax1.twinx()
sns.histplot(data = df, x = "SSC_mass", discrete = True, ax = ax2, bins = 10)
plt.show()