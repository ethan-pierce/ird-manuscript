"""Plot rafted sediment concentrations from field data."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style = 'ticks')

cut = 0.1

df_sw = pd.read_csv('data/NuupKangerlua_ssc_results.csv', sep = ',')
# print(len(df_sw['Name'].unique()))
print(df_sw.shape[0])
df_sw = df_sw[df_sw['SSC_mass'] > cut]
df_sw['Region'] = 'Nuup Kangerlua'

df_cw = pd.read_csv(
    'data/SAMPLEWEIGHTS_EQ22icebergsamples.csv', 
    header = 5, 
    names = ['Sample', 'GRL', 'Field weight', 'Lab weight', '% loss', 'Tape weight', 'Empty weight', 'Sed. weight', 'SSC_mass'],
    sep = '\s+'
)
print(df_cw.shape[0])
df_cw = df_cw[(df_cw['SSC_mass'] > 0) & (df_cw['SSC_mass'] != 1.0)]
df_cw['SSC_mass'] = df_cw['SSC_mass'] * 100
df_cw = df_cw[df_cw['SSC_mass'] > cut]
df_cw['Region'] = 'Ikerasak'

sco18 = pd.read_csv('data/Kangertittivaq2018.csv')
sco18['SSC_mass'] = sco18['RSC'] * 100
sco18 = sco18[sco18['SSC_mass'] > cut][['SSC_mass']]
df_ce = sco18
df_ce['Region'] = 'Kangertittivaq'
print(df_ce.shape[0])

df = pd.concat([df_sw, df_cw, df_ce])

fig, ax = plt.subplots(figsize = (11, 6))
sns.despine(fig)
sns.histplot(
    df, x = 'SSC_mass', hue = 'Region', 
    multiple = 'stack', bins = 50, 
    palette = sns.color_palette(['red', 'magenta', 'blue'])
)
plt.xlim([0.1, 45])
plt.ylabel('Number of samples')
plt.xlabel('Rafted sediment concentration (% mass)')
sns.move_legend(ax, 'upper left', bbox_to_anchor = (0.05, 0.95), title = 'Region')

pct25 = df['SSC_mass'].quantile(0.25)
median = df['SSC_mass'].median()
pct75 = df['SSC_mass'].quantile(0.75)

ax.plot([median, median], [21, 24], 'k', lw = 2)
ax.plot([pct25, pct75], [22.5, 22.5], 'k:', lw = 2)
ax.annotate('[', [pct25, 22.25], ha = 'center', va = 'center', fontsize = 20, color = 'k')
ax.annotate(']', [pct75 - 0.1, 22.25], ha = 'center', va = 'center', fontsize = 20, color = 'k')
ax.annotate(f'{np.round(median, 2)}%', [median + 0.25, 19.25], ha = 'center', va = 'center', fontsize = 12, color = 'k')
ax.annotate(f'{np.round(pct75, 2)}%', [pct75 + 0.5, 19.25], ha = 'center', va = 'center', fontsize = 12, color = 'k')
ax.annotate('Median + IQR', [2, 25], ha = 'left', va = 'center', fontsize = 12, color = 'k')

ax.add_patch(plt.Rectangle((15.5, 0), 28, 10, color = 'k', linestyle = '--', alpha = 0.1, lw = 2))
ax.annotate('$\mathit{Solid}$ or $\mathit{stratified}$ basal ice facies', [29, 11], ha = 'center', va = 'center', fontsize = 12, color = 'k')
plt.savefig('figures/rafted-sediment-concentration.png', dpi = 300, transparent = True)
# plt.show()
