"""Collate sample locations and RSC data."""

import numpy as np
import pandas as pd

sw = pd.read_excel('all_iceberg_samples.ods', sheet_name = 1)
cw = pd.read_excel('all_iceberg_samples.ods', sheet_name = 2)
ce18 = pd.read_excel('all_iceberg_samples.ods', sheet_name = 3)
ce19 = pd.read_excel('all_iceberg_samples.ods', sheet_name = 4)

locs = pd.read_csv('all-iceberg-locations.csv')

for idx, row in locs.iterrows():
    if row['region'] == 'SW':
        iceberg = row['name'].split('_')[-1]
    elif row['region'] == 'CW':
        iceberg = row['name'].split('ib')[-1]
        print(iceberg)
    elif row['region'] == 'CE':
        