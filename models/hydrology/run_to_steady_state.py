"""Models the steady-state distributed drainage system."""

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import equinox as eqx
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from glacierbento import Field
from glacierbento.utils import freeze_grid
from glacierbento.components import DistributedDrainageSystem, ShallowIceApproximation


with open('./models/hydrology/outputs/post-hydrology-grids.pickle', 'rb') as f:
    landlab_grids = pickle.load(f)

failed = []

for key, tmg in landlab_grids.items():
    if key in ['charcot-gletscher', 'graah-gletscher']: # lazy but lets us keep the indentation level

        glacier = key.replace('-', ' ').title()
        print(f'Running hydrology model for {glacier}...')

        grid = freeze_grid(tmg)

        model = DistributedDrainageSystem(grid)
        sia = ShallowIceApproximation(grid)

        fields = {
            'ice_thickness': Field(jnp.asarray(tmg.at_node['ice_thickness'][:]), 'm', 'node'),
            'bed_elevation': Field(jnp.asarray(tmg.at_node['bed_elevation'][:]), 'm', 'node'),
            'surface_elevation': Field(jnp.asarray(tmg.at_node['surface_elevation'][:]), 'm', 'node'),
            'basal_melt_rate': Field(jnp.asarray(tmg.at_node['basal_melt_rate'][:]), 'm/s', 'node'),
            'sheet_flow_height': Field(jnp.zeros(grid.number_of_nodes) + 0.001, 'm', 'node')
        }

        sia_output = sia.run_one_step(0.0, fields)        
        U_deformation = grid.map_mean_of_links_to_node(jnp.abs(sia_output['deformation_velocity'].value))
        U_surface = jnp.asarray(np.sqrt(tmg.at_node['vx'][:]**2 + tmg.at_node['vy'][:]**2))
        U_sliding = jnp.where(U_surface > U_deformation, U_surface - U_deformation, 0.0)
        fields['sliding_velocity'] = Field(U_sliding, 'm/s', 'node')
        tmg.add_field('sliding_velocity', U_sliding, at = 'node', clobber = True)

        phi0 = jnp.asarray(
            model.params['ice_density'] 
            * model.params['gravity'] 
            * tmg.at_node['ice_thickness'][:]
        )
        fields['potential'] = Field(phi0, 'Pa', 'node')

        ice_pressure = (
            fields['ice_thickness'].value * model.params['ice_density'] * model.params['gravity']
        )

        base_pressure = (
            fields['bed_elevation'].value * model.params['water_density'] * model.params['gravity']
        )

        @eqx.filter_jit
        def update(dt, fields):
            output = model.run_one_step(dt, fields)

            updated_potential = jnp.where(
                output['potential'].value < base_pressure,
                base_pressure,
                jnp.where(
                    output['potential'].value > ice_pressure + base_pressure,
                    ice_pressure + base_pressure,
                    output['potential'].value
                )
            )

            updated_flow_height = jnp.where(
                output['sheet_flow_height'].value < 1e-3,
                1e-3,
                output['sheet_flow_height'].value,
            )

            output = eqx.tree_at(
                lambda t: (t['potential'], t['sheet_flow_height']),
                output,
                (Field(updated_potential, 'Pa', 'node'), Field(updated_flow_height, 'm', 'node'))
            )

            fields = eqx.tree_at(
                lambda t: (t['potential'], t['sheet_flow_height']),
                fields,
                (output['potential'], output['sheet_flow_height'])
            )

            return fields

        fields = update(1.0, fields)

        dts = [0.0]
        max_diffs = [0.0]
        prev_phi = jnp.zeros(grid.number_of_nodes)
        fields_history = []

        if key in ['bredegletsjer', 'graah-gletscher', 'kista-dan-gletsjer', 'magga-dan-gletsjer', 'sydbrae']:
            max_i = 20
        else:
            max_i = 100

        for i in range(max_i):
            dt = 60 * 60 * np.min([i, 24])
            
            try:
                fields = update(dt, fields)
            except:
                print('Failed on iteration ' + str(i))
                failed.append(key)
                break

            dts.append(dt)
            diffs = jnp.abs(fields['potential'].value - prev_phi)
            max_diffs.append(jnp.percentile(diffs, 75))
            prev_phi = fields['potential'].value.copy()
            fields_history.append(fields)

            print('Completed step ' + str(i) + '; max diff: ' + f'{np.round(max_diffs[-1], decimals = 6)}' + ' Pa.')

            if (i > 3) & (max_diffs[-1] < 1000):
                break

        print('Total simulation time: ' + f'{np.round(np.sum(dts) / 60 / 60 / 24, 2)}' + ' days.')
        
        # plt.plot(np.cumsum(dts[1:]) / 60, max_diffs[1:])
        # plt.title('Max. change in sheet flow height (m)')
        # plt.xlabel('Simulation time (minutes)')
        # plt.show()

        # imshow_grid(tmg, fields['potential'].value, cmap = 'viridis')
        # plt.title('Hydraulic potential (Pa)')
        # plt.show()

        # imshow_grid(tmg, model.calc_effective_pressure(fields), cmap = 'viridis')
        # plt.title('Effective pressure (Pa)')
        # plt.show()

        # imshow_grid(tmg, fields['sheet_flow_height'].value, cmap = 'viridis')
        # plt.title('Sheet flow height (m)')
        # plt.show()

        effective_pressure = model.calc_effective_pressure(fields)
        tmg.add_field('effective_pressure', effective_pressure, at = 'node', clobber = True)
        tmg.add_field('hydraulic_potential', fields['potential'].value, at = 'node', clobber = True)
        tmg.add_field('sheet_flow_height', fields['sheet_flow_height'].value, at = 'node', clobber = True)
        landlab_grids[key] = tmg

        with open('./models/hydrology/outputs/' + key + '_history.pickle', 'wb') as f:
            pickle.dump(fields_history, f)

        with open('./models/hydrology/outputs/post-hydrology-grids.pickle', 'wb') as f:
            pickle.dump(landlab_grids, f)

print('Failed on the following keys: ', failed)