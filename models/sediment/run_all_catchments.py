"""Models sediment entrainment and glacial transport."""

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import equinox as eqx
import matplotlib.pyplot as plt
from landlab.plot import imshow_grid
from glacierbento import Field
from glacierbento.utils import freeze_grid, plot_triangle_mesh, plot_links
from glacierbento.components import FrozenFringe, DispersedLayer, SimpleGlacialEroder, TVDAdvector

regions = {
    'rolige-brae': 'CE',
    'sermeq-avannarleq': 'CW',
    'charcot-gletscher': 'CE',
    'sydbrae': 'CE',
    'kangiata-nunaata-sermia': 'SW',
    'eielson-gletsjer': 'CE',
    'narsap-sermia': 'SW',
    'kangilernata-sermia': 'CW',
    'dode-brae': 'CE',
    'daugaard-jensen-gletsjer': 'CE',
    'vestfjord-gletsjer': 'CE',
    'sermeq-kullajeq': 'CW',
    'bredegletsjer': 'CE',
    'magga-dan-gletsjer': 'CE',
    'graah-gletscher': 'CE',
    'akullersuup-sermia': 'SW',
    'eqip-sermia': 'CW',
    'kista-dan-gletsjer': 'CE'
}
CE_split_one = ['rolige-brae', 'charcot-gletscher', 'sydbrae', 'dode-brae', 'daugaard-jensen-gletsjer']
CE_split_two = ['vestfjord-gletsjer', 'bredegletsjer', 'magga-dan-gletsjer', 'graah-gletscher', 'kista-dan-gletsjer']

with open('./models/hydrology/outputs/post-hydrology-grids.pickle', 'rb') as f:
    landlab_grids = pickle.load(f)

for key, tmg in landlab_grids.items():
    if (regions[key] == 'CE') & (key in CE_split_two):

        glacier = key.replace('-', ' ').title()
        print(f'Running sediment transport model for {glacier}...')

        grid = freeze_grid(tmg)

        eroder = SimpleGlacialEroder(grid)
        fringe = FrozenFringe(grid)
        dispersed = DispersedLayer(grid)
        advector = TVDAdvector(grid, fields_to_advect = ['fringe_thickness', 'dispersed_thickness'])

        surface_elevation = jnp.asarray(tmg.at_node['surface_elevation'][:])
        velocity_on_links_unsigned = grid.map_mean_of_link_nodes_to_link(
            jnp.asarray(tmg.at_node['sliding_velocity'][:])
        )
        
        velocity_on_links = jnp.where(
            surface_elevation[grid.node_at_link_head] > surface_elevation[grid.node_at_link_tail],
            -velocity_on_links_unsigned,
            velocity_on_links_unsigned
        )
        velocity_x_link, velocity_y_link = grid.resolve_values_on_links(velocity_on_links)
        velocity_x = grid.map_mean_of_links_to_node(velocity_x_link)
        velocity_y = grid.map_mean_of_links_to_node(velocity_y_link)

        fields = {
            'sliding_velocity': Field(jnp.asarray(tmg.at_node['sliding_velocity'][:]), 'm/s', 'node'),
            'effective_pressure': Field(jnp.asarray(tmg.at_node['effective_pressure'][:]), 'Pa', 'node'),
            'till_thickness': Field(jnp.zeros(grid.number_of_nodes) + 1e-3, 'm', 'node'),
            'ice_thickness': Field(jnp.asarray(tmg.at_node['ice_thickness'][:]), 'm', 'node'),
            'basal_melt_rate': Field(jnp.asarray(tmg.at_node['basal_melt_rate'][:]), 'm/s', 'node'),
            'fringe_thickness': Field(jnp.zeros(grid.number_of_nodes) + 1e-3, 'm', 'node'),
            'theta': Field(jnp.ones(grid.number_of_nodes), '', 'node'),
            'base_temperature': Field(fringe._calc_base_temperature(), 'K', 'node'),
            'dispersed_thickness': Field(jnp.zeros(grid.number_of_nodes) + 1e-3, 'm', 'node'),
            'velocity_x': Field(velocity_x, 'm/s', 'link'),
            'velocity_y': Field(velocity_y, 'm/s', 'link')
        }

        advector = advector.initialize(fields)

        print('Model and fields initialized.')

        print('Stable time step:', advector.calc_stable_time_step(0.1) / 60 / 60 / 24, ' days.')

        @eqx.filter_jit
        def update(dt, fields):

            erosion = eroder.run_one_step(dt, fields)
            fields = eqx.tree_at(lambda t: t['till_thickness'].value, fields, erosion['till_thickness'].value)

            previous_fringe = fields['fringe_thickness'].value
            fringe_update = fringe.run_one_step(dt, fields)
            updated_fringe = jnp.where(
                (fringe_update['fringe_thickness'].value - previous_fringe) > fields['till_thickness'].value,
                previous_fringe + fields['till_thickness'].value,
                fringe_update['fringe_thickness'].value
            )
            updated_fringe = jnp.where(updated_fringe >= fringe.params['min_fringe'], updated_fringe, fringe.params['min_fringe'])
            updated_till = fields['till_thickness'].value - (updated_fringe - previous_fringe)
            updated_till = jnp.where(updated_till >= 0, updated_till, 0.0)
            fields = eqx.tree_at(
                lambda t: (t['fringe_thickness'].value, t['till_thickness'].value),
                fields,
                (updated_fringe, updated_till)
            )

            updated_theta = fringe._calc_undercooling(fields)
            fields = eqx.tree_at(lambda t: t['theta'].value, fields, updated_theta)

            previous_dispersed = fields['dispersed_thickness'].value
            dispersed_update = dispersed.run_one_step(dt, fields)
            updated_dispersed = jnp.where(
                fields['fringe_thickness'].value == fringe.params['min_fringe'],
                previous_dispersed - fields['basal_melt_rate'].value * dt,
                dispersed_update['dispersed_thickness'].value
            )
            # updated_dispersed = jnp.where(updated_dispersed >= fringe.params['min_fringe'], updated_dispersed, fringe.params['min_fringe'])
            fields = eqx.tree_at(lambda t: t['dispersed_thickness'].value, fields, updated_dispersed)

            advection = advector.run_one_step(dt, fields)
            advected_fringe = jnp.maximum(advection['fringe_thickness'].value, fringe.params['min_fringe'])
            advected_dispersed = jnp.maximum(advection['dispersed_thickness'].value, fringe.params['min_fringe'])
            fields = eqx.tree_at(
                lambda t: (t['fringe_thickness'].value, t['dispersed_thickness'].value),
                fields, 
                (advected_fringe, advected_dispersed)
            )

            fringe_bc = jnp.where(
                grid.status_at_node != 0,
                fringe.params['min_fringe'],
                fields['fringe_thickness'].value
            )
            dispersed_bc = jnp.where(
                grid.status_at_node != 0,
                fringe.params['min_fringe'],
                fields['dispersed_thickness'].value
            )
            fields = eqx.tree_at(
                lambda t: (t['fringe_thickness'].value, t['dispersed_thickness'].value),
                fields,
                (fringe_bc, dispersed_bc)
            )

            return fields

        import time
        start = time.time()
        fields = update(1.0, fields)
        timing = time.time() - start
        print(f'Estimated {timing} seconds per iteration.')

        for i in range(20):
            fields = update(i * 1000, fields)

        dt = advector.calc_stable_time_step(0.6)
        results = {'time': [], 'fields': []}
        total_time = 60 * 60 * 24 * 365 * 400
        nt = int(total_time / dt)
        print('Number of iterations:', nt)

        save_every = int(nt / 100)

        for i in range(nt):
            fields = update(dt, fields)

            if i % save_every == 0:
                print(f'Completed iteration {i}. Time elapsed: {(i + 1) * dt / (60 * 60 * 24 * 365)} years.')
                results['fields'].append(fields)
                results['time'].append(i * dt)

        print(f'Total simulation time: {(i + 1) * dt / (60 * 60 * 24 * 365)} years.')

        tmg.add_field('fringe_thickness', fields['fringe_thickness'].value, at = 'node', clobber = True)
        tmg.add_field('dispersed_thickness', fields['dispersed_thickness'].value, at = 'node', clobber = True)
        tmg.add_field('till_thickness', fields['till_thickness'].value, at = 'node', clobber = True)
        # landlab_grids[key] = tmg

        with open(f'./models/sediment/outputs/history/{key}-history.pickle', 'wb') as f:
            pickle.dump(results, f)

        with open(f'./models/sediment/outputs/grids/{key}-grid.pickle', 'wb') as f:
            pickle.dump(tmg, f)
