"""Model basal sediment entrainment and transport."""

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import equinox as eqx
from landlab_triangle import TriangleModelGrid

from ird_model.utils.core import Field
from ird_model.utils.static_grid import freeze_grid, StaticGrid
from ird_model.components.frozen_fringe import FrozenFringe
from ird_model.components.dispersed_layer import DispersedLayer
from ird_model.components.simple_glacial_eroder import SimpleGlacialEroder
from ird_model.components.tvd_advection import TVDAdvector


def run_sediment_transport(tmg: TriangleModelGrid, config: dict) -> dict[str, Field]:
    """Run the sediment transport model."""
    grid = freeze_grid(tmg)
    eroder, advector, fringe, dispersed = setup_components(grid, config)
    fields = set_initial_fields(grid, tmg, fringe)
    advector = advector.initialize(fields)

    @eqx.filter_jit
    def update(dt: float, fields: dict[str, Field]):
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

    print('Model and fields initialized.')
    print('Spinning up sediment entrainment module')

    fields = update(1.0, fields)

    import time
    start = time.time()
    fields = update(1.0, fields)
    timing = time.time() - start
    print(f'Estimated {timing} seconds per iteration.')

    for i in range(20):
        fields = update(1.0, fields)
    for i in range(20):
        fields = update((i + 1) * 100, fields)
    
    print('Stable time step:', advector.calc_stable_time_step(config['CFL']) / 60 / 60 / 24, ' days.')

    dt = advector.calc_stable_time_step(config['CFL'])
    total_time = config['n_years'] * 365 * 24 * 60 * 60
    nt = int(total_time / float(dt))
    print(f'Running {config["n_years"]} years ({nt:,} iterations) - Estimated time: {nt * timing / 60:.1f} min') 

    for i in range(nt):
        fields = update(dt, fields)

    tmg.add_field('fringe_thickness', fields['fringe_thickness'].value, at = 'node', clobber = True)
    tmg.add_field('dispersed_thickness', fields['dispersed_thickness'].value, at = 'node', clobber = True)
    tmg.add_field('till_thickness', fields['till_thickness'].value, at = 'node', clobber = True)

    return tmg
    
def setup_components(grid: StaticGrid, config: dict):
    """Set up the components for the sediment transport model."""
    eroder = SimpleGlacialEroder(grid)
    eroder.update_param('rate_coefficient', config['erosion.coefficient'] * 31556926)
    eroder.update_param('sliding_exponent', config['erosion.exponent'])

    advector = TVDAdvector(grid, fields_to_advect = ['fringe_thickness', 'dispersed_thickness'])

    fringe = FrozenFringe(grid)
    fringe.update_param('surface_energy', config['fringe.surface_energy'])
    fringe.update_param('pore_throat_radius', config['fringe.pore_throat_radius'])
    fringe.update_param('till_porosity', config['fringe.till_porosity'])
    fringe.update_param('till_permeability', config['fringe.till_permeability'])
    fringe.update_param('till_grain_radius', config['fringe.till_grain_radius'])
    fringe.update_param('film_thickness', config['fringe.film_thickness'])
    fringe.update_param('alpha', config['fringe.alpha'])
    fringe.update_param('beta', config['fringe.beta'])
    fringe.update_param('min_fringe', config['fringe_thickness.minimum'])

    dispersed = DispersedLayer(grid)
    dispersed.update_param('critical_depth', 100.0)

    return (eroder, advector, fringe, dispersed)

def set_initial_fields(grid: StaticGrid, tmg: TriangleModelGrid, fringe: FrozenFringe):
    """Set the initial fields for the sediment transport model."""
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
        'velocity_x': Field(tmg.at_node['vx'][:], 'm/s', 'node'),
        'velocity_y': Field(tmg.at_node['vy'][:], 'm/s', 'node')
    }

    return fields
    