"""Models steady-state pressure in a distributed drainage system."""

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from landlab_triangle import TriangleModelGrid

from ird_model.utils.core import Field
from ird_model.utils.static_grid import freeze_grid, StaticGrid
from ird_model.components import DistributedDrainageSystem


def run_to_steady_state(
    tmg: TriangleModelGrid,
    config: dict
):
    """Run the hydrology model to steady state."""
    grid = freeze_grid(tmg)

    model = DistributedDrainageSystem(grid)
    model.params['cavity_spacing'] = config['cavity_spacing']
    model.params['bed_bump_height'] = config['bed_bump_height']
    model.params['sheet_conductivity'] = config['sheet_conductivity']
    model.params['flow_exp_a'] = config['flow_exp_a']
    model.params['flow_exp_b'] = config['flow_exp_b']
  
    fields = set_initial_fields(tmg, model, config)

    min_sheet_flow_height = config['sheet_flow_height.minimum']

    @eqx.filter_jit
    def update(fields: dict[str, Field], dt: float):
        output = model.run_one_step(dt, fields)

        updated_potential = jnp.where(
            output['potential'].value < fields['base_pressure'].value,
            fields['base_pressure'].value,
            jnp.where(
                output['potential'].value > fields['ice_pressure'].value + fields['base_pressure'].value,
                fields['ice_pressure'].value + fields['base_pressure'].value,
                output['potential'].value
            )
        )

        updated_flow_height = jnp.where(
            output['sheet_flow_height'].value < min_sheet_flow_height,
            min_sheet_flow_height,
            output['sheet_flow_height'].value,
        )

        fields = eqx.tree_at(
            lambda t: (t['potential'], t['sheet_flow_height']),
            fields,
            (Field(updated_potential, 'Pa', 'node'), Field(updated_flow_height, 'm', 'node'))
        )

        return fields

    fields = update(fields, 1.0)

    dts = [0.0]
    max_diffs = [0.0]
    prev_phi = jnp.zeros(grid.number_of_nodes)

    for i in range(config['time_steps']):
        dt = np.minimum(config['dt'], 60 * 60 * i) # Spin up to maximum dt

        try:
            fields = update(fields, dt)
        except:
            print('Failed on iteration ' + str(i))
            break

        dts.append(dt)
        diffs = jnp.abs(fields['potential'].value - prev_phi)
        max_diffs.append(jnp.percentile(diffs, config['max_diff.percentile']))
        prev_phi = fields['potential'].value.copy()

        print('Completed step ' + str(i) + '; max diff: ' + f'{np.round(max_diffs[-1], decimals = 6)}' + ' Pa.')

        if (i > 3) & (max_diffs[-1] < config['max_diff.cutoff']):
            break

    print('Total simulation time: ' + f'{np.round(np.sum(dts) / 60 / 60 / 24, 2)}' + ' days.')

    effective_pressure = model.calc_effective_pressure(fields)
    tmg.add_field('effective_pressure', effective_pressure, at = 'node', clobber = True)
    tmg.add_field('hydraulic_potential', fields['potential'].value, at = 'node', clobber = True)
    tmg.add_field('sheet_flow_height', fields['sheet_flow_height'].value, at = 'node', clobber = True)
    
    return tmg
    
def set_initial_fields(tmg: TriangleModelGrid, model: DistributedDrainageSystem, config: dict):
    """Set the initial fields for the hydrology model."""
    phi0 = jnp.asarray(
        model.params['ice_density'] 
        * model.params['gravity'] 
        * tmg.at_node['ice_thickness'][:]
        * config['potential.initial']
    )

    ice_pressure = (
        tmg.at_node['ice_thickness'][:] * model.params['ice_density'] * model.params['gravity']
    )

    base_pressure = (
        tmg.at_node['bed_elevation'][:] * model.params['water_density'] * model.params['gravity']
    )

    fields = {
        'ice_thickness': Field(jnp.asarray(tmg.at_node['ice_thickness'][:]), 'm', 'node'),
        'bed_elevation': Field(jnp.asarray(tmg.at_node['bed_elevation'][:]), 'm', 'node'),
        'surface_elevation': Field(jnp.asarray(tmg.at_node['surface_elevation'][:]), 'm', 'node'),
        'basal_melt_rate': Field(jnp.asarray(tmg.at_node['basal_melt_rate'][:]), 'm/s', 'node'),
        'sheet_flow_height': Field(jnp.zeros(tmg.number_of_nodes) + 0.001, 'm', 'node'),
        'sliding_velocity': Field(jnp.asarray(tmg.at_node['sliding_velocity'][:]), 'm/s', 'node'),
        'potential': Field(jnp.asarray(phi0), 'Pa', 'node'),
        'ice_pressure': Field(jnp.asarray(ice_pressure), 'Pa', 'node'),
        'base_pressure': Field(jnp.asarray(base_pressure), 'Pa', 'node')
    }

    return fields
