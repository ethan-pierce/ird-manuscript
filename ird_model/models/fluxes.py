"""Calculate sediment fluxes from model output."""

import numpy as np

from landlab_triangle import TriangleModelGrid
from ird_model.utils.static_grid import freeze_grid, StaticGrid


def calc_fluxes(tmg: TriangleModelGrid, config: dict):
    """Calculate sediment fluxes from model output."""
    fringe_porosity = config['sediment']['fringe.till_porosity']
    config = config['fluxes']

    terminus, terminus_cells = find_terminus(tmg, config)
    terminus_velocity, cell_outflow_width = calc_velocity_outflow(tmg, config)

    fringe_pct99 = np.percentile(tmg.at_node['fringe_thickness'][tmg.node_at_cell], config['fringe_thickness.cutoff'] - 0.5)
    fringe = np.where(tmg.at_node['fringe_thickness'][tmg.node_at_cell] > fringe_pct99, fringe_pct99, tmg.at_node['fringe_thickness'][tmg.node_at_cell])
    fringe_sediment = fringe[terminus_cells] * (1 - fringe_porosity) * 2700
    fringe_flux = np.sum(fringe_sediment * terminus_velocity * cell_outflow_width)

    dispersed_pct99 = np.percentile(tmg.at_node['dispersed_thickness'][tmg.node_at_cell], config['dispersed_thickness.cutoff'])
    dispersed = np.where(tmg.at_node['dispersed_thickness'][tmg.node_at_cell] > dispersed_pct99, dispersed_pct99, tmg.at_node['dispersed_thickness'][tmg.node_at_cell])
    dispersed_sediment = dispersed[terminus_cells] * config['dispersed.concentration'] * 2700
    dispersed_flux = np.sum(dispersed_sediment * terminus_velocity * cell_outflow_width)

    ice_thickness = tmg.at_node['ice_thickness'][tmg.node_at_cell]
    ice_thickness = np.where(ice_thickness > 0, ice_thickness, 0)
    ice_flux = np.sum(ice_thickness[terminus_cells] * terminus_velocity * cell_outflow_width * 917)

    return fringe_flux, dispersed_flux, ice_flux

def find_terminus(grid, config: dict):
    """Find the terminus of the glacier."""
    terminus = np.where(
        (grid.status_at_node != 0)
        & (grid.node_x > config['terminus.min_x'])
        & (grid.node_x < config['terminus.max_x'])
        & (grid.node_y > config['terminus.min_y'])
        & (grid.node_y < config['terminus.max_y']),
        1,
        0
    )
    adjacent_nodes = grid.adjacent_nodes_at_node[terminus == 1]
    adjacent_nodes = adjacent_nodes[adjacent_nodes != -1]
    terminus_cells = np.unique(grid.cell_at_node[adjacent_nodes])
    terminus_cells = terminus_cells[terminus_cells != -1]

    return terminus, terminus_cells

def calc_velocity_outflow(tmg: TriangleModelGrid, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the velocity of the glacier at the terminus."""
    grid = freeze_grid(tmg)

    terminus, terminus_cells = find_terminus(grid, config)

    terminus_faces = tmg.faces_at_cell[terminus_cells]
    outflow_faces = np.where(grid.status_at_link[grid.link_at_face[terminus_faces]] != 0, 1, 0)
    face_width = grid.length_of_face[terminus_faces] * outflow_faces
    cell_outflow_width = np.sum(face_width, axis = 1)

    velocity = np.abs(tmg.at_node['sliding_velocity'])[grid.node_at_cell] * 31556926
    terminus_velocity = velocity[terminus_cells]

    return terminus_velocity, cell_outflow_width