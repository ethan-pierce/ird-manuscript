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

    from ird_model.utils.plotting import plot_triangle_mesh
    plot_triangle_mesh(tmg, terminus)
    quit()

    fringe_at_cells = tmg.at_node['fringe_thickness'][tmg.node_at_cell][terminus_cells]
    fringe_sediment = fringe_at_cells * (1 - fringe_porosity) * 2700
    fringe_flux = np.sum(fringe_sediment * terminus_velocity * cell_outflow_width)

    dispersed_at_cells = tmg.at_node['dispersed_thickness'][tmg.node_at_cell][terminus_cells]
    dispersed_sediment = dispersed_at_cells * config['dispersed.concentration'] * 2700
    dispersed_flux = np.sum(dispersed_sediment * terminus_velocity * cell_outflow_width)

    return fringe_flux, dispersed_flux

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