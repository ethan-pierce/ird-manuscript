import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from landlab.plot import imshow_grid

with open('./models/hydrology/outputs/post-hydrology-grids.pickle', 'rb') as f:
    grids = pickle.load(f)

for key, grid in grids.items():
    glacier = key.replace('-', ' ').title()
    field = grid.at_node['effective_pressure'][:]

    # imshow_grid(grid, field, cmap = 'viridis', vmin = 0, vmax = np.percentile(field, 90))
    # plt.title(f'{glacier} effective pressure (Pa)')
    # plt.savefig(f'./models/hydrology/outputs/figures/pressure/{key}-effective-pressure.png', dpi = 300)
    # plt.close()

    # imshow_grid(grid, grid.at_node['sheet_flow_height'][:], cmap = 'viridis')
    # plt.title(f'{glacier} sheet flow height (m)')
    # plt.savefig(f'./models/hydrology/outputs/figures/sheetflow/{key}-sheet-flow.png', dpi = 300)
    # plt.close()

    if key in ['eqip-sermia', 'kangilernata-sermia', 'sermeq-avannarleq', 'sermeq-kullajeq']:
        im = plt.scatter(grid.node_x, grid.node_y, c = field, cmap = 'viridis', s = 0.1, vmin = 0, vmax = 1)

plt.colorbar(im)
plt.show()

# rolige = grids['rolige-brae']
# imshow_grid(
#     rolige, 
#     rolige.at_node['sheet_flow_height'][:], 
#     plot_name = 'Rolige Brae sheet flow height (m)',
#     cmap = 'viridis'
# )
# plt.show()

# old_pressure = rolige.at_node['ice_thickness'][:] * 917 * 9.81 * 0.8
# imshow_grid(rolige, old_pressure, cmap = 'viridis', vmin = 0, vmax = np.percentile(old_pressure, 90))
# plt.title('Rolige Brae effective pressure (Pa)')
# plt.show()

# new_pressure = rolige.at_node['effective_pressure'][:]
# imshow_grid(rolige, new_pressure, cmap = 'viridis', vmin = 0, vmax = np.percentile(new_pressure, 90))
# plt.title('Rolige Brae effective pressure (Pa)')
# plt.show()

# imshow_grid(rolige, old_pressure - new_pressure, cmap = 'RdBu_r', symmetric_cbar = True)
# plt.title('Difference (scaled - modeled) (Pa)')
# plt.show()