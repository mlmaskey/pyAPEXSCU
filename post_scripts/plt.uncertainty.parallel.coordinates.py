import os
import warnings
from utilis_parallel_coordinates import parallel_coordinates

warnings.filterwarnings('ignore')

# choose attributes below
# ['PRCP', 'WYLD', 'YSD', 'RUS2', 'ET', 'PET', 'DPRK', 'BIOM', 'STD', 'STDL', 'STL', 'YLDF', 'YLDG', 'TN', 'TP',
#  'LAI', 'WS', 'TS', 'NS', 'PS', 'Mode']

graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

plt = parallel_coordinates(site='Farm_1', scenario='non_grazing',
                           cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDF', 'TN', 'TP'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_1', scenario='grazing',
                           cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDF', 'TN', 'TP'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_g_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_g_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_g_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_8', scenario='non_grazing',
                           cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDG', 'TN', 'TP'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_n_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_n_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_n_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_8', scenario='grazing',
                           cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDG', 'TN', 'TP'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_g_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_g_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_g_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_1', scenario='non_grazing',
                           cols=['WYLD', 'DPRK', 'PET', 'BIOM', 'YLDF', 'WS', 'TS', 'NS', 'PS'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_stress_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_stress_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_stress_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_1', scenario='grazing',
                           cols=['WYLD', 'DPRK', 'PET', 'BIOM', 'YLDF', 'WS', 'TS', 'NS', 'PS'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_g_stress_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_g_stress_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_g_stress_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_8', scenario='non_grazing',
                           cols=['WYLD', 'DPRK', 'PET', 'BIOM', 'YLDG', 'WS', 'TS', 'NS', 'PS'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 8))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_stress_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_stress_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_n_stress_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

plt = parallel_coordinates(site='Farm_8', scenario='grazing',
                           cols=['WYLD', 'DPRK', 'PET', 'BIOM', 'YLDG', 'WS', 'TS', 'NS', 'PS'],
                           colors=['blue', 'cyan', 'yellow', 'green', 'orange', 'magenta', 'red', 'black'],
                           fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_g_stress_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_g_stress_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_g_stress_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")
