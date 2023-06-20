import matplotlib.pyplot as plt
from utility import plot_line_percent
import warnings
import os
import collections

warnings.filterwarnings('ignore')
print('\014')


graph_dir = '..\post_analysis\Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)
file_lists4farm = [r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_1'
                   r'/non_grazing/Heatmap-data_OF.csv',
                   r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_1'
                   r'/grazing/Heatmap-data_OF.csv',
                   r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_8'
                   r'/non_grazing/Heatmap-data_OF.csv',
                   r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_8'
                   r'/grazing/Heatmap-data_OF.csv']

fig, axes = plt.subplots(2, 2, figsize=(20, 8), sharex=True, sharey=False)
plot_line_percent(file_lists4farm[0], ax=axes[0, 0], col_name='Percent_change', x_label=''r'$\Delta{\theta}$, %',
                  y_label=''r'$\Delta{OF}$, %', title_str='WRE1 (without grazing)')
axes[0, 0].legend_.remove()
plot_line_percent(file_lists4farm[1], ax=axes[1, 0], col_name='Percent_change', x_label=''r'$\Delta{\theta}$, %',
                  y_label=''r'$\Delta{OF}$, %', title_str='WRE1 (with grazing)')
axes[1, 0].legend_.remove()
plot_line_percent(file_lists4farm[2], ax=axes[0, 1], col_name='Percent_change', x_label=''r'$\Delta{\theta}$, %',
                  y_label=''r'$\Delta{OF}$, %', title_str='WRE8 (without grazing)')
axes[0, 1].legend_.remove()
plot_line_percent(file_lists4farm[3], ax=axes[1, 1], col_name='Percent_change', x_label=''r'$\Delta{\theta}$, %',
                  y_label=''r'$\Delta{OF}$, %', title_str='WRE8 (with grazing)')

axes[1, 1].legend_.remove()
plt.tight_layout()

# Adding legend outside the plot area
entries = collections.OrderedDict()
for ax in axes.flatten():
    for handle, label in zip(*ax.get_legend_handles_labels()):
        entries[label] = handle
legend = fig.legend(
    entries.values(), entries.keys(),
    loc='lower center', bbox_to_anchor=(0.5, 0), ncol=11)

bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
fig.tight_layout(rect=(0, bbox.y1, 1, 1), h_pad=0.5, w_pad=0.5)

plt.savefig(os.path.join(graph_dir, 'Sensitivity_parameter_line-plot.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_parameter_line-plot.pdf'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_parameter_line-plot.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()