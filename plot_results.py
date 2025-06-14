import utils
import conf
from utils import *
from conf import *
import importlib
importlib.reload(conf)
importlib.reload(utils)

elements = ['Ti', 'Cu', 'Fe', 'Mn']
# i=0
# results = bunch_load_pickle(*results_path)[i]
# results_path = 'results-06-10-08-46-481.pkl'
with open(results_path[0].name, 'rb') as f:
    results = pickle.load(f)


plt.style.use('seaborn-v0_8')

plot_features = [
    ['xanes'],
    ['xanes', 'x_pdf', 'n_pdf'],
    ['xanes', 'x_pdf', 'nx_pdf'],
]

baseline_features = [
    ['xanes', 'diff_x_pdf'],
]

## results plot
def results_plot(tar, ylabel, title, ylim=[0.6,1]):
    fig, axes = plt.subplots(1, len(elements), figsize=(11.2,6.4))
    fig, axes, bars = bunch_bar_plot(results, 'test_scores', tar, elements, plot_features, fig=fig, axes=axes)
    fig, axes, hls = bunch_hline_plot(results, 'test_scores', tar, elements, baseline_features, fig=fig, axes=axes)

    # plot
    for j, ax in enumerate(axes):
        set_ax_style(ax, xticks=convert_features_name(plot_features), ylabel=ylabel, title=elements[j], ylim=ylim)
        if j != 0:
            set_ax_style(ax, ylabel='', yticks=[])

    # setp
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.setp(hls, color=colors[1])
    fig.legend([hls[0]], ['XANES+dPDF'], bbox_to_anchor=(0.98,0.88))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.suptitle(title, fontsize=18)

    plt.tight_layout()

## feature importance plot
def feature_importance_plot(tar, title):
    feature_names = convert_features_name(plot_features)
    fig, axes = plt.subplots(len(plot_features), len(elements), figsize=(12,6))
    for i, ele in enumerate(elements):
        fig, tmp_axes =feature_importances_plot(results, 'importances', tar, ele, plot_features, fig=fig, axes=axes[:,i])
        axes[0,i].set_title(elements[i])
        if i ==0:
            for j, ax in enumerate(tmp_axes):
                ax.set_ylabel(feature_names[j], rotation=45, ha='right')
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()


# results_plot('cn', 'F1 score', 'Coordination number')
# plt.savefig('imgs/results_cn.png', dpi=625)
# results_plot('cs', 'F1 score', 'Charged states')
# plt.savefig('imgs/results_cs.png', dpi=625)
# results_plot('bl', 'RMSE (% mean BL)', 'Bond length', ylim=[0,0.04])
# plt.savefig('imgs/results_bl.png', dpi=625)


# feature_importance_plot('cn', 'Coordination number')
# plt.savefig('imgs/importances_cn.png', dpi=625)
# feature_importance_plot('cs', 'Charged states')
# plt.savefig('imgs/importances_cs.png', dpi=625)
# feature_importance_plot('bl', 'Bond length')
# plt.savefig('imgs/importances_bl.png', dpi=625)
plt.show()

if __name__ == '__main__':
    pass
