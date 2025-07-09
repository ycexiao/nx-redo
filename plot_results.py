import utils
import conf
from utils import *
from conf import *
import importlib

importlib.reload(conf)
importlib.reload(utils)

targets = ["cs", "cn", "bl"]
elements = ["Ti", "Cu", "Fe", "Mn"]
# i = 0
# results = bunch_load_pickle(*results_path)[i]
# results_path = "results-40.pkl"
# with open(results_path, "rb") as f:
# results = pickle.load(f)

plt.style.use("seaborn-v0_8")

plot_features = [
    ["x_pdf"],
    ["nx_pdf"],
    ["x_pdf", "n_pdf"],
    ["xanes", "x_pdf"],
    ["xanes", "nx_pdf"],
    ["xanes", "x_pdf", "n_pdf"],
]

baseline_features = [
    ["xanes"],
    ["xanes", "diff_x_pdf"],
]


## results plot
def results_plot(
    results_database: easyDatabase,
    tar="cs",
    title="Oxidation states",
    ylabel="F1 score",
    ylim=None,
):
    fig, axes = plt.subplots(1, len(elements), figsize=(11.2, 6.4))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    baseline_names = convert_features_name(baseline_features)

    target = tar
    for j in range(len(elements)):
        scores = []
        errors = []
        for k in range((len(plot_features))):
            keys_dict = {
                "target": target,
                "element": elements[j],
                "features": plot_features[k],
            }
            data = results_database.filter_data(keys_dict).value
            scores.append(data["score"])
            errors.append(data["error"])

        bar = axes[j].bar(range(len(plot_features)), scores, yerr=errors)
        for k in range(len(scores)):
            axes[j].text(
                range(len(plot_features))[k],
                scores[k],
                "{:.3f}".format(scores[k]),
            )

    for j in range(len(elements)):
        scores = []
        for k in range(len(baseline_features)):
            keys_dict = {
                "target": target,
                "element": elements[j],
                "features": baseline_features[k],
            }
            data = results_database.filter_data(keys_dict).value
            mean_value = data["score"]
            std_value = data["error"]
            axes[j].set_xlim([-1, len(plot_features)])
            xlim = axes[j].get_xlim()
            axes[j].fill_between(
                np.linspace(*xlim, 100),
                np.ones(100) * (mean_value - std_value / 2),
                np.ones(100) * (mean_value + std_value / 2),
                alpha=0.5,
                color=colors[k + 2],
                label=baseline_names[k],
            )
            if ylim:
                axes[j].set_ylim(*ylim)

    # plot
    for j, ax in enumerate(axes):
        set_ax_style(
            ax,
            xticks=convert_features_name(plot_features),
            ylabel=ylabel,
            title=elements[j],
        )
        if j != 0:
            set_ax_style(ax, ylabel="", yticks=[])

    # setp
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    # plt.setp(hls, color=colors[1])
    # fig.legend([hls[0]], ["XANES+dPDF"], bbox_to_anchor=(0.98, 0.88))
    plt.subplots_adjust(left=0.1, right=0.5, top=0.9, bottom=0.1)
    # fig.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.suptitle(title, fontsize=18)
    plt.legend()
    plt.tight_layout()


## feature importance plot
def feature_importance_plot(results_database, tar, title):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    feature_names = convert_features_name(plot_features)
    baseline_names = convert_features_name(baseline_features)
    fig, axes = plt.subplots(
        len(plot_features), len(elements), figsize=(12, 6)
    )
    for j in range(len(plot_features)):
        for k in range(len(elements)):
            keys_dict = {
                "target": tar,
                "features": plot_features[j],
                "element": elements[k],
            }
            data = results_database.filter_data(keys_dict).value
            importances = data["importances"]
            axes[j, k].plot(
                range(len(importances[0])),
                *importances,
                color="gray",
                alpha=0.2,
            )
            axes[j, k].set_xticks([])
            means = np.mean(importances, axis=0)
            for h in range(int(len(means) // 100)):
                axes[j, k].plot(
                    np.arange(h * 100, h * 100 + 100),
                    means[h * 100 : h * 100 + 100],
                    color=colors[1 + h],
                )

            peaks, _ = find_peaks(means, height=0.02)
            ylim = axes[j, k].get_ylim()
            axes[j, k].set_xlim([0, 300])
            for ind in peaks:
                axes[j, k].vlines(
                    np.arange(len(importances[0]))[ind],
                    *ylim,
                    color="k",
                    linestyle="--",
                    linewidth=0.5,
                )

            if j == 0:
                axes[j, k].set_title(elements[k])
            if k == 0:
                axes[j, k].set_ylabel(
                    feature_names[j], rotation=45, ha="right"
                )

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)


if __name__ == "__main__":
    filename = "results-40-in-database.pkl"
    with open(filename, "rb") as f:
        results_database = pickle.load(f)

    run_kwargs = [
        {
            "tar": "cs",
            "title": "Oxidation State",
            "ylabel": "weighted mean F1 score",
            "ylim": [0.6, 0.95],
        },
        {
            "tar": "cn",
            "title": "Coordination Number",
            "ylabel": "weighted mean F1 score",
            "ylim": [0.6, 0.95],
        },
        {
            "tar": "bl",
            "title": "Bond Length",
            "ylabel": "RMSE (% mean BL)",
            "ylim": None,
        },
    ]
    # for kwarg in run_kwargs:
    #     results_plot(results_database, **kwarg)
    #     plt.legend(bbox_to_anchor=(1.25, 1.15))
    # plt.show()
    # imp_kwargs = [
    # {"tar": "cs", "title": "Oxidation State"},
    # {"tar": "cn", "title": "Coordination Number"},
    # {"tar": "bl", "title": "Bond Length"},
    # ]

    # for kwarg in imp_kwargs:
    # feature_importance_plot(results_database, **kwarg)
    # plt.show()
    # results_plot
    # results_database,
    # tar="bl",
    # title="Oxidation State",
    # ylabel="weighted mean F1 score",
    # ylim=None,
    # )
    # feature_importance_plot(
    #     results_database, tar="cs", title="Coordination number"
    # )
    # features = ["xanes", "x_pdf", "nx_pdf"]
    # target = "cn"
    # element = "Ti"
    # lookup(results, "scores", features, target, element)
    #
