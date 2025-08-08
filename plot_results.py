from matplotlib import pyplot as plt
from helper import easyDatabase
from ml import convert_features_name
import numpy as np
from scipy.signal import find_peaks
import pickle
import matplotlib

matplotlib.use("TkAgg")

targets = ["cs", "cn", "bl"]
elements = ["Ti", "Cu", "Fe", "Mn"]
# i = 0
# results = bunch_load_pickle(*results_path)[i]
# results_path = "results-40.pkl"
# with open(results_path, "rb") as f:
# results = pickle.load(f)

plt.style.use("seaborn-v0_8")

plot_features = [
    ["nx_pdf"],
    ["x_pdf", "n_pdf"],
    ["xanes", "n_pdf"],
    ["xanes", "x_pdf"],
    ["xanes", "nx_pdf"],
    ["xanes", "x_pdf", "n_pdf"],
]

baseline_features = [
    ["n_pdf"],
    ["x_pdf"],
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
            data = results_database.filter_data(
                ["target", "element", "features"],
                [target, elements[j], plot_features[k]],
            ).value
            scores.append(np.mean(data["test_scores"]))
            errors.append(np.std(data["test_scores"]))

        bar = axes[j].bar(range(len(plot_features)), scores, yerr=errors)
        for k in range(len(scores)):
            axes[j].text(
                range(len(plot_features))[k],
                scores[k],
                "{:.3f}".format(scores[k]),
            )

    for j in range(len(elements)):
        scores = []
        errors = []
        for k in range(len(baseline_features)):
            keys_dict = {
                "target": target,
                "element": elements[j],
                "features": baseline_features[k],
            }
            data = results_database.filter_data(
                ["target", "element", "features"],
                [target, elements[j], baseline_features[k]],
            ).value
            scores.append(np.mean(data["test_scores"]))
            errors.append(np.std(data["test_scores"]))
            axes[j].set_xlim([-1, len(plot_features)])
            xlim = axes[j].get_xlim()
            axes[j].fill_between(
                np.linspace(*xlim, 100),
                np.ones(100) * (scores[-1] - errors[-1] / 2),
                np.ones(100) * (scores[-1] + errors[-1] / 2),
                alpha=0.5,
                color=colors[k + 2],
                label=baseline_names[k],
            )
            if ylim:
                axes[j].set_ylim(*ylim)
    for j in range(len(elements)):
        axes[j].set_title(elements[j])
        axes[j].set_xticks(np.arange(len(plot_features)) + 1)
        axes[j].set_xticklabels(
            convert_features_name(plot_features), rotation=45, ha="right"
        )
        if j == 0:
            axes[j].set_ylabel(ylabel)
        else:
            axes[j].set_yticks([])

    # setp
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()


## feature importance plot
def feature_importance_plot(results_database, tar, title):
    plot_features = [
        ["x_pdf", "n_pdf"],
        ["nx_pdf"],
        ["xanes"],
        ["xanes", "n_pdf"],
        ["xanes", "nx_pdf"],
        ["xanes", "x_pdf", "n_pdf"],
    ]
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
            data = results_database.filter_data(
                ["target", "features", "element"],
                [tar, plot_features[j], elements[k]],
            ).value
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
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)


if __name__ == "__main__":
    # data_path = "newest_combined_data.pkl"
    # with open(data_path, "rb") as f:
    # data = pickle.load(f)
    # results_database = data["results"]

    ## add xanes_npdf into results-40-database
    # xanes_npdf_database = result1easyDatabase(result_data)
    # filename = "results-40-in-database.pkl"
    # with open(filename, "rb") as f:
    # results_database = pickle.load(f)
    # results_database.keynames = results_database.members[0].get_keynames()
    # results_database = results_database.combine(xanes_npdf_database)
    # tmp = results_database.filter_data(
    #     {"target": "bl", "element": "Ti", "features": ["xanes", "n_pdf"]}
    # )
    # print(tmp.value)

    filename = "combined_data.pkl"
    filename = "newest_combined_data.pkl"
    with open(filename, "rb") as f:
        results_database = pickle.load(f)
    results_database = results_database["results"]
    run_kwargs = [
        {
            "tar": "cs",
            "title": "Oxidation State",
            "ylabel": "weighted mean F1 score",
            "ylim": [0, 1],
        },
        {
            "tar": "cn",
            "title": "Coordination Number",
            "ylabel": "weighted mean F1 score",
            "ylim": [0, 1],
        },
        {
            "tar": "bl",
            "title": "Bond Length",
            "ylabel": "RMSE (% mean BL)",
            "ylim": [0, 0.06],
        },
    ]
    save_file_names = [
        "result_cs.png",
        "result_cn.png",
        "result_bl.png",
    ]
    for i, kwarg in enumerate(run_kwargs):
        results_plot(results_database, **kwarg)
        legends = [
            ax.legend() for ax in plt.gcf().axes if ax.get_legend() is not None
        ]
        plt.tight_layout()
        pars = plt.gcf().subplotpars
        plt.subplots_adjust(right=pars.right - 0.04, top=pars.top - 0.03)
        plt.legend(bbox_to_anchor=(1.2, 1.23))
        plt.savefig("imgs/" + save_file_names[i])
    plt.show()

    # importance plot
    # imp_kwargs = [
    #     {"tar": "cs", "title": "Oxidation State"},
    #     {"tar": "cn", "title": "Coordination Number"},
    #     {"tar": "bl", "title": "Bond Length"},
    # ]
    # for kwarg in imp_kwargs:
    #     feature_importance_plot(results_database, **kwarg)
    # plt.show()

    ## Second Shell
    # data_path = "bl2nd_combined_data.pkl"
    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)
    # results_database = data["results"]

    # plot_kwargs = {
    #     "tar": "bl_2nd",
    #     "title": "Second Shell Mean Length - RF",
    #     "ylim": [0, 0.08],
    # }
    # results_plot(results_database, **plot_kwargs)

    # legends = [
    #     ax.legend() for ax in plt.gcf().axes if ax.get_legend() is not None
    # ]
    # plt.tight_layout()
    # pars = plt.gcf().subplotpars
    # plt.subplots_adjust(right=pars.right - 0.04, top=pars.top - 0.03)
    # plt.legend(bbox_to_anchor=(1.2, 1.23))
    # plt.show()
    # plt.savefig("imgs/second_shell_rf.png")

    ## Second Shell knn
    # data_path = "knn_bl2nd_combined_data"
    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)
    # results_database = data["results"]

    # plot_kwargs = {
    #     "tar": "bl_2nd",
    #     "title": "Second Shell Mean Length - kNN",
    #     "ylim": [0, 0.08],
    # }
    # results_plot(results_database, **plot_kwargs)

    # legends = [
    #     ax.legend() for ax in plt.gcf().axes if ax.get_legend() is not None
    # ]
    # plt.tight_layout()
    # pars = plt.gcf().subplotpars
    # plt.subplots_adjust(right=pars.right - 0.04, top=pars.top - 0.03)
    # plt.legend(bbox_to_anchor=(1.2, 1.23))
    # plt.savefig("imgs/second_shell_knn.png")

    # result plot knn
    # filename = "combined_data.pkl"
    # filename = "knn_combined_data"
    # with open(filename, "rb") as f:
    #     results_database = pickle.load(f)
    # results_database = results_database["results"]
    # save_file_names = [
    #     "result_cs_knn.png",
    #     "result_cn_knn.png",
    #     "result_bl_knn.png",
    # ]
    # run_kwargs = [
    #     {
    #         "tar": "cs",
    #         "title": "Oxidation State - kNN",
    #         "ylabel": "weighted mean F1 score",
    #         "ylim": [0, 1],
    #     },
    #     {
    #         "tar": "cn",
    #         "title": "Coordination Number - kNN",
    #         "ylabel": "weighted mean F1 score",
    #         "ylim": [0, 1],
    #     },
    #     {
    #         "tar": "bl",
    #         "title": "Bond Length - kNN",
    #         "ylabel": "RMSE (% mean BL)",
    #         "ylim": [0, 0.06],
    #     },
    # ]
    # for i, kwarg in enumerate(run_kwargs):
    #     results_plot(results_database, **kwarg)
    #     legends = [
    #         ax.legend() for ax in plt.gcf().axes if ax.get_legend() is not None
    #     ]
    #     plt.tight_layout()
    #     pars = plt.gcf().subplotpars
    #     plt.subplots_adjust(right=pars.right - 0.04, top=pars.top - 0.03)
    #     plt.legend(bbox_to_anchor=(1.2, 1.23))
    #     plt.savefig("imgs/" + save_file_names[i])
