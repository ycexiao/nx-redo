import numpy as np
from scipy.signal import find_peaks
import pickle
import random
from pathlib import Path

from helper import resultDatabase
from ml import convert_features_name

import matplotlib
from matplotlib import pyplot as plt
from bg_mpl_stylesheets.styles import all_styles
from bg_mpl_stylesheets.colors import Colors

plt.style.use(all_styles["bg-style"])
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

matplotlib.use("TkAgg")
# bg_blue,bg_red,bg_green,bg_light_blue,bg_light_grey,bg_yellow,bg_brown,bg_burgundy,bg_olive_green,bg_muted_olive,bg_beige,bg_grey
# colors = {Colors(hex).name: hex for hex in cycle}
# chosen_color_names = ["bg_light_blue", "bg_red", "bg_light_grey", "bg_yellow", "bg_olive_green", "bg_muted_olive", "bg_beige", "bg_grey"]
# random.shuffle(chosen_color_names)
# chosen_colors = [colors[name] for name in chosen_color_names]

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
targets = ["cs", "cn", "bl"]
elements = ["Ti", "Cu", "Fe", "Mn"]


def result_plot_one(
    ax, ys, yerrs, base_ys, base_yerrs, ylabels, base_ylabels, ylim=None
):
    ax.set_xlim([-1, len(ys)])
    xlim = ax.get_xlim()
    ax.bar(range(len(ys)), ys, yerr=yerrs, color=cycle[0])
    for i in range(len(ys)):
        ax.text(
            i,
            ys[i],
            "{:.3f}".format(ys[i]),
        )
    for i in range(len(base_ys)):
        ax.fill_between(
            np.linspace(*xlim, 100),
            np.ones(100) * (base_ys[i] - base_yerrs[i] / 2),
            np.ones(100) * (base_ys[i] + base_yerrs[i] / 2),
            alpha=0.6,
            color=cycle[i],
            label=base_ylabels[i],
        )
    ax.set_xticks(
        np.arange(0, len(ylabels) + 1),
        [None, *ylabels],
        rotation=45,
        ha="right",
    )
    # ax.set_xticklabels(ylabels, rotation=45, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    return ax


## results plot
def results_plot(
    results_database: resultDatabase,
    tar="cs",
    model_type="rf",
    title="Oxidation states",
    ylabel="F1 score",
    ylim=[0, 1],
    save_name=None,
):
    fig, axes = plt.subplots(1, len(elements), figsize=(11.2, 6.4))
    baseline_names = convert_features_name(baseline_features)
    features_names = convert_features_name(plot_features)
    target = tar
    for j in range(len(elements)):
        scores = []
        errors = []
        base_scores = []
        base_errors = []
        for k in range((len(plot_features))):
            data = results_database.filter_data(
                ["target", "element", "features", "model_type"],
                [target, elements[j], plot_features[k], model_type],
            ).value
            scores.append(np.mean(data["test_scores"]))
            errors.append(np.std(data["test_scores"]))
        for k in range(len(baseline_features)):
            data = results_database.filter_data(
                ["target", "element", "features", "model_type"],
                [target, elements[j], baseline_features[k], model_type],
            ).value
            base_scores.append(np.mean(data["test_scores"]))
            base_errors.append(np.std(data["test_scores"]))

        if ylim:
            result_plot_one(
                axes[j],
                scores,
                errors,
                base_scores,
                base_errors,
                features_names,
                baseline_names,
                ylim=ylim,
            )
        else:
            result_plot_one(
                axes[j],
                scores,
                errors,
                base_scores,
                base_errors,
                features_names,
                baseline_names,
            )

        axes[j].set_title(elements[j])
        if j == 0:
            axes[j].set_ylabel(ylabel)
        if j != 0:
            axes[j].set_yticks([])

    # setp
    fig.suptitle(title, fontsize=18)
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.2, 1.23))
    # legends = [
    #     ax.legend() for ax in axes if ax.get_legend() is not None
    # ]
    pars = plt.gcf().subplotpars
    plt.subplots_adjust(right=pars.right - 0.04, top=pars.top - 0.03)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)


importances_plot_features = [
    ["x_pdf", "n_pdf"],
    ["nx_pdf"],
    ["xanes"],
    ["xanes", "n_pdf"],
    ["xanes", "nx_pdf"],
    ["xanes", "x_pdf", "n_pdf"],
]
## feature importance plot
def feature_importance_plot(results_database, tar, title, model_type="rf"):
    feature_names = convert_features_name(importances_plot_features)
    baseline_names = convert_features_name(baseline_features)
    fig, axes = plt.subplots(
        len(importances_plot_features), len(elements), figsize=(12, 6)
    )
    for j in range(len(importances_plot_features)):
        for k in range(len(elements)):
            keys_dict = {
                "target": tar,
                "features": importances_plot_features[j],
                "element": elements[k],
            }
            data = results_database.filter_data(
                ["target", "features", "element"],
                [tar, importances_plot_features[j], elements[k]],
            ).value
            importances = data["importances"]
            axes[j, k].plot(
                range(len(importances[0])),
                *importances,
                alpha=0.2,
            )
            axes[j, k].set_xticks([])
            means = np.mean(importances, axis=0)
            for h in range(int(len(means) // 100)):
                axes[j, k].plot(
                    np.arange(h * 100, h * 100 + 100),
                    means[h * 100 : h * 100 + 100],
                )

            peaks, _ = find_peaks(means, height=0.02)
            ylim = axes[j, k].get_ylim()
            axes[j, k].set_xlim([0, 300])
            for ind in peaks:
                axes[j, k].vlines(
                    np.arange(len(importances[0]))[ind],
                    *ylim,
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
    # load results data
    print(str(Path().cwd()))
    filename = "results/trained_data.pkl"
    with open(filename, "rb") as f:
        results_database = pickle.load(f)
    keynames = ["target", "element", "features", "model_type"]
    results_database = resultDatabase(keynames).from_pkl(
        results_database, ignore_duplicate=True
    )
    print("Finished processing results_database.")

    # plot
    run_rf_kwargs = [
        {
            "tar": "cs",
            "title": "Oxidation State - Random Forest",
            "ylabel": "weighted mean F1 score",
            "ylim": [0, 1],
            "model_type": "rf",
            "save_name": "imgs/result_cs_rf.pdf",
        },
        {
            "tar": "cn",
            "title": "Coordination Number - Random Forest",
            "ylabel": "weighted mean F1 score",
            "ylim": [0, 1],
            "model_type": "rf",
            "save_name": "imgs/result_cn_rf.pdf",
        },
        {
            "tar": "bl",
            "title": "Bond Length - Random Forest",
            "ylabel": r"RMSE (\% mean BL)",
            "ylim": [0, 0.06],
            "model_type": "rf",
            "save_name": r"imgs/result_bl_rf.pdf",
        },
    ]
    for kwarg in run_rf_kwargs:
        results_plot(results_database, **kwarg)
    plt.show()

    run_knn_kwargs = [
        {
            "tar": "cs",
            "title": "Oxidation State - kNN",
            "ylabel": "weighted mean F1 score",
            "ylim": [0, 1],
            "model_type": "knn",
            "save_name": "imgs/result_cs_knn.pdf",
        },
        {
            "tar": "cn",
            "title": "Coordination Number - kNN",
            "ylabel": "weighted mean F1 score",
            "ylim": [0, 1],
            "model_type": "knn",
            "save_name": "imgs/result_cn_knn.pdf",
        },
        {
            "tar": "bl",
            "title": "Bond Length - kNN",
            "ylabel": r"RMSE (\% mean BL)",
            "ylim": [0, 0.06],
            "model_type": "knn",
            "save_name": "imgs/result_bl_knn.pdf",
        },
    ]
    for kwarg in run_knn_kwargs:
        results_plot(results_database, **kwarg)
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
