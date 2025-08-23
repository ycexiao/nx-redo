import numpy as np
from scipy.signal import find_peaks
import pickle
import random
from pathlib import Path
import json

from helper import resultDatabase
from ml import convert_features_name

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from bg_mpl_stylesheets.styles import all_styles
from bg_mpl_stylesheets.colors import Colors
from cycler import cycler
from itertools import cycle

plt.style.use(all_styles["bg-style"])
cycle_color = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bg_colors = {Colors(hex).name: hex for hex in cycle_color}

matplotlib.use("TkAgg")
# bg_blue,bg_red,bg_green,bg_light_blue,bg_light_grey,bg_yellow,bg_brown,bg_burgundy,bg_olive_green,bg_muted_olive,bg_beige,bg_grey

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
result_plot_color_map = {
    "nxPDF": bg_colors["bg_blue"],
    "xPDF+nPDF": bg_colors["bg_light_blue"],
    "XANES+nxPDF": "#283618",
    "XANES+xPDF+nPDF": "#606c38",
    "nPDF": bg_colors["bg_burgundy"],
    "xPDF": bg_colors["bg_brown"],
    "XANES+nPDF": bg_colors["bg_burgundy"],
    "XANES+xPDF": bg_colors["bg_brown"],
    "XANES": bg_colors["bg_yellow"],
    "XANES+dPDF": bg_colors["bg_red"],
}


def bunch_filter_features(database, tar, ele, features, model_type):
    y = []
    yerr = []
    for i in range(len(features)):
        data = database.filter_data(
            ["target", "element", "features", "model_type"],
            [tar, ele, features[i], model_type],
        ).value
        y.append(np.mean(data["test_scores"]))
        yerr.append(np.std(data["test_scores"]))
    label = convert_features_name(features)
    return {"y": y, "yerr": yerr, "label": label}


def result_plot_one(
    ax,
    left_vertical_dict,
    left_horizontal_dict,
    right_vertical_dict,
    right_horizontal_dict,
    both_horizontal_dict,
):
    # artist position
    bar_width = 0.8
    bar_mid_distance = 1
    group_sep_ratio = 0.5
    group_sep_width = bar_mid_distance * (1 + group_sep_ratio)
    margin_ratio = 0.5
    margin_width = bar_mid_distance * margin_ratio

    start_l = 0
    left_ticks = [
        start_l + bar_mid_distance * i
        for i in range(len(left_vertical_dict["label"]))
    ]
    start_r = left_ticks[-1] + group_sep_width
    right_ticks = [
        start_r + bar_mid_distance * i
        for i in range(len(right_vertical_dict["label"]))
    ]
    left_vertical_dict["xticks"] = left_ticks
    right_vertical_dict["xticks"] = right_ticks
    left_horizontal_dict["xticks"] = left_ticks
    xlim = (
        left_ticks[0] - bar_width / 2 * 3,
        right_ticks[-1] + bar_width / 2 * 3,
    )
    ax.set_xlim(*xlim)
    left_horizontal_dict["xticks"] = [xlim[0], np.mean(xlim)]
    right_horizontal_dict["xticks"] = [np.mean(xlim), xlim[1]]
    both_horizontal_dict["xticks"] = xlim
    xticks = np.array([*left_ticks, *right_ticks])
    names = [*left_vertical_dict["label"], *right_vertical_dict["label"]]
    ax.set_xticks(xticks + 0.5, names, rotation=30, ha="right")

    for plot_dict in [left_vertical_dict, right_vertical_dict]:
        out = ax.bar(
            plot_dict["xticks"],
            plot_dict["y"],
            yerr=plot_dict["yerr"],
            width=bar_width,
            color=[
                result_plot_color_map[plot_dict["label"][i]]
                for i in range(len(plot_dict["label"]))
            ],
        )
        for i in range(len(plot_dict["y"])):
            ax.text(
                plot_dict["xticks"][i],
                plot_dict["y"][i] + plot_dict["yerr"][i] / 2,
                "{:.3f}".format(plot_dict["y"][i]),
                fontsize=14,
                ha="center",
                va="bottom",
            )

    for plot_dict in [
        left_horizontal_dict,
        right_horizontal_dict,
        both_horizontal_dict,
    ]:
        plot_dict["handles"] = []
        print(plot_dict["xticks"]),
        for i in range(len(plot_dict["y"])):
            out = ax.fill_between(
                np.linspace(
                    plot_dict["xticks"][0], plot_dict["xticks"][-1], 100
                ),
                np.ones(100) * (plot_dict["y"][i] - plot_dict["yerr"][i] / 2),
                np.ones(100) * (plot_dict["y"][i] + plot_dict["yerr"][i] / 2),
                label=plot_dict["label"][i],
                color=result_plot_color_map[plot_dict["label"][i]],
                alpha=0.7,
            )
            plot_dict["handles"].append(out)

    # left_legend = ax.legend(
    #     handles=left_horizontal_dict["handles"], loc="lower left"
    # )
    # right_legend = ax.legend(
    #     handles=right_horizontal_dict["handles"], loc="lower right"
    # )
    # ax.add_artist(left_legend)
    # ax.add_artist(right_legend)
    ax.set_ylim([0, 1])
    ax.set_xlim(
        xticks[0] - bar_width / 2 - margin_width,
        xticks[-1] + bar_width / 2 + margin_width,
    )
    ax.vlines(
        np.mean(xlim), [0], [1], linestyle="dashed", linewidth=2, color="k"
    )
    return ax, both_horizontal_dict["handles"], left_horizontal_dict["handles"]


def result_plot(
    results_database: resultDatabase,
    tar="cs",
    model_type="rf",
    title="Oxidation states",
    ylabel="F1 score",
    ylim=[0, 1],
    fname=None,
    save=False,
):
    fig, axes = plt.subplots(1, len(elements), figsize=(12, 6))
    fig.subplots_adjust(bottom=0.2, wspace=0.1)
    left_vertical_features = [["nx_pdf"], ["x_pdf", "n_pdf"]]
    right_vertical_features = [
        ["xanes", "nx_pdf"],
        ["xanes", "x_pdf", "n_pdf"],
    ]
    left_horizontal_features = [["x_pdf"], ["n_pdf"]]
    right_horizontal_features = [["xanes", "x_pdf"], ["xanes", "n_pdf"]]
    both_horizontal_features = [["xanes"], ["xanes", "diff_x_pdf"]]

    for i in range(len(elements)):
        features = [
            left_vertical_features,
            left_horizontal_features,
            right_vertical_features,
            right_horizontal_features,
            both_horizontal_features,
        ]
        features_dicts = [
            bunch_filter_features(
                results_database, tar, elements[i], fea, "rf"
            )
            for fea in features
        ]
        _, both_handles, left_handles = result_plot_one(
            axes[i], *features_dicts
        )
        axes[i].set_title(elements[i], fontsize=18)
        axes[i].tick_params(axis="x", length=0)
        axes[i].set_ylim(ylim)
        if i != 0:
            axes[i].set_yticks([])
        else:
            axes[i].set_ylabel(ylabel, fontsize=20)
        if i == 3:
            axes[i].legend(
                handles=[*both_handles, *left_handles],
                labels=["XANES", "XANES+dPDF", "(XANES+)xPDF", "(XANES+)nPDF"],
                loc="lower right",
                bbox_to_anchor=(0, 0, 1, 1),
            )
    fig.suptitle(title, fontsize=24, y=0.99)
    if fname and save:
        plt.savefig(fname)


importances_plot_features = [
    ["x_pdf", "n_pdf"],
    ["nx_pdf"],
    ["xanes"],
    ["xanes", "n_pdf"],
    ["xanes", "nx_pdf"],
    ["xanes", "x_pdf", "n_pdf"],
]


importance_plot_color_map = {
    "XANES": bg_colors["bg_muted_olive"],
    "nPDF": bg_colors["bg_blue"],
    "xPDF": bg_colors["bg_light_blue"],
    "nxPDF": bg_colors["bg_green"],
}


def importance_plot_one(ax, importances, plot_colors, ylim=None):
    ax.plot(
        range(len(importances[0])),
        *importances,
        alpha=0.6,
        color=bg_colors["bg_grey"],
    )
    means = np.mean(importances, axis=0)
    for h in range(int(len(means) // 100)):
        ax.plot(
            np.arange(h * 100, h * 100 + 100),
            means[h * 100 : h * 100 + 100],
            color=plot_colors[h],
        )
    peaks, _ = find_peaks(means, height=0.02)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()
    ax.set_xlim([0, 300])
    for ind in peaks:
        # ax.vlines(
        #     np.arange(len(importances[0]))[ind],
        #     *ylim,
        #     linestyle="--",
        #     linewidth=0.5,
        # )
        pass
    ax.set_yticks([])
    ax.set_xticks([0, 100, 200])


## feature importance plot
def importance_plot(
    results_database,
    tar,
    title,
    model_type="rf",
    ylim=None,
    fname=None,
    save=False,
):
    feature_names = convert_features_name(importances_plot_features)
    fig, axes = plt.subplots(
        len(importances_plot_features),
        len(elements),
        figsize=(12, 6),
        sharex=True,
        sharey=True,
    )
    for j in range(len(importances_plot_features)):
        for k in range(len(elements)):
            data = results_database.filter_data(
                ["target", "features", "element", "model_type"],
                [tar, importances_plot_features[j], elements[k], model_type],
            ).value
            plot_colors = [
                importance_plot_color_map[convert_features_name(fea)]
                for fea in importances_plot_features[j]
            ]
            importances = data["importances"]
            importance_plot_one(
                axes[j, k], importances, plot_colors, ylim=ylim
            )

            if j == 0:
                axes[j, k].set_title(elements[k], fontsize=20)
            if k == 0:
                axes[j, k].set_ylabel(
                    feature_names[j], rotation=45, ha="right", fontsize=16
                )

            axes[j, k].tick_params(top=False)
    fig.suptitle(title, fontsize=24)
    # plt.subplots_adjust(
    #     left=0.2, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.2
    # )
    custom_legend = [
        Line2D([0], [0], color=c, label=b)
        for b, c in importance_plot_color_map.items()
    ]
    fig.legend(
        handles=custom_legend,
        ncols=len(custom_legend),
        loc="lower center",
        # bbox_to_anchor=(0.7, 0.12),
    )
    plt.subplots_adjust(left=0.18, bottom=0.20, hspace=0.1, wspace=0.1)
    if fname and save:
        plt.savefig(fname)


if __name__ == "__main__":
    # load results data
    filename = "results/trained_data.pkl"
    with open(filename, "rb") as f:
        results_database = pickle.load(f)
    keynames = ["target", "element", "features", "model_type"]
    results_database = resultDatabase(keynames).from_pkl(
        results_database, ignore_duplicate=True
    )
    print("Finished processing results_database.")

    # load plot args
    with open("result_plot_with_rf_knn_kwargs.json", "r") as f:
        results_plot_rf_knn_kwargs = json.load(f)

    with open("result_plot_kwargs.json", "r") as f:
        results_plot_kwargs = json.load(f)

    with open("importance_plot_kwargs.json", "r") as f:
        importances_plot_kwargs = json.load(f)

    for kwargs in results_plot_kwargs:
        result_plot(results_database, **kwargs, save=True)
        pass

    for kwargs in importances_plot_kwargs:
        importance_plot(results_database, **kwargs, save=True)
        pass
    plt.show()
