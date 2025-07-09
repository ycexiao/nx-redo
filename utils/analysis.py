import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def filter_data(data, conditions):
    """
    Filter data entries use conditions.
    This function is used to extract information from a bunch of traning results.

    Parameters:
    ----------
    data: dict
        dict of training results
    condition: list
        a list of functions.
        e.g. [fun1, fun2].
        data with fun1(data)==True and fun2(data)=True
    """
    masks = np.ones(len(data), dtype=bool)
    outcome_exists = np.any(masks)

    for i in range(len(conditions)):
        mask = np.array(
            list(map(lambda x: conditions[i](x), data)), dtype="bool"
        )
        masks *= mask
        if not np.any(masks):
            raise KeyError(
                f"Unable to filter the outcome since the {i} condition."
            )

    inds = np.arange(len(data))[masks]
    return [data[i] for i in inds]


def generate_condition(key, value, mode="str"):
    """
    Parameters:
    ----------
    key: str
        specify items in the results dictionary.
    value: str
        the value to be matched
    mode: 'str' or other
        default 'str'. Return True when set(x(key)) == set(value).
    """
    if mode == "str":
        if isinstance(value, list):
            return lambda x: set(x[key]) == set(value)
        else:
            return lambda x: x[key] == value
    else:
        return lambda x: len(x[key]) == value


def generate_batch_conditions(*values):
    """
    Helper functions to generate search conditions.
    Specially tuned input to fit in this project

    Parameters:
    -----------
    """
    maps = ["features", "target", "element"]
    return [generate_condition(maps[i], values[i]) for i in range(len(values))]


def lookup(Data, key, feature, target, element):
    """
    Helper functions to look up scores.
    """

    conditions = generate_batch_conditions(feature, target, element)

    tmp_data = filter_data(Data, conditions)[0]

    if key == "scores":
        mean = tmp_data["test_scores"].mean()
        sd = tmp_data["test_scores"].std()
        return [mean, sd]
    else:
        return tmp_data[key]


class SmartPloter:
    def __init__(self, length):
        self.length = length
        self.metrics = np.empty(length)
        self.errors = np.empty(length)
        self.counter = 0

    def add_with_error(self, *arr):
        if len(arr[0]) != self.length:
            raise ValueError("Added array is of different length")

        if self.counter != 0:
            self.metrics = np.vstack([self.metrics, arr[0]])
            self.errors = np.vstack([self.errors, arr[1]])
            self.counter += 1
        else:
            self.metrics = arr[0]
            self.errors = arr[1]
            self.counter += 1

    def bar(self, ax=None, **plot_params):
        handles = []
        if ax is None:
            fig, ax = plt.subplots()
        metrics = self.metrics
        errors = self.errors
        x0 = np.arange(self.length)
        if self.counter == 1:
            ax.bar(
                x=x0, height=metrics, width=2 / 3, yerr=errors, **plot_params
            )
        else:
            width = 1 / (self.counter + 1)
            for i in range(self.counter):
                x = x0 + i * width
                h = ax.bar(
                    x=x,
                    height=metrics[i],
                    width=width,
                    yerr=errors[i],
                    **plot_params,
                )
                handles.append(h)
                for j in range(len(x)):
                    ax.text(
                        x[j],
                        metrics[i][j],
                        "{:.3f}".format(metrics[i][j]),
                        horizontalalignment="center",
                    )

        return ax, handles


def bunch_bar_plot(
    data,
    key,
    first_level_label,
    second_level_labels,
    third_level_labels,
    fig=None,
    axes=None,
):
    """
    Draw a fig with multiple bar plots.

    Parameters
    ----------
    data: list
        a list of dictionaries. Each dictionaries have keys that can used to filter the wanted data.
    first_level_label: str
        the first filter condition, and will be used as super title.
    seond_level_labels: list
        a list of string. The second filter condition, and will be used as title for each subfigure.
    third_level_labels: list
        a list of string. The third filter condition, and will be used as the xlabels in each subfigure.
    """
    if fig is None or axes is None:
        print("Start a new draw")
        fig, axes = plt.subplots(1, len(second_level_labels))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    bars = []
    for j, scd in enumerate(second_level_labels):
        means = []
        stds = []
        for k, thd in enumerate(third_level_labels):
            scores = lookup(
                data, key, target=first_level_label, element=scd, feature=thd
            )
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        bar = axes[j].bar(range(len(third_level_labels)), means, yerr=stds)
        bars.extend(bar)
        for h in range(len(means)):
            axes[j].text(
                range(len(third_level_labels))[h],
                means[h],
                "{:.3f}".format(means[h]),
            )

    return fig, axes, bars


def feature_importances_plot(
    data,
    key,
    first_level_label,
    second_level_labels,
    third_level_labels,
    fig=None,
    axes=None,
):
    """
    Draw a fig with multiple line plots.

    Parameters
    ----------
    data: list
    first_level_label: str
    second_level_labels: list
    third_level_labels: list
    """
    if fig is None or axes is None:
        print("Start a new draw")
        fig, axes = plt.subplots(len(third_level_labels), 1)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, thd in enumerate(third_level_labels):
        values = lookup(
            data,
            key,
            target=first_level_label,
            element=second_level_labels,
            feature=thd,
        )
        means = np.mean(values, axis=0)
        axes[k].plot(range(len(values[0])), *values, color="gray", alpha=0.2)

        for h in range(int(len(means) // 100)):
            axes[k].plot(
                np.arange(h * 100, h * 100 + 100),
                means[h * 100 : h * 100 + 100],
                color=colors[1 + h],
            )

        peaks, _ = find_peaks(means, height=0.02)
        ylim = axes[k].get_ylim()
        axes[k].set_xlim([0, 300])
        for ind in peaks:
            axes[k].vlines(
                np.arange(len(values[0]))[ind],
                *ylim,
                color="k",
                linestyle="--",
                linewidth=0.5,
            )

    return fig, axes


def bunch_hline_plot(
    data,
    key,
    first_level_label,
    second_level_labels,
    third_level_labels,
    fill=False,
    fig=None,
    axes=None,
):
    """
    Draw a fig with multiple hlines.

    Parameters
    ----------
    data: list
        a list of dictionaries. Each dictionaries have keys that can used to filter the wanted data.
    first_level_label: str
        the first filter condition, and will be used as super title.
    second_level_labels: list
        a list of string. The second filter condition, and will be used as title for each subfigure.
    third_level_labels: list
        a list of string. The third filter condition, and will be used as the xlabels in each subfigure.
    """
    if fig is None or axes is None:
        raise AttributeError("axes not exist. Cannot get xlim from axes.")

    hls = []
    for j, scd in enumerate(second_level_labels):
        xlim = axes[j].get_xlim()
        for k, thd in enumerate(third_level_labels):
            scores = lookup(
                data, key, target=first_level_label, element=scd, feature=thd
            )
            mean = np.mean(scores)
            std = np.std(scores)
            if fill:
                axes[j].fill_between(
                    np.linspace(*xlim, 100),
                    (mean - std / 2) * np.one(100),
                    (mean + std / 2) * np.ones(100),
                )
            else:
                hl = axes[j].hlines(mean, *xlim)
                hls.append(hl)

    return fig, axes, hls


def set_ax_style(
    ax,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    title=None,
):
    def safe_set(method, value, **kwargs):
        if value is not None:
            method(value, **kwargs)

    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)) + 0.5)
        ax.set_xticklabels(xticks)
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10, ha="right")
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
        plt.setp(ax.xaxis.label, fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        plt.setp(ax.yaxis.label, fontsize=14)
    if title is not None:
        ax.set_title(title)
        plt.setp(ax.title, fontsize=16)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
