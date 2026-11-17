from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import numpy as np

try:
    from .fit_helper import ModelFitter
except ImportError:
    from fit_helper import ModelFitter


Y_AXIS_SCALE = 1e5
Y_AXIS_LABEL = r"Cell Counts ($10^5$)"
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16

TH0_COLOR = (35 / 255, 139 / 255, 69 / 255)
TH1_COLOR = (165 / 255, 15 / 255, 21 / 255)
TFH_COLOR = (8 / 255, 81 / 255, 156 / 255)
DARK_GREY = "0.35"

PRECURSOR_PLOT_ROW = [
    "Ki-67+ Tbet- Bcl6-",
    "Ki-67+ Tbet+ / inter Bcl6-",
    "Ki-67+ Tbet- Bcl6+",
]

INTERMEDIATE_PLOT_ROW = [
    "Ki-67 inter Tbet- Bcl6-",
    "Ki-67 inter Tbet+ / inter Bcl6-",
    "Ki-67 inter Tbet- Bcl6+",
]

EFFECTOR_PLOT_ROW = [
    "Ki-67- Tbet- Bcl6-",
    "Ki-67- Tbet+ / inter Bcl6-",
    "Ki-67- Tbet- Bcl6+",
]

PRECURSOR_INTERMEDIATE_EFFECTOR_PLOT_ORDER = [
    *PRECURSOR_PLOT_ROW,
    *INTERMEDIATE_PLOT_ROW,
    *EFFECTOR_PLOT_ROW,
]

INTERMEDIATE_PRECURSOR_EFFECTOR_PLOT_ORDER = [
    *INTERMEDIATE_PLOT_ROW,
    *PRECURSOR_PLOT_ROW,
    *EFFECTOR_PLOT_ROW,
]

INTERMEDIATE_EFFECTOR_PRECURSOR_PLOT_ORDER = [
    *INTERMEDIATE_PLOT_ROW,
    *EFFECTOR_PLOT_ROW,
    *PRECURSOR_PLOT_ROW,
]

PRECURSOR_EFFECTOR_INTERMEDIATE_PLOT_ORDER = [
    *PRECURSOR_PLOT_ROW,
    *EFFECTOR_PLOT_ROW,
    *INTERMEDIATE_PLOT_ROW,
]

EFFECTOR_INTERMEDIATE_PRECURSOR_PLOT_ORDER = [
    *EFFECTOR_PLOT_ROW,
    *INTERMEDIATE_PLOT_ROW,
    *PRECURSOR_PLOT_ROW,
]

EFFECTOR_PRECURSOR_INTERMEDIATE_PLOT_ORDER = [
    *EFFECTOR_PLOT_ROW,
    *PRECURSOR_PLOT_ROW,
    *INTERMEDIATE_PLOT_ROW,
]

PLOT_LABELS = {
    "Ki-67+ Tbet- Bcl6-": "Pre-Th0 prol.",
    "Ki-67 inter Tbet- Bcl6-": "Pre-Th0",
    "Ki-67- Tbet- Bcl6-": "Th0",
    "Ki-67+ Tbet+ / inter Bcl6-": "Pre-Th1 prol.",
    "Ki-67 inter Tbet+ / inter Bcl6-": "Pre-Th1",
    "Ki-67- Tbet+ / inter Bcl6-": "Th1",
    "Ki-67+ Tbet- Bcl6+": "Pre-Tfh prol.",
    "Ki-67 inter Tbet- Bcl6+": "Pre-Tfh",
    "Ki-67- Tbet- Bcl6+": "Tfh",
}

PLOT_COLORS = {
    "Ki-67+ Tbet- Bcl6-": TH0_COLOR,
    "Ki-67 inter Tbet- Bcl6-": TH0_COLOR,
    "Ki-67- Tbet- Bcl6-": TH0_COLOR,
    "Ki-67+ Tbet+ / inter Bcl6-": TH1_COLOR,
    "Ki-67 inter Tbet+ / inter Bcl6-": TH1_COLOR,
    "Ki-67- Tbet+ / inter Bcl6-": TH1_COLOR,
    "Ki-67+ Tbet- Bcl6+": TFH_COLOR,
    "Ki-67 inter Tbet- Bcl6+": TFH_COLOR,
    "Ki-67- Tbet- Bcl6+": TFH_COLOR,
}

PLOT_ORDERS = {
    "int_pre_eff": INTERMEDIATE_PRECURSOR_EFFECTOR_PLOT_ORDER,
    "pre_int_eff": PRECURSOR_INTERMEDIATE_EFFECTOR_PLOT_ORDER,
    "int_eff_pre": INTERMEDIATE_EFFECTOR_PRECURSOR_PLOT_ORDER,
    "pre_eff_int": PRECURSOR_EFFECTOR_INTERMEDIATE_PLOT_ORDER,
    "eff_int_pre": EFFECTOR_INTERMEDIATE_PRECURSOR_PLOT_ORDER,
    "eff_pre_int": EFFECTOR_PRECURSOR_INTERMEDIATE_PLOT_ORDER,
}


def scaled_tick_labels(ticks: Sequence[float], scale: float) -> list[str]:
    return [f"{tick / scale:g}" for tick in ticks]


def plot_order_for_variant(model_variant: str) -> list[str]:
    try:
        return list(PLOT_ORDERS[model_variant])
    except KeyError as exc:
        known_variants = ", ".join(sorted(PLOT_ORDERS))
        raise ValueError(
            f"Unknown model variant '{model_variant}'. "
            f"Known variants: {known_variants}"
        ) from exc


def plot_fit_diagnostics(
    fitter: ModelFitter,
    params: Mapping[str, float] | Any,
    output_path: str | Path,
    num_timepoints: int = 250,
) -> Path:
    fit_frame = fitter.prediction_frame(params).copy()
    present = set(fit_frame["observable"])
    plot_order = plot_order_for_variant(fitter.config.model_variant)
    observables = [name for name in plot_order if name in present]

    dense_times = np.linspace(
        0.0,
        float(fit_frame["fit_day"].max()),
        num=max(2, num_timepoints),
    )
    simulated = fitter.model.get_observables(
        t_eval=dense_times,
        parameter_values=params,
        observable_map=fitter.observable_map,
        initial_state=fitter.config.initial_state,
        method=fitter.config.simulation_method,
    ).melt(
        id_vars="time",
        var_name="observable",
        value_name="model_prediction",
    ).rename(columns={"time": "fit_day"})

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(9.5, 9.0),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for index, (axis, observable) in enumerate(zip(axes_flat, observables)):
        observable_simulation = simulated[
            simulated["observable"] == observable
        ].sort_values("fit_day")
        observable_data = fit_frame[
            fit_frame["observable"] == observable
        ].sort_values("fit_day")
        color = PLOT_COLORS[observable]

        axis.plot(
            observable_simulation["fit_day"],
            observable_simulation["model_prediction"],
            color=color,
            linewidth=3.0,
            zorder=1,
        )
        axis.errorbar(
            observable_data["fit_day"],
            observable_data["data_value"],
            yerr=observable_data["data_sem"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            barsabove=True,
            zorder=3,
        )
        axis.set_title(PLOT_LABELS[observable], fontsize=20)
        axis.set_yscale("log")
        axis.set_ylim(1.0, 5e5)
        axis.set_yticks([1e1, 1e3, 1e5])
        axis.set_xticks([0, 2, 4, 6])

        if index % 3 == 0:
            axis.set_ylabel("Cell counts", fontsize=18)
            axis.yaxis.set_minor_locator(
                LogLocator(base=10, subs=range(2, 10), numticks=100)
            )
            axis.tick_params(axis="y", which="minor", left=True, length=3, width=1)
        else:
            axis.tick_params(axis="y", labelleft=False)
            axis.tick_params(axis="y", which="minor", left=False)
        axis.tick_params(axis="both", labelsize=18)
        axis.grid(False)

    for axis in axes_flat[len(observables) :]:
        axis.axis("off")

    data_legend_handle = axes_flat[0].errorbar(
        [np.nan],
        [np.nan],
        yerr=[1.0],
        fmt="o",
        color=DARK_GREY,
        ecolor=DARK_GREY,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="data",
    )
    legend_handles = [
        data_legend_handle,
        Line2D(
            [0],
            [0],
            color=DARK_GREY,
            linewidth=3.0,
            label="model fit",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=18,
    )
    figure.supxlabel(
        "Time after immunization (days)",
        fontsize=18,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    figure.subplots_adjust(hspace=0.4)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path
