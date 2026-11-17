from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd

from data import (
    PreparedData,
    observable_map_for_variant,
)

MODEL_VARIANT = "int_pre_eff"
PARAMETER_TABLE = Path(
    "fit_outputs/2026-10-17_10-24-42_parallel_delay_8/fitted_parameters.csv"
)

#int, prec, final
plot_order = (
    # Row 1: intermediate cells
    "Ki-67 inter Tbet- Bcl6-",
    "Ki-67 inter Tbet+ / inter Bcl6-",
    "Ki-67 inter Tbet- Bcl6+",

    # Row 2: precursor cells
    "Ki-67+ Tbet- Bcl6-",
    "Ki-67+ Tbet+ / inter Bcl6-",
    "Ki-67+ Tbet- Bcl6+",

    # Row 3: differentiated cells
    "Ki-67- Tbet- Bcl6-",
    "Ki-67- Tbet+ / inter Bcl6-",
    "Ki-67- Tbet- Bcl6+",
)

# These two files are created by a previous ordinary plot_sim.py run.
CACHE_DIRECTORY = PARAMETER_TABLE.parent
BEST_FIT_FILE = CACHE_DIRECTORY / "plot_sim_best_fit.csv"
RIBBON_FILE = CACHE_DIRECTORY / "plot_sim_ribbon_summary.csv"
OUTPUT_FILE = CACHE_DIRECTORY / "plot_error_2.png"

# figure settings
FIGURE_SIZE = (9.5, 9.0)
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
MARKER_SIZE = 8
MODEL_LINEWIDTH = 3.0
RIBBON_ALPHA = 0.22


best_fit = pd.read_csv(BEST_FIT_FILE)
ribbon = pd.read_csv(RIBBON_FILE)

observable_map = observable_map_for_variant(MODEL_VARIANT)
data = PreparedData(model_variant=MODEL_VARIANT).build_fit_dataframe()
data = data[data["observable"].isin(observable_map)].copy()



PLOT_TITLES = {
    "Ki-67 inter Tbet- Bcl6-": "Pre-Th0",
    "Ki-67 inter Tbet+ / inter Bcl6-": "Pre-Th1",
    "Ki-67 inter Tbet- Bcl6+": "Pre-Tfh",
    "Ki-67+ Tbet- Bcl6-": "Pre-Th0 prol.",
    "Ki-67+ Tbet+ / inter Bcl6-": "Pre-Th1 prol.",
    "Ki-67+ Tbet- Bcl6+": "Pre-Tfh prol.",
    "Ki-67- Tbet- Bcl6-": "Th0",
    "Ki-67- Tbet+ / inter Bcl6-": "Th1",
    "Ki-67- Tbet- Bcl6+": "Tfh",
}

th0_color=(35/255,139/255,69/255)
th1_color=(165/255,15/255,21/255)
tfh_color=(8/255,81/255,156/255)

# Set the model-line color for each subplot here.
PLOT_COLORS = {
    "Ki-67 inter Tbet- Bcl6-": th0_color,
    "Ki-67 inter Tbet+ / inter Bcl6-": th1_color,
    "Ki-67 inter Tbet- Bcl6+": tfh_color,
    "Ki-67+ Tbet- Bcl6-": th0_color,
    "Ki-67+ Tbet+ / inter Bcl6-": th1_color,
    "Ki-67+ Tbet- Bcl6+": tfh_color,
    "Ki-67- Tbet- Bcl6-": th0_color,
    "Ki-67- Tbet+ / inter Bcl6-": th1_color,
    "Ki-67- Tbet- Bcl6+": tfh_color,
}

# Set the data-dot and error-bar color for each subplot here.
DATA_COLORS = {
    "Ki-67 inter Tbet- Bcl6-": th0_color,
    "Ki-67 inter Tbet+ / inter Bcl6-": th1_color,
    "Ki-67 inter Tbet- Bcl6+": tfh_color,
    "Ki-67+ Tbet- Bcl6-": th0_color,
    "Ki-67+ Tbet+ / inter Bcl6-": th1_color,
    "Ki-67+ Tbet- Bcl6+": tfh_color,
    "Ki-67- Tbet- Bcl6-": th0_color,
    "Ki-67- Tbet+ / inter Bcl6-": th1_color,
    "Ki-67- Tbet- Bcl6+": tfh_color,
}

observables = [name for name in plot_order if name in set(best_fit["observable"])]

fig, axes = plt.subplots(3, 3, figsize=FIGURE_SIZE, squeeze=False)

for index, (axis, observable) in enumerate(zip(axes.flat, observables)):
    best = best_fit[best_fit["observable"] == observable].sort_values("fit_day")
    summary = ribbon[ribbon["observable"] == observable].sort_values("fit_day")
    observed_data = data[data["observable"] == observable].sort_values("fit_day")
    color = PLOT_COLORS[observable]
    data_color = DATA_COLORS[observable]

    upper = summary["mean_prediction"] + summary["std_prediction"]
    positive_band = upper > 0.0
    lower = np.where(
        positive_band,
        np.maximum(
            summary["mean_prediction"] - summary["std_prediction"],
            np.finfo(float).tiny,
        ),
        np.nan,
    )
    upper = np.where(positive_band, upper, np.nan)

    axis.fill_between(
        summary["fit_day"], lower, upper, color=color, alpha=RIBBON_ALPHA
    )
    axis.plot(
        best["fit_day"],
        best["model_prediction"],
        color=color,
        linewidth=MODEL_LINEWIDTH,
        zorder=1,
    )
    axis.errorbar(
        observed_data["fit_day"],
        observed_data["data_value"],
        yerr=observed_data["data_sem"],
        fmt="o",
        color=data_color,
        ecolor=data_color,
        #elinewidth=1.0,
        #capsize=3,
        elinewidth=1.5,
        capsize=4,
        capthick=5.0,
        markersize=MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=1.5,
        barsabove=True,
        zorder=3,
    )

    title = PLOT_TITLES[observable]
    axis.set_title(title, fontsize=TITLE_FONTSIZE)

    if index % 3 == 0:
        axis.set_ylabel("Cell counts", fontsize=LABEL_FONTSIZE)
    else:
        axis.tick_params(axis="y", labelleft=False)
    axis.set_yscale("log")
    axis.set_ylim(1.0, 5e5)
    axis.set_yticks([1e1, 1e3, 1e5])
    axis.set_xticks([0, 2, 4, 6])
    if index % 3 == 0:
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=range(2, 10), numticks=100)
        )
        axis.tick_params(axis="y", which="minor", left=True, length=3, width=1)
    else:
        axis.tick_params(axis="y", which="minor", left=False)
    axis.tick_params(axis="both", labelsize=TICK_FONTSIZE)

for axis in axes.flat[len(observables):]:
    axis.axis("off")

fig.supxlabel("Time after immunization (days)", fontsize=LABEL_FONTSIZE)
fig.tight_layout(rect=(0, 0, 1, 1), h_pad=1.6)
fig.subplots_adjust(hspace=0.4)
fig.savefig(OUTPUT_FILE, dpi=300)
plt.show()
