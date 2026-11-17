from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

CHI2_LABEL = r"$\chi^2$ ($\times 10^{9}$)"

MODEL_AIC_VALUES = {
    "(i)": 739.85,
    "(ii)": 759.96,
    "(iii)": 761.58, #tbc
    "(iv)": 782.35, #tbc
    "(v)": 801.58, #model combi_5
    "(vi)": 828.28, #model combi_4
}


FONTSIZE = 16

def plot_model_selection(output_path: str | Path) -> Path:
    model_names = list(MODEL_AIC_VALUES.keys())
    aic_values = list(MODEL_AIC_VALUES.values())
    reference_aic = MODEL_AIC_VALUES["(i)"]
    plotted_models = [
        (model_name, aic_value - reference_aic)
        for model_name, aic_value in zip(model_names, aic_values, strict=True)
    ]
    plotted_names = [model_name for model_name, _ in plotted_models]
    delta_aic_values = [delta_aic for _, delta_aic in plotted_models]

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.barh(plotted_names, delta_aic_values, height=0.6, color="grey")
    ax.set_xlabel(r"$\Delta$ AIC", fontsize=FONTSIZE)
    ax.set_ylabel("Model", fontsize=FONTSIZE)


    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONTSIZE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return output_path



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AIC differences relative to model (i).")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "model_selection.png",
        help="Path to save the output plot.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = plot_model_selection(args.output_path)
    #table_output_path = plot_model_selection_table(args.table_output_path)
    #print(f"Saved model selection plot to: {output_path}")
    #print(f"Saved model selection table to: {table_output_path}")
    plt.show()


if __name__ == "__main__":
    main()
