from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data import MODEL_VARIANTS
    from .diagnostics import plot_fit_diagnostics
    from .fit_helper import FitConfig, ModelFitter, run_multistart_fit
except ImportError:
    from data import MODEL_VARIANTS
    from diagnostics import plot_fit_diagnostics
    from fit_helper import FitConfig, ModelFitter, run_multistart_fit


def run_fit(
    output_dir: str | Path | None = None,
    *,
    model_variant: str = "int_pre_eff",
    fit_method: str = "leastsq",
    num_timepoints: int = 250,
    multi_start: int | None = None,
    random_seed: int | None = None,
) -> dict[str, Path]:
    if multi_start is not None and multi_start < 1:
        raise ValueError(f"multi_start must be at least 1, got {multi_start}.")

    fitter = ModelFitter(config=FitConfig(model_variant=model_variant))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "fit_outputs"
    )
    output_suffix = (
        f"{model_variant}_multistart{multi_start}"
        if multi_start is not None
        else model_variant
    )
    fit_output_dir = output_root / f"{timestamp}_{output_suffix}"
    fit_output_dir.mkdir(parents=True, exist_ok=True)

    if multi_start is None:
        initial_parameters = fitter.parameter_builder.make_lmfit_parameters()
        initial_values = {
            name: float(value)
            for name, value in initial_parameters.valuesdict().items()
        }
        result = fitter.fit(params=initial_parameters, method=fit_method)
    else:
        multistart = run_multistart_fit(
            fitter,
            multi_start,
            fit_method=fit_method,
            random_seed=random_seed,
        )
        best_start = multistart.best_start
        result = best_start.result
        initial_values = best_start.initial_values

    fit_statistics: list[tuple[str, str | int | float]] = [
        ("fitting_method", str(getattr(result, "method", fit_method))),
        ("function_evaluations", int(getattr(result, "nfev", 0))),
        ("data_points", int(getattr(result, "ndata", 0))),
        ("variables", int(getattr(result, "nvarys", 0))),
        ("chi_square", float(getattr(result, "chisqr", np.nan))),
        ("AIC", float(getattr(result, "aic", np.nan))),
        ("BIC", float(getattr(result, "bic", np.nan))),
    ]
    summary_rows = [
        {
            "category": "fit_statistic",
            "name": name,
            "value": value,
            "remark": "",
        }
        for name, value in fit_statistics
    ]
    for parameter_name in sorted(result.params):
        parameter = result.params[parameter_name]
        remark = (
            f"initial={initial_values[parameter_name]:.12g}"
            if parameter.vary
            else "fixed"
        )
        summary_rows.append(
            {
                "category": "parameter",
                "name": parameter_name,
                "value": float(parameter.value),
                "remark": remark,
            }
        )

    summary_csv_path = fit_output_dir / "fit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    plot_path = plot_fit_diagnostics(
        fitter,
        result.params,
        fit_output_dir / "fit.png",
        num_timepoints=num_timepoints,
    )

    return {
        "output_dir": fit_output_dir,
        "summary_csv": summary_csv_path,
        "plot": plot_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the paper ODE models to three-condition absolute-count "
            "summary data."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fit_outputs",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--model-variant",
        default="int_pre_eff",
        choices=sorted(MODEL_VARIANTS),
        help="ODE model to use for fitting.",
    )
    parser.add_argument(
        "--method",
        default="leastsq",
        help="lmfit minimization method passed to lmfit.minimize().",
    )
    parser.add_argument(
        "--num-timepoints",
        type=int,
        default=250,
        help="Number of time points in the smooth diagnostic curves.",
    )
    parser.add_argument(
        "--multi-start",
        type=int,
        default=None,
        help=(
            "Number of local fits with different initial guesses. "
            "Start 1 uses the parameter defaults."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for multistart initialization.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        outputs = run_fit(
            output_dir=args.output_dir,
            model_variant=args.model_variant,
            fit_method=args.method,
            num_timepoints=args.num_timepoints,
            multi_start=args.multi_start,
            random_seed=args.seed,
        )
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\nInstall the missing package and re-run `python fit.py`."
        ) from exc

    print(f"Fit completed. Outputs saved in: {outputs['output_dir']}")
    print(f"CSV fit summary: {outputs['summary_csv']}")
    print(f"Fit plot: {outputs['plot']}")


if __name__ == "__main__":
    main()
