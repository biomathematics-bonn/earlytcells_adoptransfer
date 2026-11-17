from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

try:
    from .data import MODEL_VARIANTS, PreparedData, observable_map_for_variant
    from .diagnostics import (
        AXIS_LABEL_FONTSIZE,
        EFFECTOR_INTERMEDIATE_PRECURSOR_PLOT_ORDER,
        EFFECTOR_PRECURSOR_INTERMEDIATE_PLOT_ORDER,
        INTERMEDIATE_EFFECTOR_PRECURSOR_PLOT_ORDER,
        INTERMEDIATE_PRECURSOR_EFFECTOR_PLOT_ORDER,
        LEGEND_FONTSIZE,
        PLOT_LABELS,
        PRECURSOR_EFFECTOR_INTERMEDIATE_PLOT_ORDER,
        PRECURSOR_INTERMEDIATE_EFFECTOR_PLOT_ORDER,
        TICK_FONTSIZE,
        TITLE_FONTSIZE,
        Y_AXIS_LABEL,
        Y_AXIS_SCALE,
        scaled_tick_labels,
    )
    from .model import SubpopModel
    from .parameters import ModelParameters, ParameterSpec
except ImportError:
    from data import MODEL_VARIANTS, PreparedData, observable_map_for_variant
    from diagnostics import (
        AXIS_LABEL_FONTSIZE,
        EFFECTOR_INTERMEDIATE_PRECURSOR_PLOT_ORDER,
        EFFECTOR_PRECURSOR_INTERMEDIATE_PLOT_ORDER,
        INTERMEDIATE_EFFECTOR_PRECURSOR_PLOT_ORDER,
        INTERMEDIATE_PRECURSOR_EFFECTOR_PLOT_ORDER,
        LEGEND_FONTSIZE,
        PLOT_LABELS,
        PRECURSOR_EFFECTOR_INTERMEDIATE_PLOT_ORDER,
        PRECURSOR_INTERMEDIATE_EFFECTOR_PLOT_ORDER,
        TICK_FONTSIZE,
        TITLE_FONTSIZE,
        Y_AXIS_LABEL,
        Y_AXIS_SCALE,
        scaled_tick_labels,
    )
    from model import SubpopModel
    from parameters import ModelParameters, ParameterSpec


def load_parameter_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)

    legacy_columns = {"parameter", "fit_value"}
    if legacy_columns.issubset(frame.columns):
        return frame.copy()

    summary_columns = {"category", "name", "value", "remark"}
    if summary_columns.issubset(frame.columns):
        parameters = frame[frame["category"] == "parameter"].copy()
        if parameters.empty:
            raise ValueError(f"No parameter rows found in {path}.")
        parameters = parameters.rename(
            columns={"name": "parameter", "value": "fit_value"}
        )
        parameters["fit_value"] = pd.to_numeric(
            parameters["fit_value"], errors="raise"
        )
        return parameters.drop(columns=["category"])

    raise ValueError(
        "Parameter CSV must be either a fitted-parameter table containing "
        "'parameter' and 'fit_value', or a fit summary containing "
        "'category', 'name', 'value', and 'remark'. "
        f"Got columns: {list(frame.columns)}."
    )


def recompute_expressions(
    values: Mapping[str, float],
    specs: Mapping[str, ParameterSpec],
) -> dict[str, float]:
    resolved = dict(values)
    for name, spec in specs.items():
        if spec.expr is not None:
            resolved[name] = float(eval(spec.expr, {"__builtins__": {}}, resolved))
    return resolved


def build_best_fit_parameters(
    parameter_frame: pd.DataFrame,
    model_variant: str,
) -> tuple[dict[str, float], dict[str, ParameterSpec]]:
    builder = ModelParameters(model_variant=model_variant)
    values = builder.defaults_dict()
    values.update(
        zip(
            parameter_frame["parameter"],
            parameter_frame["fit_value"].astype(float),
        )
    )
    return recompute_expressions(values, builder.specs), builder.specs


def _ci_width(parameter_row: pd.Series) -> tuple[float, float] | None:
    ci_low = parameter_row.get("CI_low")
    ci_up = parameter_row.get("CI_up")
    if pd.isna(ci_low) or pd.isna(ci_up):
        return None
    ci_low = float(ci_low)
    ci_up = float(ci_up)
    if not np.isfinite(ci_low) or not np.isfinite(ci_up) or ci_low >= ci_up:
        return None
    return ci_low, ci_up


def _sample_transformed(
    *,
    best: float,
    spec: ParameterSpec,
    parameter_name: str,
    size: int,
    rng: np.random.Generator,
    sampling_method: str,
    cv: float,
) -> np.ndarray:
    eps = 1e-9
    z_90 = 1.6448536269514722
    parameter_min = spec.min
    parameter_max = spec.max

    if parameter_name.startswith("p_"):
        center = min(max(best, eps), 1.0 - eps)
        if sampling_method == "cv":
            sigma = cv / max(1.0 - center, eps)
        else:
            raise RuntimeError("CI bounds must be supplied for CI sampling.")
        return np.asarray(expit(rng.normal(logit(center), sigma, size=size)), dtype=float)

    if (
        parameter_min is not None
        and parameter_max is not None
        and parameter_min < parameter_max
    ):
        scaled = (best - parameter_min) / (parameter_max - parameter_min)
        center = min(max(scaled, eps), 1.0 - eps)
        if sampling_method == "cv":
            original_sd = cv * max(abs(best), eps)
            sigma = original_sd / max(
                (parameter_max - parameter_min) * center * (1.0 - center),
                eps,
            )
        else:
            raise RuntimeError("CI bounds must be supplied for CI sampling.")
        return parameter_min + (parameter_max - parameter_min) * np.asarray(
            expit(rng.normal(logit(center), sigma, size=size)),
            dtype=float,
        )

    if best > 0.0 and (parameter_min is None or parameter_min >= 0.0):
        sigma = float(np.sqrt(np.log1p(cv**2)))
        mu = np.log(best) - 0.5 * sigma**2
        samples = rng.lognormal(mu, sigma, size=size)
    else:
        samples = rng.normal(best, cv * max(abs(best), eps), size=size)

    if parameter_min is not None:
        samples = np.maximum(samples, parameter_min)
    if parameter_max is not None:
        samples = np.minimum(samples, parameter_max)
    return np.asarray(samples, dtype=float)


def sample_parameter(
    *,
    parameter_name: str,
    parameter_row: pd.Series,
    spec: ParameterSpec,
    size: int,
    rng: np.random.Generator,
    sampling_method: str,
    cv: float,
) -> np.ndarray:
    best = float(parameter_row["fit_value"])
    bounds = _ci_width(parameter_row)
    if sampling_method in {"ci", "ci_clipped"} and bounds is None:
        return np.full(size, best)

    if sampling_method == "cv":
        return _sample_transformed(
            best=best,
            spec=spec,
            parameter_name=parameter_name,
            size=size,
            rng=rng,
            sampling_method=sampling_method,
            cv=cv,
        )

    assert bounds is not None
    ci_low, ci_up = bounds
    eps = 1e-9
    z_90 = 1.6448536269514722

    def draw(draw_size: int) -> np.ndarray:
        if parameter_name.startswith("p_"):
            low = min(max(ci_low, eps), 1.0 - eps)
            up = min(max(ci_up, eps), 1.0 - eps)
            sigma = max((logit(up) - logit(low)) / (2.0 * z_90), 0.0)
            center = min(max(best, eps), 1.0 - eps)
            return np.asarray(expit(rng.normal(logit(center), sigma, size=draw_size)))

        if (
            spec.min is not None
            and spec.max is not None
            and spec.min < spec.max
        ):
            width = spec.max - spec.min
            low = min(max((ci_low - spec.min) / width, eps), 1.0 - eps)
            up = min(max((ci_up - spec.min) / width, eps), 1.0 - eps)
            center = min(max((best - spec.min) / width, eps), 1.0 - eps)
            sigma = max((logit(up) - logit(low)) / (2.0 * z_90), 0.0)
            return spec.min + width * np.asarray(
                expit(rng.normal(logit(center), sigma, size=draw_size))
            )

        if best > 0.0 and ci_low > 0.0 and ci_up > 0.0:
            sigma = max((np.log(ci_up) - np.log(ci_low)) / (2.0 * z_90), 0.0)
            mu = np.log(best) - 0.5 * sigma**2
            samples = rng.lognormal(mu, sigma, size=draw_size)
        else:
            sigma = max((ci_up - ci_low) / (2.0 * z_90), 0.0)
            samples = rng.normal(best, sigma, size=draw_size)

        if spec.min is not None:
            samples = np.maximum(samples, spec.min)
        if spec.max is not None:
            samples = np.minimum(samples, spec.max)
        return np.asarray(samples, dtype=float)

    if sampling_method == "ci":
        return draw(size)

    accepted: list[np.ndarray] = []
    accepted_count = 0
    for _ in range(100):
        if accepted_count >= size:
            break
        candidates = draw(size - accepted_count)
        valid = candidates[(candidates >= ci_low) & (candidates <= ci_up)]
        if valid.size:
            accepted.append(valid)
            accepted_count += valid.size
    if accepted_count < size:
        raise RuntimeError(
            f"Could not generate {size} in-CI samples for '{parameter_name}'."
        )
    return np.concatenate(accepted)[:size]


def plot_simulation_ensemble(
    *,
    params_csv: str | Path,
    model_variant: str = "int_pre_eff",
    num_samples: int = 100,
    sampling_method: str = "cv",
    cv: float = 0.01,
    num_timepoints: int = 250,
    seed: int | None = 1,
    output_path: str | Path | None = None,
    log_y: bool = False,
    paper_style: bool = False,
    reuse_cache: bool = False,
    cache_output_path: str | Path | None = None,
) -> Path:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            "plot_sim.py supports int_pre_eff, pre_int_eff, int_eff_pre, "
            "pre_eff_int, eff_int_pre, and eff_pre_int."
        )
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if cv < 0.0:
        raise ValueError("cv must be non-negative.")

    params_csv = Path(params_csv)
    output = (
        Path(output_path)
        if output_path is not None
        else params_csv.resolve().parent / "plot_sim.png"
    )
    cache_anchor = Path(cache_output_path) if cache_output_path is not None else output
    best_fit_cache = cache_anchor.with_name(f"{cache_anchor.stem}_best_fit.csv")
    ribbon_cache = cache_anchor.with_name(f"{cache_anchor.stem}_ribbon_summary.csv")
    observable_map = observable_map_for_variant(model_variant)
    data = PreparedData(model_variant=model_variant).build_fit_dataframe()
    data = data[data["observable"].isin(observable_map)].copy()
    if reuse_cache:
        if not best_fit_cache.exists() or not ribbon_cache.exists():
            raise FileNotFoundError(
                "Cached plot data were not found. Run plot_sim.py once without "
                "--reuse-cache first. Expected:\n"
                f"  {best_fit_cache}\n  {ribbon_cache}"
            )
        best_simulation = pd.read_csv(best_fit_cache)
        summary = pd.read_csv(ribbon_cache)
        required_best = {"fit_day", "observable", "model_prediction"}
        required_summary = {
            "fit_day", "observable", "mean_prediction", "std_prediction"
        }
        if not required_best.issubset(best_simulation.columns) or not required_summary.issubset(summary.columns):
            raise ValueError("Cached plot data have unexpected columns.")
        simulated_count = 0
        failed_samples = 0
    else:
        parameter_frame = load_parameter_frame(params_csv)
        best_fit, specs = build_best_fit_parameters(parameter_frame, model_variant)
        parameter_index = parameter_frame.set_index("parameter")
        free_names = [
            name
            for name, spec in specs.items()
            if (
                spec.vary
                and spec.expr is None
                and not name.startswith("alpha")
                and name in parameter_index.index
            )
        ]
        rng = np.random.default_rng(seed)
        samples = {
            name: sample_parameter(
                parameter_name=name,
                parameter_row=parameter_index.loc[name],
                spec=specs[name],
                size=num_samples,
                rng=rng,
                sampling_method=sampling_method,
                cv=cv,
            )
            for name in free_names
        }
        times = np.linspace(0.0, float(data["fit_day"].max()), max(2, num_timepoints))
        model = SubpopModel(model_variant=model_variant, observable_map=observable_map)
        best_simulation = model.get_observables(
            times, parameter_values=best_fit, observable_map=observable_map
        ).melt(
            id_vars="time", var_name="observable", value_name="model_prediction"
        ).rename(columns={"time": "fit_day"})

        simulated_samples: list[pd.DataFrame] = []
        failed_samples = 0
        for sample_index in range(num_samples):
            sampled_parameters = dict(best_fit)
            for name in free_names:
                sampled_parameters[name] = float(samples[name][sample_index])
            sampled_parameters = recompute_expressions(sampled_parameters, specs)
            try:
                sample_frame = model.get_observables(
                    times,
                    parameter_values=sampled_parameters,
                    observable_map=observable_map,
                )
            except Exception:
                failed_samples += 1
                continue
            sample_frame = sample_frame.melt(
                id_vars="time", var_name="observable", value_name="model_prediction"
            ).rename(columns={"time": "fit_day"})
            sample_frame["sample"] = sample_index
            simulated_samples.append(sample_frame)

        if not simulated_samples:
            raise RuntimeError("All sampled simulations failed.")
        ensemble = pd.concat(simulated_samples, ignore_index=True)
        summary = (
            ensemble.groupby(["observable", "fit_day"], as_index=False)
            .agg(
                mean_prediction=("model_prediction", "mean"),
                std_prediction=("model_prediction", "std"),
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        best_simulation.to_csv(best_fit_cache, index=False)
        summary.to_csv(ribbon_cache, index=False)
        simulated_count = len(simulated_samples)

    present = set(best_simulation["observable"])
    if model_variant == "eff_pre_int":
        plot_order = EFFECTOR_PRECURSOR_INTERMEDIATE_PLOT_ORDER
    elif model_variant == "eff_int_pre":
        plot_order = EFFECTOR_INTERMEDIATE_PRECURSOR_PLOT_ORDER
    elif model_variant == "pre_eff_int":
        plot_order = PRECURSOR_EFFECTOR_INTERMEDIATE_PLOT_ORDER
    elif model_variant == "int_eff_pre":
        plot_order = INTERMEDIATE_EFFECTOR_PRECURSOR_PLOT_ORDER
    elif model_variant == "int_pre_eff":
        plot_order = INTERMEDIATE_PRECURSOR_EFFECTOR_PLOT_ORDER
    else:
        plot_order = PRECURSOR_INTERMEDIATE_EFFECTOR_PLOT_ORDER
    observables = [
        observable for observable in plot_order if observable in present
    ]
    title_fontsize = 20 if paper_style else TITLE_FONTSIZE
    axis_label_fontsize = 18 if paper_style else AXIS_LABEL_FONTSIZE
    tick_fontsize = 18 if paper_style else TICK_FONTSIZE
    legend_fontsize = 18 if paper_style else LEGEND_FONTSIZE
    marker_size = 8 if paper_style else 6
    # Adjust this value to change only the curve width in the paper figure.
    model_linewidth = 3.0 if paper_style else 2.0
    figure_size = (9.5, 9.0) if paper_style else (12.0, 11.4)
    fig, axes = plt.subplots(3, 3, figsize=figure_size, squeeze=False)
    axes_flat = axes.flatten()

    lowest_extent = float(
        np.nanmin(data["data_value"] - data["data_sem"].fillna(0.0))
    )
    upper_extent = max(
        float(np.nanmax(data["data_value"] + data["data_sem"].fillna(0.0))),
        float(np.nanmax(best_simulation["model_prediction"])),
        float(np.nanmax(summary["mean_prediction"] + summary["std_prediction"])),
    )
    y_limits = (
        min(-5000.0, lowest_extent - 5000.0),
        upper_extent * 1.005 if upper_extent > 0.0 else 1.0,
    )
    y_ticks = [0, 100000, 200000, 300000]

    for index, (axis, observable) in enumerate(zip(axes_flat, observables)):
        observable_summary = summary[summary["observable"] == observable].sort_values(
            "fit_day"
        )
        observable_best = best_simulation[
            best_simulation["observable"] == observable
        ].sort_values("fit_day")
        observable_data = data[data["observable"] == observable].sort_values("fit_day")
        upper = (
            observable_summary["mean_prediction"]
            + observable_summary["std_prediction"]
        )
        if log_y:
            # A logarithmic axis cannot show zero. Do not clamp an
            # uncertainty band to an arbitrary positive value: omitting it
            # until its upper bound is positive avoids a spurious ribbon at
            # the lower y-axis limit near time zero.
            positive_band = upper > 0.0
            lower = np.where(
                positive_band,
                np.maximum(
                    observable_summary["mean_prediction"]
                    - observable_summary["std_prediction"],
                    np.finfo(float).tiny,
                ),
                np.nan,
            )
            upper = np.where(positive_band, upper, np.nan)
        else:
            lower = np.maximum(
                observable_summary["mean_prediction"]
                - observable_summary["std_prediction"],
                0.0,
            )

        axis.fill_between(
            observable_summary["fit_day"],
            lower,
            upper,
            color="tab:blue",
            alpha=0.22,
            label="mean +/- std",
        )
        axis.plot(
            observable_best["fit_day"],
            observable_best["model_prediction"],
            color="tab:blue",
            linewidth=model_linewidth,
            label="model",
        )
        yerr = observable_data["data_sem"].to_numpy(dtype=float)
        if np.all(np.isnan(yerr)):
            yerr = None
        axis.errorbar(
            observable_data["fit_day"],
            observable_data["data_value"],
            yerr=yerr,
            fmt="o",
            color="black",
            ecolor="0.4",
            elinewidth=1.0,
            capsize=3,
            markersize=marker_size,
            label="data",
        )

        display_label = PLOT_LABELS[observable]
        axis.set_title(
            display_label,
            fontsize=title_fontsize,
        )
        if index % 3 == 0:
            axis.set_ylabel(
                "Cell counts" if log_y else Y_AXIS_LABEL,
                fontsize=axis_label_fontsize,
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
        if log_y:
            axis.set_yscale("log")
            axis.set_ylim(0.5, 5e5)
            axis.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
            axis.set_xticks([0, 2, 4, 6])
        else:
            axis.set_ylim(*y_limits)
            axis.set_yticks(y_ticks)
            axis.set_yticklabels(scaled_tick_labels(y_ticks, Y_AXIS_SCALE))
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        axis.grid(False)

    if not paper_style:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            fontsize=legend_fontsize,
        )
    fig.supxlabel(
        "Time after immunization (days)",
        fontsize=axis_label_fontsize,
    )
    if log_y:
        fig.tight_layout(
            rect=(0, 0, 1, 1.0 if paper_style else 0.96),
            h_pad=1.6 if paper_style else 3.0,
        )
        fig.subplots_adjust(hspace=0.4 if paper_style else 0.55)
    elif paper_style:
        fig.tight_layout(rect=(0, 0, 1, 1.0), h_pad=1.2)
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    if reuse_cache:
        print(f"Reused cached trajectories and ribbon summary from: {output.parent}")
    else:
        print(f"Simulated {simulated_count} parameter sets ({failed_samples} failed).")
    return output


def infer_model_variant(params_csv: str | Path) -> str:
    """Infer the model variant from a timestamped fit-output path."""
    path_text = str(Path(params_csv)).lower()
    for variant in sorted(MODEL_VARIANTS, key=len, reverse=True):
        if variant in path_text:
            return variant
    return "int_pre_eff"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot full_model_3 trajectories with parameter-uncertainty shading."
    )
    parser.add_argument("--params-csv", type=Path, required=True)
    parser.add_argument(
        "--model-variant",
        default=None,
        choices=sorted(MODEL_VARIANTS),
        help=(
            "Model variant. By default it is inferred from the fitted-parameter "
            "path, falling back to int_pre_eff."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--sampling-method",
        choices=["ci", "ci_clipped", "cv"],
        default="cv",
    )
    parser.add_argument("--cv", type=float, default=0.01)
    parser.add_argument("--num-timepoints", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse the saved best-fit and ribbon-summary CSV files instead of resampling.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Create only the logarithmic-y plot.",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_variant = args.model_variant or infer_model_variant(args.params_csv)
    effective_seed = (
        args.seed
        if args.seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    output = plot_simulation_ensemble(
        params_csv=args.params_csv,
        model_variant=model_variant,
        num_samples=args.num_samples,
        sampling_method=args.sampling_method,
        cv=args.cv,
        num_timepoints=args.num_timepoints,
        seed=effective_seed,
        output_path=args.output_path,
        log_y=args.log_y,
        reuse_cache=args.reuse_cache,
    )
    print(f"Simulation plot saved to: {output}")

    if not args.log_y:
        log_output = output.with_name(f"{output.stem}_log{output.suffix}")
        plot_simulation_ensemble(
            params_csv=args.params_csv,
            model_variant=model_variant,
            num_samples=args.num_samples,
            sampling_method=args.sampling_method,
            cv=args.cv,
            num_timepoints=args.num_timepoints,
            seed=effective_seed,
            output_path=log_output,
            log_y=True,
            reuse_cache=True,
            cache_output_path=output,
        )
        print(f"Log-scale simulation plot saved to: {log_output}")

        paper_output = output.with_name(f"{output.stem}_paper{output.suffix}")
        plot_simulation_ensemble(
            params_csv=args.params_csv,
            model_variant=model_variant,
            num_samples=args.num_samples,
            sampling_method=args.sampling_method,
            cv=args.cv,
            num_timepoints=args.num_timepoints,
            seed=effective_seed,
            output_path=paper_output,
            log_y=True,
            paper_style=True,
            reuse_cache=True,
            cache_output_path=output,
        )
        print(f"Publication-style simulation plot saved to: {paper_output}")


if __name__ == "__main__":
    main()
