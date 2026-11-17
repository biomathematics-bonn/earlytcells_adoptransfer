from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

try:
    from lmfit import Minimizer, conf_interval, minimize
except ImportError:
    Minimizer = None
    conf_interval = None
    minimize = None

try:
    from .data import PreparedData, observable_map_for_variant
    from .model import SubpopModel
    from .parameters import ModelParameters
except ImportError:
    from data import PreparedData, observable_map_for_variant
    from model import SubpopModel
    from parameters import ModelParameters


@dataclass
class FitConfig:
    model_variant: str = "int_pre_eff"
    observable_map: Mapping[str, Sequence[str]] | None = None
    initial_state: Mapping[str, float] | Sequence[float] | None = None
    simulation_method: str = "LSODA"


@dataclass
class MultiStartResult:
    start_index: int
    result: Any
    initial_values: dict[str, float]
    minimizer: Any


@dataclass
class MultiStartSummary:
    best_start: MultiStartResult
    summary: pd.DataFrame
    initial_values: pd.DataFrame
    fitted_values: pd.DataFrame


class ModelFitter:
    """Fit one ODE model variant to the prepared absolute-count data."""

    def __init__(
        self,
        config: FitConfig | None = None,
        data_loader: PreparedData | None = None,
        parameter_builder: ModelParameters | None = None,
    ) -> None:
        self.config = config or FitConfig()
        self.data_loader = data_loader or PreparedData(
            model_variant=self.config.model_variant
        )
        self.parameter_builder = parameter_builder or ModelParameters(
            model_variant=self.config.model_variant
        )
        self.observable_map = dict(
            self.config.observable_map
            or observable_map_for_variant(self.config.model_variant)
        )
        self.data = self.data_loader.build_fit_dataframe()
        self.data = self.data[
            self.data["observable"].isin(self.observable_map)
        ].reset_index(drop=True)
        self._last_minimizer = None
        self.model = SubpopModel(
            model_variant=self.config.model_variant,
            initial_state=self.config.initial_state,
            observable_map=self.observable_map,
        )

    def prediction_frame(self, params: Mapping[str, float] | Any) -> pd.DataFrame:
        if self.data.empty:
            raise ValueError("No data selected for fitting.")

        unique_times = np.sort(self.data["fit_day"].unique())
        observables = self.model.get_observables(
            t_eval=unique_times,
            parameter_values=params,
            observable_map=self.observable_map,
            initial_state=self.config.initial_state,
            method=self.config.simulation_method,
        )
        long_data = observables.melt(
            id_vars="time",
            var_name="observable",
            value_name="model_prediction",
        ).rename(columns={"time": "fit_day"})
        return self.data.merge(long_data, on=["fit_day", "observable"], how="left")

    def residual(self, params: Any) -> np.ndarray:
        merged = self.prediction_frame(params)
        residual = merged["data_value"] - merged["model_prediction"]
        return residual.fillna(0.0).to_numpy(dtype=float)

    def fit(self, params: Any = None, **minimize_kwargs: Any):
        if minimize is None:
            raise ImportError(
                "lmfit is required to run fitting. Install it before fitting."
            )

        params = params or self.parameter_builder.make_lmfit_parameters()
        if Minimizer is not None:
            self._last_minimizer = Minimizer(self.residual, params)
            return self._last_minimizer.minimize(**minimize_kwargs)

        self._last_minimizer = None
        return minimize(self.residual, params, **minimize_kwargs)

    def randomized_lmfit_parameters(
        self,
        rng: np.random.Generator,
    ) -> tuple[Any, dict[str, float]]:
        params = self.parameter_builder.make_lmfit_parameters()
        eps = 1e-9

        for parameter_name, parameter_spec in self.parameter_builder.specs.items():
            parameter = params[parameter_name]
            if parameter.expr is not None or not parameter.vary:
                continue

            min_value = parameter_spec.min
            max_value = parameter_spec.max
            default_value = float(parameter_spec.value)

            if (
                parameter_name.startswith("p_")
                and min_value is not None
                and max_value is not None
            ):
                sampled_value = float(rng.uniform(min_value, max_value))
            elif min_value is not None and max_value is not None:
                if min_value > 0.0 and max_value > 0.0:
                    sampled_value = float(
                        np.exp(
                            rng.uniform(
                                np.log(max(min_value, eps)),
                                np.log(max(max_value, min_value + eps)),
                            )
                        )
                    )
                else:
                    sampled_value = float(rng.uniform(min_value, max_value))
            elif default_value > 0.0 and (
                min_value is None or min_value >= 0.0
            ):
                lower_bound = max(
                    min_value if min_value is not None else eps,
                    default_value / 5.0,
                    eps,
                )
                upper_bound = max(
                    lower_bound * (1.0 + 1e-6),
                    default_value * 5.0,
                )
                sampled_value = float(
                    np.exp(rng.uniform(np.log(lower_bound), np.log(upper_bound)))
                )
            else:
                span = max(abs(default_value), 1.0)
                lower_bound = default_value - 5.0 * span
                upper_bound = default_value + 5.0 * span
                if min_value is not None:
                    lower_bound = max(lower_bound, min_value)
                if max_value is not None:
                    upper_bound = min(upper_bound, max_value)
                sampled_value = (
                    float(lower_bound)
                    if upper_bound <= lower_bound
                    else float(rng.uniform(lower_bound, upper_bound))
                )

            parameter.set(value=sampled_value)

        initial_values = {
            key: float(value) for key, value in params.valuesdict().items()
        }
        return params, initial_values

    def fitted_parameter_table(
        self,
        result,
        *,
        round_display: bool = False,
    ) -> pd.DataFrame:
        z_value_90 = 1.6448536269514722
        eps = 1e-9
        rows: list[dict[str, float | str | None]] = []

        for parameter_name, parameter in result.params.items():
            fit_value = float(parameter.value)
            stderr = None if parameter.stderr is None else float(parameter.stderr)
            min_value = (
                None
                if parameter.min in (None, -np.inf)
                else float(parameter.min)
            )
            max_value = (
                None
                if parameter.max in (None, np.inf)
                else float(parameter.max)
            )

            ci_low: float | None = None
            ci_up: float | None = None
            if stderr is not None:
                if parameter_name.startswith("p_"):
                    p = min(max(fit_value, eps), 1.0 - eps)
                    se_logit = stderr / (p * (1.0 - p))
                    logit_p = float(logit(p))
                    ci_low = float(expit(logit_p - z_value_90 * se_logit))
                    ci_up = float(expit(logit_p + z_value_90 * se_logit))
                elif (
                    min_value is not None
                    and max_value is not None
                    and min_value < max_value
                ):
                    scaled = (fit_value - min_value) / (max_value - min_value)
                    scaled = min(max(scaled, eps), 1.0 - eps)
                    se_bounded_logit = stderr / (
                        (max_value - min_value) * scaled * (1.0 - scaled)
                    )
                    bounded_logit = float(logit(scaled))
                    ci_low = float(
                        min_value
                        + (max_value - min_value)
                        * expit(bounded_logit - z_value_90 * se_bounded_logit)
                    )
                    ci_up = float(
                        min_value
                        + (max_value - min_value)
                        * expit(bounded_logit + z_value_90 * se_bounded_logit)
                    )
                elif fit_value > 0.0 and (
                    min_value is None or min_value >= 0.0
                ):
                    se_log = stderr / fit_value
                    log_value = float(np.log(fit_value))
                    ci_low = float(np.exp(log_value - z_value_90 * se_log))
                    ci_up = float(np.exp(log_value + z_value_90 * se_log))
                else:
                    ci_low = fit_value - z_value_90 * stderr
                    ci_up = fit_value + z_value_90 * stderr

            rows.append(
                {
                    "parameter": parameter_name,
                    "fit_value": fit_value,
                    "stderr": stderr,
                    "CI_low": ci_low,
                    "CI_up": ci_up,
                }
            )

        parameter_table = pd.DataFrame(rows).sort_values(
            "parameter",
            ignore_index=True,
        )
        if round_display:
            rounded_columns = ["fit_value", "stderr", "CI_low", "CI_up"]
            parameter_table[rounded_columns] = parameter_table[
                rounded_columns
            ].round(2)
        return parameter_table

    def confidence_interval_table(
        self,
        result,
        sigmas: Sequence[float] = (1.0, 1.96),
    ) -> pd.DataFrame:
        ci_result = self._calculate_confidence_intervals(
            result,
            sigmas=sigmas,
            parameter_names=[
                name for name, parameter in result.params.items() if parameter.vary
            ],
        )
        rows = [
            self._confidence_interval_row(result, name, entries, sigmas)
            for name, entries in ci_result.items()
        ]
        return pd.DataFrame(rows).sort_values("parameter", ignore_index=True)

    def confidence_interval_diagnostics(
        self,
        result,
        sigmas: Sequence[float] = (1.0,),
    ) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        rows: list[dict[str, float | str | None]] = []
        failures: list[dict[str, str]] = []

        for parameter_name, parameter in result.params.items():
            if not parameter.vary:
                continue
            try:
                ci_result = self._calculate_confidence_intervals(
                    result,
                    sigmas=sigmas,
                    parameter_names=[parameter_name],
                )
            except Exception as exc:
                failures.append({"parameter": parameter_name, "error": str(exc)})
                continue

            entries = ci_result.get(parameter_name, [])
            rows.append(
                self._confidence_interval_row(
                    result,
                    parameter_name,
                    entries,
                    sigmas,
                )
            )

        columns = ["parameter", "best_fit"]
        for sigma in sigmas:
            sigma_label = str(sigma).replace(".", "_")
            columns.extend(
                [f"lower_{sigma_label}sigma", f"upper_{sigma_label}sigma"]
            )
        table = pd.DataFrame(rows, columns=columns)
        if not table.empty:
            table = table.sort_values("parameter", ignore_index=True)
        return table, failures

    def _calculate_confidence_intervals(
        self,
        result,
        *,
        sigmas: Sequence[float],
        parameter_names: Sequence[str],
    ):
        if conf_interval is None:
            raise ImportError("lmfit.conf_interval is unavailable.")
        if self._last_minimizer is None:
            raise RuntimeError(
                "Confidence intervals require a stored lmfit Minimizer instance."
            )
        return conf_interval(
            self._last_minimizer,
            result,
            sigmas=list(sigmas),
            p_names=list(parameter_names),
        )

    @staticmethod
    def _confidence_interval_row(
        result,
        parameter_name: str,
        entries,
        sigmas: Sequence[float],
    ) -> dict[str, float | str | None]:
        row: dict[str, float | str | None] = {
            "parameter": parameter_name,
            "best_fit": float(result.params[parameter_name].value),
        }
        bounds = {float(sigma): [None, None] for sigma in sigmas}

        for sigma, value in entries:
            if sigma == 0.0:
                row["best_fit"] = float(value)
                continue
            abs_sigma = abs(float(sigma))
            if abs_sigma not in bounds:
                continue
            if sigma < 0:
                bounds[abs_sigma][0] = float(value)
            else:
                bounds[abs_sigma][1] = float(value)

        for sigma in sigmas:
            sigma_label = str(sigma).replace(".", "_")
            lower, upper = bounds[float(sigma)]
            row[f"lower_{sigma_label}sigma"] = lower
            row[f"upper_{sigma_label}sigma"] = upper
        return row


def run_multistart_fit(
    fitter: ModelFitter,
    number_of_starts: int,
    *,
    fit_method: str = "leastsq",
    random_seed: int | None = None,
) -> MultiStartSummary:
    """Run repeated local fits and select the result with lowest chi-square."""
    if number_of_starts < 1:
        raise ValueError("number_of_starts must be at least 1.")

    rng = np.random.default_rng(random_seed)
    start_results: list[MultiStartResult] = []
    summary_rows: list[dict[str, Any]] = []
    initial_rows: list[dict[str, Any]] = []
    fitted_rows: list[dict[str, Any]] = []

    for start_index in range(1, number_of_starts + 1):
        if start_index == 1:
            start_params = fitter.parameter_builder.make_lmfit_parameters()
            initial_values = {
                name: float(value)
                for name, value in start_params.valuesdict().items()
            }
        else:
            start_params, initial_values = fitter.randomized_lmfit_parameters(rng)

        result = fitter.fit(params=start_params, method=fit_method)
        start_result = MultiStartResult(
            start_index=start_index,
            result=result,
            initial_values=initial_values,
            minimizer=fitter._last_minimizer,
        )
        start_results.append(start_result)

        print(
            f"Run {start_index}/{number_of_starts} completed "
            f"(chisqr={float(getattr(result, 'chisqr', np.nan)):.6g}, "
            f"AIC={float(getattr(result, 'aic', np.nan)):.6g})",
            flush=True,
        )
        summary_rows.append(
            {
                "start_index": start_index,
                "success": bool(getattr(result, "success", True)),
                "message": str(getattr(result, "message", "")),
                "nfev": getattr(result, "nfev", np.nan),
                "chisqr": getattr(result, "chisqr", np.nan),
                "redchi": getattr(result, "redchi", np.nan),
                "aic": getattr(result, "aic", np.nan),
                "bic": getattr(result, "bic", np.nan),
            }
        )
        initial_rows.append({"start_index": start_index, **initial_values})
        fitted_rows.append(
            {
                "start_index": start_index,
                **{
                    name: float(value)
                    for name, value in result.params.valuesdict().items()
                },
            }
        )

    def result_score(item: MultiStartResult) -> tuple[float, float]:
        chisqr = float(getattr(item.result, "chisqr", np.inf))
        aic = float(getattr(item.result, "aic", np.inf))
        return (
            chisqr if np.isfinite(chisqr) else np.inf,
            aic if np.isfinite(aic) else np.inf,
        )

    best_start = min(start_results, key=result_score)
    fitter._last_minimizer = best_start.minimizer
    return MultiStartSummary(
        best_start=best_start,
        summary=pd.DataFrame(summary_rows).sort_values("start_index"),
        initial_values=pd.DataFrame(initial_rows).sort_values("start_index"),
        fitted_values=pd.DataFrame(fitted_rows).sort_values("start_index"),
    )
