from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from lmfit import Parameters
except ImportError:
    Parameters = Any


@dataclass
class ParameterSpec:
    value: float
    min: float | None = None
    max: float | None = None
    vary: bool = False
    expr: str | None = None


def parameter_specs_for_variant(model_variant: str) -> dict[str, ParameterSpec]:
    if model_variant == "int_pre_eff":
        specs: dict[str, ParameterSpec] = {
            "gamma_shape": ParameterSpec(5.0, min=1.0, vary=False),
            "gamma_rate": ParameterSpec(1.3, min=1e-12, max=10.0, vary=False),
            "gamma_shape_int": ParameterSpec(10.0, min=1.0, vary=False),
            "gamma_rate_int": ParameterSpec(6.5, min=1e-12, max=15.0, vary=False),

            "d_naive": ParameterSpec(1.1, min=0.0, max=2.0, vary=True),
            "d_early_activated": ParameterSpec(0.28, min=0.0, max=2.0, vary=True),
            "d_inter_th0": ParameterSpec(0.17, min=0.0, max=2.0, vary=True),
            "d_pth1": ParameterSpec(0.24, min=0.0, max=2.0, vary=True),
            "d_interth1": ParameterSpec(1.5, min=0.0,max=3.0, vary=True),
            "d_ptfh": ParameterSpec(0.004, min=0.0, max=2.0, vary=True),
            "d_intertfh": ParameterSpec(0.17, min=0.0, max=2.0, vary=True),

            "r_inter_th0": ParameterSpec(12.39, min=3.0, max=30.0, vary=True),
            "r_interth1": ParameterSpec(12.09, min=3.0, max=30.0, vary=True),
            "r_intertfh": ParameterSpec(14.64, min=3.0, max=30.0, vary=True),
            "r_early_activated": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_pth1": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_ptfh": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_tfh": ParameterSpec(2.18, min=1.0, max=5.0, vary=True),

            "p_inter_th0": ParameterSpec(0.5912, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_inter_th1": ParameterSpec(0.2111, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_inter_tfh": ParameterSpec(0.1659, min=0.0, max=1.0, vary=False),  # d3 fractions

            "delta": ParameterSpec(3.5, min=0.0, max=10.0, vary=False),
            "delta_inter_th0": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_ptfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_th1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_tfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_pth1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),

            "alpha": ParameterSpec(0.86, min=0.0, max=1.0, vary=True),
            "alpha_ptfh": ParameterSpec(0.83, min=0.0, max=1.0, vary=True),
            "alpha_pth1": ParameterSpec(0.93, min=0.0, max=1.0,vary=True),
            "alpha_inter_tfh": ParameterSpec(0.29, min=0.0, max=1.0, vary=True),
            "alpha_inter_th0": ParameterSpec(0.25, min=0.0, max=1.0, vary=True),
            "alpha_inter_th1": ParameterSpec(0.17, min=0.0, max=1.0, vary=True),
        }
        return specs

    if model_variant == "int_eff_pre":
        specs: dict[str, ParameterSpec] = {
            "gamma_shape": ParameterSpec(5.0, min=1.0, vary=False),
            "gamma_rate": ParameterSpec(1.3, min=1e-12, max=10.0, vary=False),
            "gamma_shape_int": ParameterSpec(10.0, min=1.0, vary=False),
            "gamma_rate_int": ParameterSpec(6.5, min=1e-12, max=15.0, vary=False),

            "d_naive": ParameterSpec(1.57, min=0.0, max=2.0, vary=True),
            "d_inter_th0": ParameterSpec(0.3, min=0.0, max=2.0, vary=True),
            "d_interth1": ParameterSpec(1.03, min=0.0, max=3.0, vary=True),
            "d_intertfh": ParameterSpec(0.15, min=0.0, max=2.0, vary=True),
            "d_th0": ParameterSpec(0.99, min=0.0, max=2.0, vary=True),
            "d_th1": ParameterSpec(2.0, min=0.0, max=2.0, vary=True),
            "d_tfh": ParameterSpec(1.3, min=0.0, max=2.0, vary=True),

            "r_inter_th0": ParameterSpec(12.9, min=3.0, max=30.0, vary=True),
            "r_interth1": ParameterSpec(12.39, min=3.0, max=30.0, vary=True),
            "r_intertfh": ParameterSpec(15, min=3.0, max=30.0, vary=True),
            "r_early_activated": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_pth1": ParameterSpec(54, min=10.0, max=60.0, vary=True),
            "r_ptfh": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_tfh": ParameterSpec(1, min=1.0, max=5.0, vary=True),

            "p_inter_th0": ParameterSpec(0.5912, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_inter_th1": ParameterSpec(0.2111, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_inter_tfh": ParameterSpec(0.1659, min=0.0, max=1.0, vary=False),  # d3 fractions

            "delta": ParameterSpec(3.5, min=0.0, max=10.0, vary=False),
            "delta_inter_th0": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_ptfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_th1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_tfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_pth1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),

            "alpha": ParameterSpec(0.88, min=0.0, max=1.0, vary=True),
            "alpha_ptfh": ParameterSpec(0.82, min=0.0, max=1.0, vary=True),
            "alpha_pth1": ParameterSpec(0.88, min=0.0, max=1.0, vary=True),
            "alpha_inter_tfh": ParameterSpec(0.3, min=0.0, max=1.0, vary=True),
            "alpha_inter_th0": ParameterSpec(0.25, min=0.0, max=1.0, vary=True),
            "alpha_inter_th1": ParameterSpec(0.2, min=0.0, max=1.0, vary=True),
        }
        return specs

    if model_variant == "pre_int_eff":
        specs: dict[str, ParameterSpec] = {
            "gamma_shape": ParameterSpec(5.0, min=1.0, vary=False),
            "gamma_rate": ParameterSpec(1.3, min=1e-12, max=10.0, vary=False),
            "gamma_shape_int": ParameterSpec(10.0, min=1.0, vary=False),
            "gamma_rate_int": ParameterSpec(6.5, min=1e-12, max=15.0, vary=False),

            "d_naive": ParameterSpec(0.5, min=0.0, max=2.0, vary=True),
            "d_early_activated": ParameterSpec(1.3, min=0.0, max=2.0, vary=True),
            "d_inter_th0": ParameterSpec(0.12, min=0.0, max=2.0, vary=True),
            "d_pth1": ParameterSpec(0.6, min=0.0, max=2.0, vary=True),
            "d_interth1": ParameterSpec(0.2, min=0.0,max=3.0, vary=True),
            "d_ptfh": ParameterSpec(0.002, min=0.0, max=2.0, vary=True),
            "d_intertfh": ParameterSpec(0.02, min=0.0, max=2.0, vary=True),

            "r_inter_th0": ParameterSpec(3, min=3.0, max=30.0, vary=True),
            "r_interth1": ParameterSpec(3, min=3.0, max=30.0, vary=True),
            "r_intertfh": ParameterSpec(30, min=3.0, max=30.0, vary=True),
            "r_early_activated": ParameterSpec(40, min=10.0, max=60.0, vary=True),
            "r_pth1": ParameterSpec(40, min=10.0, max=60.0, vary=True),
            "r_ptfh": ParameterSpec(40, min=10.0, max=60.0, vary=True),
            "r_tfh": ParameterSpec(1.39, min=1.0, max=5.0, vary=True),

            "p_early_activated": ParameterSpec(0.3492, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_pth1": ParameterSpec(0.151, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_ptfh": ParameterSpec(0.3599, min=0.0, max=1.0, vary=False),  # d3 fractions

            "delta": ParameterSpec(3.5, min=0.0, max=10.0, vary=False),
            "delta_inter_th0": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_ptfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_th1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_tfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_pth1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),

            "alpha": ParameterSpec(0.7, min=0.0, max=1.0, vary=True),
            "alpha_ptfh": ParameterSpec(0.7, min=0.0, max=1.0, vary=True),
            "alpha_pth1": ParameterSpec(0.8, min=0.0, max=1.0,vary=True),
            "alpha_inter_tfh": ParameterSpec(0.5, min=0.0, max=1.0, vary=True),
            "alpha_inter_th0": ParameterSpec(0.02, min=0.0, max=1.0, vary=True),
            "alpha_inter_th1": ParameterSpec(0.003, min=0.0, max=1.0, vary=True),

        }
        return specs

    if model_variant == "pre_eff_int":
        specs: dict[str, ParameterSpec] = {
            "gamma_shape": ParameterSpec(5.0, min=1.0, vary=False),
            "gamma_rate": ParameterSpec(1.3, min=1e-12, max=10.0, vary=False),
            "gamma_shape_int": ParameterSpec(10.0, min=1.0, vary=False),
            "gamma_rate_int": ParameterSpec(6.5, min=1e-12, max=15.0, vary=False),

            "d_naive": ParameterSpec(2.0, min=0.0, max=2.0, vary=True),
            "d_early_activated": ParameterSpec(0.39, min=0.0, max=2.0, vary=True),
            "d_pth1": ParameterSpec(0.44, min=0.0, max=2.0, vary=True),
            "d_ptfh": ParameterSpec(0.0044, min=0.0, max=2.0, vary=True),
            "d_th0": ParameterSpec(0.2, min=0.0, max=3.0, vary=True),
            "d_th1": ParameterSpec(0.09, min=0.0,max=3.0, vary=True),
            "d_tfh": ParameterSpec(0.3, min=0.0, max=3.0, vary=True),

            "r_inter_th0": ParameterSpec(18.5, min=3.0, max=30.0, vary=True),
            "r_interth1": ParameterSpec(16.79, min=3.0, max=30.0, vary=True),
            "r_intertfh": ParameterSpec(29.99, min=3.0, max=30.0, vary=True),
            "r_early_activated": ParameterSpec(41.3, min=10.0, max=60.0, vary=True),
            "r_pth1": ParameterSpec(40.17, min=10.0, max=60.0, vary=True),
            "r_ptfh": ParameterSpec(32.13, min=10.0, max=60.0, vary=True),
            "r_tfh": ParameterSpec(2.51, min=1.0, max=5.0, vary=True),

            "p_early_activated": ParameterSpec(0.3492, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_pth1": ParameterSpec(0.151, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_ptfh": ParameterSpec(0.3599, min=0.0, max=1.0, vary=False),  # d3 fractions

            "delta": ParameterSpec(3.5, min=0.0, max=10.0, vary=False),
            "delta_inter_th0": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_ptfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_th1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_tfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_pth1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),

            "alpha": ParameterSpec(0.73, min=0.0, max=1.0, vary=True),
            "alpha_ptfh": ParameterSpec(0.64, min=0.0, max=1.0, vary=True),
            "alpha_pth1": ParameterSpec(0.71, min=0.0, max=1.0,vary=True),
            "alpha_inter_tfh": ParameterSpec(0.45, min=0.0, max=1.0, vary=True),
            "alpha_inter_th0": ParameterSpec(0.35, min=0.0, max=1.0, vary=True),
            "alpha_inter_th1": ParameterSpec(0.31, min=0.0, max=1.0, vary=True),
        }
        return specs

    if model_variant == "eff_int_pre":
        specs: dict[str, ParameterSpec] = {
            "gamma_shape": ParameterSpec(5.0, min=1.0, vary=False),
            "gamma_rate": ParameterSpec(1.3, min=1e-12, max=10.0, vary=False),
            "gamma_shape_int": ParameterSpec(10.0, min=1.0, vary=False),
            "gamma_rate_int": ParameterSpec(6.5, min=1e-12, max=15.0, vary=False),

            "d_naive": ParameterSpec(5.0, min=0.0, max=5.0, vary=True),
            "d_th0": ParameterSpec(0.45, min=0.0, max=2.0, vary=True),
            "d_inter_th0": ParameterSpec(0.08, min=0.0, max=2.0, vary=True),
            "d_th1": ParameterSpec(0.15, min=0.0, max=2.0, vary=True),
            "d_interth1": ParameterSpec(1.44, min=0.0,max=3.0, vary=True),
            "d_tfh": ParameterSpec(0.62, min=0.0, max=2.0, vary=True),
            "d_intertfh": ParameterSpec(0.33, min=0.0, max=2.0, vary=True),

            "r_inter_th0": ParameterSpec(12.18, min=3.0, max=30.0, vary=True),
            "r_interth1": ParameterSpec(11.95, min=3.0, max=30.0, vary=True),
            "r_intertfh": ParameterSpec(24.58, min=3.0, max=30.0, vary=True),
            "r_early_activated": ParameterSpec(80, min=10.0, max=80.0, vary=True),
            "r_pth1": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_ptfh": ParameterSpec(60, min=10.0, max=60.0, vary=True),
            "r_tfh": ParameterSpec(5, min=1.0, max=5.0, vary=True),

            "p_th0": ParameterSpec(0.6232, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_th1": ParameterSpec(0.3745, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_tfh": ParameterSpec(0.002, min=0.0, max=1.0, vary=False),  # d3 fractions

            "delta": ParameterSpec(3.5, min=0.0, max=10.0, vary=False),
            "delta_inter_th0": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_ptfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_th1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_tfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_pth1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),

            "alpha": ParameterSpec(0.98, min=0.0, max=1.0, vary=True),
            "alpha_ptfh": ParameterSpec(0.75, min=0.0, max=1.0, vary=True),
            "alpha_pth1": ParameterSpec(0.92, min=0.0, max=1.0,vary=True),
            "alpha_inter_tfh": ParameterSpec(0.39, min=0.0, max=2.0, vary=True),
            "alpha_inter_th0": ParameterSpec(0.25, min=0.0, max=2.0, vary=True),
            "alpha_inter_th1": ParameterSpec(0.17, min=0.0, max=2.0, vary=True),
        }
        return specs

    if model_variant == "eff_pre_int":
        specs: dict[str, ParameterSpec] = {
            "gamma_shape": ParameterSpec(5.0, min=1.0, vary=False),
            "gamma_rate": ParameterSpec(1.3, min=1e-12, max=10.0, vary=False),
            "gamma_shape_int": ParameterSpec(10.0, min=1.0, vary=False),
            "gamma_rate_int": ParameterSpec(6.5, min=1e-12, max=15.0, vary=False),

            "d_naive": ParameterSpec(2.0, min=0.0, max=2.0, vary=True),
            "d_th0": ParameterSpec(0.02, min=0.0, max=2.0, vary=True),
            "d_early_activated": ParameterSpec(1.5, min=0.0, max=2.0, vary=True),
            "d_th1": ParameterSpec(0.004, min=0.0, max=2.0, vary=True),
            "d_pth1": ParameterSpec(0.7, min=0.0,max=3.0, vary=True),
            "d_tfh": ParameterSpec(1.05, min=0.0, max=2.0, vary=True),
            "d_ptfh": ParameterSpec(0.02, min=0.0, max=2.0, vary=True),

            "r_inter_th0": ParameterSpec(3.32, min=3.0, max=30.0, vary=True),
            "r_interth1": ParameterSpec(3.62, min=3.0, max=30.0, vary=True),
            "r_intertfh": ParameterSpec(29.77, min=3.0, max=30.0, vary=True),
            "r_early_activated": ParameterSpec(63.25, min=10.0, max=80.0, vary=True),
            "r_pth1": ParameterSpec(76.62, min=10.0, max=80.0, vary=True),
            "r_ptfh": ParameterSpec(74.42, min=10.0, max=80.0, vary=True),
            "r_tfh": ParameterSpec(5.95, min=1.0, max=10.0, vary=True),

            "p_th0": ParameterSpec(0.6232, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_th1": ParameterSpec(0.3745, min=0.0, max=1.0, vary=False),  # d3 fractions
            "p_tfh": ParameterSpec(0.002, min=0.0, max=1.0, vary=False),  # d3 fractions

            "delta": ParameterSpec(3.5, min=0.0, max=10.0, vary=False),
            "delta_inter_th0": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_ptfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_th1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_inter_tfh": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),
            "delta_pth1": ParameterSpec(3.5, min=0.0, vary=False, expr="delta"),

            "alpha": ParameterSpec(0.78, min=0.0, max=1.0, vary=True),
            "alpha_ptfh": ParameterSpec(0.88, min=0.0, max=1.0, vary=True),
            "alpha_pth1": ParameterSpec(0.89, min=0.0, max=1.0,vary=True),
            "alpha_inter_tfh": ParameterSpec(0.44, min=0.0, max=1.0, vary=True),
            "alpha_inter_th0": ParameterSpec(0.0, min=0.0, max=1.0, vary=True),
            "alpha_inter_th1": ParameterSpec(0.0, min=0.0, max=1.0, vary=True),
        }
        return specs



    known_variants = "int_pre_eff, pre_int_eff, int_eff_pre, pre_eff_int, eff_int_pre, eff_pre_int"
    raise ValueError(f"Unknown model variant {model_variant!r}. Known variants: {known_variants}")

class ModelParameters:
    """Helper for building parameter dictionaries or lmfit parameter sets."""

    def __init__(
        self,
        specs: Mapping[str, ParameterSpec] | None = None,
        model_variant: str = "int_pre_eff",
    ) -> None:
        self.model_variant = model_variant
        self.specs = dict(specs or parameter_specs_for_variant(model_variant))

    def make_lmfit_parameters(
        self,
        overrides: Mapping[str, float | ParameterSpec | Mapping[str, Any]] | None = None,
    ) -> Parameters:
        try:
            from lmfit import Parameters as LmfitParameters
        except ImportError as exc:
            raise ImportError(
                "lmfit is required to build fitting parameters. Install it before running model fitting."
            ) from exc

        specs = self._merged_specs(overrides)
        params = LmfitParameters()
        plain_specs = [(name, spec) for name, spec in specs.items() if spec.expr is None]
        expr_specs = [(name, spec) for name, spec in specs.items() if spec.expr is not None]

        for name, spec in [*plain_specs, *expr_specs]:
            kwargs: dict[str, Any] = {"value": spec.value, "vary": spec.vary}
            if spec.min is not None:
                kwargs["min"] = spec.min
            if spec.max is not None:
                kwargs["max"] = spec.max
            if spec.expr is not None:
                kwargs["expr"] = spec.expr
            params.add(name, **kwargs)
        return params

    def defaults_dict(
        self,
        overrides: Mapping[str, float | ParameterSpec | Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        specs = self._merged_specs(overrides)
        values = {name: spec.value for name, spec in specs.items() if spec.expr is None}
        expr_specs = {name: spec for name, spec in specs.items() if spec.expr is not None}

        for name, spec in expr_specs.items():
            if spec.expr is None:
                continue
            values[name] = float(eval(spec.expr, {"__builtins__": {}}, values))

        return {name: float(values.get(name, spec.value)) for name, spec in specs.items()}

    def _merged_specs(
        self,
        overrides: Mapping[str, float | ParameterSpec | Mapping[str, Any]] | None,
    ) -> dict[str, ParameterSpec]:
        merged = dict(self.specs)
        if overrides is None:
            return merged

        for name, override in overrides.items():
            if isinstance(override, ParameterSpec):
                merged[name] = override
                continue

            if isinstance(override, Mapping):
                base = merged.get(name, ParameterSpec(value=0.0))
                merged[name] = ParameterSpec(
                    value=float(override.get("value", base.value)),
                    min=override.get("min", base.min),
                    max=override.get("max", base.max),
                    vary=bool(override.get("vary", base.vary)),
                    expr=override.get("expr", base.expr),
                )
                continue

            base = merged.get(name, ParameterSpec(value=float(override)))
            merged[name] = ParameterSpec(
                value=float(override),
                min=base.min,
                max=base.max,
                vary=base.vary,
                expr=base.expr,
            )

        return merged
