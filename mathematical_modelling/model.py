from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import gamma


class SubpopModel:
    STATE_NAMES = (
        "naive",
        "early_activated",
        "inter_th0",
        "th0",
        "pre_th1",
        "inter_th1",
        "th1",
        "pre_tfh",
        "inter_tfh",
        "tfh",
    )
    INITIAL_STATE = {name: 0.0 for name in STATE_NAMES} | {"naive": 5425.0}
    OBSERVABLES = {
        "Ki-67- Tbet- Bcl6-": ("th0",),
        "Ki-67+ Tbet- Bcl6-": ("early_activated",),
        "Ki-67 inter Tbet- Bcl6-": ("inter_th0",),
        "Ki-67+ Tbet+ / inter Bcl6-": ("pre_th1",),
        "Ki-67 inter Tbet+ / inter Bcl6-": ("inter_th1",),
        "Ki-67- Tbet+ / inter Bcl6-": ("th1",),
        "Ki-67+ Tbet- Bcl6+": ("pre_tfh",),
        "Ki-67 inter Tbet- Bcl6+": ("inter_tfh",),
        "Ki-67- Tbet- Bcl6+": ("tfh",),
    }

    def __init__(
        self,
        parameter_values: Mapping[str, float] | None = None,
        model_variant: str = "int_pre_eff",
        initial_state: Mapping[str, float] | Sequence[float] | None = None,
        observable_map: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.model_functions = {
            "int_pre_eff": self.ode_int_pre_eff,
            "pre_int_eff": self.ode_pre_int_eff,
            "int_eff_pre": self.ode_int_eff_pre,
            "pre_eff_int": self.ode_pre_eff_int,
            "eff_int_pre": self.ode_eff_int_pre,
            "eff_pre_int": self.ode_eff_pre_int,
        }
        if model_variant not in self.model_functions:
            known_variants = ", ".join(self.model_functions)
            raise ValueError(
                f"Unknown model variant '{model_variant}'. "
                f"Known variants: {known_variants}"
            )
        self.model_variant = model_variant
        self.state_names = self.STATE_NAMES
        self.observable_map = dict(observable_map or self.OBSERVABLES)
        state = self.INITIAL_STATE if initial_state is None else initial_state
        self.initial_state = self._coerce_initial_state(state)
        self.params: dict[str, float] = {}
        if parameter_values is not None:
            self.set_parameters(parameter_values)

    def _coerce_initial_state(
        self,
        state: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        if isinstance(state, Mapping):
            return np.array(
                [float(state.get(name, 0.0)) for name in self.state_names]
            )

        state = np.asarray(state, dtype=float)
        if state.shape != (len(self.state_names),):
            raise ValueError(
                f"Initial state must have {len(self.state_names)} entries."
            )
        return state

    def set_parameters(self, values: Mapping[str, float] | Any) -> None:
        if hasattr(values, "valuesdict"):
            values = values.valuesdict()
        self.params = {name: float(value) for name, value in values.items()}
        for name, value in self.params.items():
            setattr(self, name, value)

    def rhs(self, t: float, state: Sequence[float]) -> tuple[float, ...]:
        return self.model_functions[self.model_variant](t, state)

    def _proliferation_onset(self, t: float) -> float:
        return float(
            gamma.cdf(
                t,
                a=self.gamma_shape_int,
                scale=1.0 / self.gamma_rate_int,
            )
        )

    def _tfh_growth(self, t: float, tfh: float) -> float:
        return (
            self.r_tfh
            * gamma.cdf(
                t,
                a=self.gamma_shape,
                scale=1.0 / self.gamma_rate,
            )
            * tfh
        )

    def ode_int_pre_eff(
        self,
        t: float,
        state: Sequence[float],
    ) -> tuple[float, ...]:
        """Intermediate to precursor to effector differentiation."""
        (
            naive,
            early_activated,
            inter_th0,
            th0,
            pre_th1,
            inter_th1,
            th1,
            pre_tfh,
            inter_tfh,
            tfh,
        ) = state
        proliferation_onset = self._proliferation_onset(t)
        dydt_naive = -self.d_naive * naive
        dydt_inter_th0 = self.d_naive * self.p_inter_th0 * naive + self.r_inter_th0 * proliferation_onset * np.exp(-self.alpha_inter_th0 * t) * inter_th0 - self.d_inter_th0 * inter_th0 - self.delta_inter_th0 * inter_th0
        dydt_early_activated = self.d_inter_th0 * inter_th0 + self.r_early_activated * proliferation_onset * np.exp(-self.alpha * t) * early_activated - self.d_early_activated * early_activated - self.delta * early_activated
        dydt_th0 = self.d_early_activated * early_activated
        dydt_inter_th1 = self.d_naive * self.p_inter_th1 * naive + self.r_interth1 * proliferation_onset * np.exp(-self.alpha_inter_th1 * t) * inter_th1 - self.d_interth1 * inter_th1 - self.delta_inter_th1 * inter_th1
        dydt_pth1 = self.d_interth1 * inter_th1 + self.r_pth1 * proliferation_onset * np.exp(-self.alpha_pth1 * t) * pre_th1 - self.d_pth1 * pre_th1 - self.delta_pth1 * pre_th1
        dydt_th1 = self.d_pth1 * pre_th1
        dydt_inter_tfh = self.d_naive * self.p_inter_tfh * naive + self.r_intertfh * proliferation_onset * np.exp(-self.alpha_inter_tfh * t) * inter_tfh - self.d_intertfh * inter_tfh - self.delta_inter_tfh * inter_tfh
        dydt_ptfh = self.d_intertfh * inter_tfh + self.r_ptfh * proliferation_onset * np.exp(-self.alpha_ptfh * t) * pre_tfh - self.d_ptfh * pre_tfh - self.delta_ptfh * pre_tfh
        dydt_tfh = self.d_ptfh * pre_tfh + self._tfh_growth(t, tfh)
        return (
            dydt_naive,
            dydt_early_activated,
            dydt_inter_th0,
            dydt_th0,
            dydt_pth1,
            dydt_inter_th1,
            dydt_th1,
            dydt_ptfh,
            dydt_inter_tfh,
            dydt_tfh,
        )

    def ode_pre_int_eff(
        self,
        t: float,
        state: Sequence[float],
    ) -> tuple[float, ...]:
        """Precursor to intermediate to effector differentiation."""
        (
            naive,
            early_activated,
            inter_th0,
            th0,
            pre_th1,
            inter_th1,
            th1,
            pre_tfh,
            inter_tfh,
            tfh,
        ) = state
        proliferation_onset = self._proliferation_onset(t)
        dydt_naive = -self.d_naive * naive
        dydt_early_activated = self.d_naive * self.p_early_activated * naive + self.r_early_activated * proliferation_onset * np.exp(-self.alpha * t) * early_activated - self.d_early_activated * early_activated - self.delta * early_activated
        dydt_inter_th0 = self.d_early_activated * early_activated + self.r_inter_th0 * proliferation_onset * np.exp(-self.alpha_inter_th0 * t) * inter_th0 - self.d_inter_th0 * inter_th0 - self.delta_inter_th0 * inter_th0
        dydt_th0 = self.d_inter_th0 * inter_th0
        dydt_pth1 = self.d_naive * self.p_pth1 * naive + self.r_pth1 * proliferation_onset * np.exp(-self.alpha_pth1 * t) * pre_th1 - self.d_pth1 * pre_th1 - self.delta_pth1 * pre_th1
        dydt_inter_th1 = self.d_pth1 * pre_th1 + self.r_interth1 * proliferation_onset * np.exp(-self.alpha_inter_th1 * t) * inter_th1 - self.d_interth1 * inter_th1 - self.delta_inter_th1 * inter_th1
        dydt_th1 = self.d_interth1 * inter_th1
        dydt_ptfh = self.d_naive * self.p_ptfh * naive + self.r_ptfh * proliferation_onset * np.exp(-self.alpha_ptfh * t) * pre_tfh - self.d_ptfh * pre_tfh - self.delta_ptfh * pre_tfh
        dydt_inter_tfh = self.d_ptfh * pre_tfh + self.r_intertfh * proliferation_onset * np.exp(-self.alpha_inter_tfh * t) * inter_tfh - self.d_intertfh * inter_tfh - self.delta_inter_tfh * inter_tfh
        dydt_tfh = self.d_intertfh * inter_tfh + self._tfh_growth(t, tfh)
        return (
            dydt_naive,
            dydt_early_activated,
            dydt_inter_th0,
            dydt_th0,
            dydt_pth1,
            dydt_inter_th1,
            dydt_th1,
            dydt_ptfh,
            dydt_inter_tfh,
            dydt_tfh,
        )

    def ode_int_eff_pre(
        self,
        t: float,
        state: Sequence[float],
    ) -> tuple[float, ...]:
        """Intermediate to effector to precursor differentiation."""
        (
            naive,
            early_activated,
            inter_th0,
            th0,
            pre_th1,
            inter_th1,
            th1,
            pre_tfh,
            inter_tfh,
            tfh,
        ) = state
        proliferation_onset = self._proliferation_onset(t)
        dydt_naive = -self.d_naive * naive
        dydt_inter_th0 = self.d_naive * self.p_inter_th0 * naive + self.r_inter_th0 * proliferation_onset * np.exp(-self.alpha_inter_th0 * t) * inter_th0 - self.d_inter_th0 * inter_th0 - self.delta_inter_th0 * inter_th0
        dydt_th0 = self.d_inter_th0 * inter_th0 - self.d_th0 * th0
        dydt_early_activated = self.d_th0 * th0 + self.r_early_activated * proliferation_onset * np.exp(-self.alpha * t) * early_activated - self.delta * early_activated
        dydt_inter_th1 = self.d_naive * self.p_inter_th1 * naive + self.r_interth1 * proliferation_onset * np.exp(-self.alpha_inter_th1 * t) * inter_th1 - self.d_interth1 * inter_th1 - self.delta_inter_th1 * inter_th1
        dydt_th1 = self.d_interth1 * inter_th1 - self.d_th1 * th1
        dydt_pth1 = self.d_th1 * th1 + self.r_pth1 * proliferation_onset * np.exp(-self.alpha_pth1 * t) * pre_th1 - self.delta_pth1 * pre_th1
        dydt_inter_tfh = self.d_naive * self.p_inter_tfh * naive + self.r_intertfh * proliferation_onset * np.exp(-self.alpha_inter_tfh * t) * inter_tfh - self.d_intertfh * inter_tfh - self.delta_inter_tfh * inter_tfh
        dydt_tfh = self.d_intertfh * inter_tfh + self._tfh_growth(t, tfh) - self.d_tfh * tfh
        dydt_ptfh = self.d_tfh * tfh + self.r_ptfh * proliferation_onset * np.exp(-self.alpha_ptfh * t) * pre_tfh - self.delta_ptfh * pre_tfh
        return (
            dydt_naive,
            dydt_early_activated,
            dydt_inter_th0,
            dydt_th0,
            dydt_pth1,
            dydt_inter_th1,
            dydt_th1,
            dydt_ptfh,
            dydt_inter_tfh,
            dydt_tfh,
        )

    def ode_pre_eff_int(
        self,
        t: float,
        state: Sequence[float],
    ) -> tuple[float, ...]:
        """Precursor to effector to intermediate differentiation."""
        (
            naive,
            early_activated,
            inter_th0,
            th0,
            pre_th1,
            inter_th1,
            th1,
            pre_tfh,
            inter_tfh,
            tfh,
        ) = state
        proliferation_onset = self._proliferation_onset(t)
        dydt_naive = -self.d_naive * naive
        dydt_early_activated = self.d_naive * self.p_early_activated * naive + self.r_early_activated * proliferation_onset * np.exp(-self.alpha * t) * early_activated - self.d_early_activated * early_activated - self.delta * early_activated
        dydt_th0 = self.d_early_activated * early_activated - self.d_th0 * th0
        dydt_inter_th0 = self.d_th0 * th0 + self.r_inter_th0 * proliferation_onset * np.exp(-self.alpha_inter_th0 * t) * inter_th0 - self.delta_inter_th0 * inter_th0
        dydt_pth1 = self.d_naive * self.p_pth1 * naive + self.r_pth1 * proliferation_onset * np.exp(-self.alpha_pth1 * t) * pre_th1 - self.d_pth1 * pre_th1 - self.delta_pth1 * pre_th1
        dydt_th1 = self.d_pth1 * pre_th1 - self.d_th1 * th1
        dydt_inter_th1 = self.d_th1 * th1 + self.r_interth1 * proliferation_onset * np.exp(-self.alpha_inter_th1 * t) * inter_th1 - self.delta_inter_th1 * inter_th1
        dydt_ptfh = self.d_naive * self.p_ptfh * naive + self.r_ptfh * proliferation_onset * np.exp(-self.alpha_ptfh * t) * pre_tfh - self.d_ptfh * pre_tfh - self.delta_ptfh * pre_tfh
        dydt_tfh = self.d_ptfh * pre_tfh + self._tfh_growth(t, tfh) - self.d_tfh * tfh
        dydt_inter_tfh = self.d_tfh * tfh + self.r_intertfh * proliferation_onset * np.exp(-self.alpha_inter_tfh * t) * inter_tfh - self.delta_inter_tfh * inter_tfh
        return (
            dydt_naive,
            dydt_early_activated,
            dydt_inter_th0,
            dydt_th0,
            dydt_pth1,
            dydt_inter_th1,
            dydt_th1,
            dydt_ptfh,
            dydt_inter_tfh,
            dydt_tfh,
        )

    def ode_eff_int_pre(
        self,
        t: float,
        state: Sequence[float],
    ) -> tuple[float, ...]:
        """Effector to intermediate to precursor differentiation."""
        (
            naive,
            early_activated,
            inter_th0,
            th0,
            pre_th1,
            inter_th1,
            th1,
            pre_tfh,
            inter_tfh,
            tfh,
        ) = state
        proliferation_onset = self._proliferation_onset(t)
        dydt_naive = -self.d_naive * naive
        dydt_th0 = self.d_naive * self.p_th0 * naive - self.d_th0 * th0
        dydt_inter_th0 = self.d_th0 * th0 + self.r_inter_th0 * proliferation_onset * np.exp(-self.alpha_inter_th0 * t) * inter_th0 - self.d_inter_th0 * inter_th0 - self.delta_inter_th0 * inter_th0
        dydt_early_activated = self.d_inter_th0 * inter_th0 + self.r_early_activated * proliferation_onset * np.exp(-self.alpha * t) * early_activated - self.delta * early_activated
        dydt_th1 = self.d_naive * self.p_th1 * naive - self.d_th1 * th1
        dydt_inter_th1 = self.d_th1 * th1 + self.r_interth1 * proliferation_onset * np.exp(-self.alpha_inter_th1 * t) * inter_th1 - self.d_interth1 * inter_th1 - self.delta_inter_th1 * inter_th1
        dydt_pth1 = self.d_interth1 * inter_th1 + self.r_pth1 * proliferation_onset * np.exp(-self.alpha_pth1 * t) * pre_th1 - self.delta_pth1 * pre_th1
        dydt_tfh = self.d_naive * self.p_tfh * naive + self._tfh_growth(t, tfh) - self.d_tfh * tfh
        dydt_inter_tfh = self.d_tfh * tfh + self.r_intertfh * proliferation_onset * np.exp(-self.alpha_inter_tfh * t) * inter_tfh - self.d_intertfh * inter_tfh - self.delta_inter_tfh * inter_tfh
        dydt_ptfh = self.d_intertfh * inter_tfh + self.r_ptfh * proliferation_onset * np.exp(-self.alpha_ptfh * t) * pre_tfh - self.delta_ptfh * pre_tfh
        return (
            dydt_naive,
            dydt_early_activated,
            dydt_inter_th0,
            dydt_th0,
            dydt_pth1,
            dydt_inter_th1,
            dydt_th1,
            dydt_ptfh,
            dydt_inter_tfh,
            dydt_tfh,
        )

    def ode_eff_pre_int(
        self,
        t: float,
        state: Sequence[float],
    ) -> tuple[float, ...]:
        """Effector to precursor to intermediate differentiation."""
        (
            naive,
            early_activated,
            inter_th0,
            th0,
            pre_th1,
            inter_th1,
            th1,
            pre_tfh,
            inter_tfh,
            tfh,
        ) = state
        proliferation_onset = self._proliferation_onset(t)
        dydt_naive = -self.d_naive * naive
        dydt_th0 = self.d_naive * self.p_th0 * naive - self.d_th0 * th0
        dydt_early_activated = self.d_th0 * th0 + self.r_early_activated * proliferation_onset * np.exp(-self.alpha * t) * early_activated - self.d_early_activated * early_activated - self.delta * early_activated
        dydt_inter_th0 = self.d_early_activated * early_activated + self.r_inter_th0 * proliferation_onset * np.exp(-self.alpha_inter_th0 * t) * inter_th0 - self.delta_inter_th0 * inter_th0
        dydt_th1 = self.d_naive * self.p_th1 * naive - self.d_th1 * th1
        dydt_pth1 = self.d_th1 * th1 + self.r_pth1 * proliferation_onset * np.exp(-self.alpha_pth1 * t) * pre_th1 - self.d_pth1 * pre_th1 - self.delta_pth1 * pre_th1
        dydt_inter_th1 = self.d_pth1 * pre_th1 + self.r_interth1 * proliferation_onset * np.exp(-self.alpha_inter_th1 * t) * inter_th1 - self.delta_inter_th1 * inter_th1
        dydt_tfh = self.d_naive * self.p_tfh * naive + self._tfh_growth(t, tfh) - self.d_tfh * tfh
        dydt_ptfh = self.d_tfh * tfh + self.r_ptfh * proliferation_onset * np.exp(-self.alpha_ptfh * t) * pre_tfh - self.d_ptfh * pre_tfh - self.delta_ptfh * pre_tfh
        dydt_inter_tfh = self.d_ptfh * pre_tfh + self.r_intertfh * proliferation_onset * np.exp(-self.alpha_inter_tfh * t) * inter_tfh - self.delta_inter_tfh * inter_tfh
        return (
            dydt_naive,
            dydt_early_activated,
            dydt_inter_th0,
            dydt_th0,
            dydt_pth1,
            dydt_inter_th1,
            dydt_th1,
            dydt_ptfh,
            dydt_inter_tfh,
            dydt_tfh,
        )

    def simulate(
        self,
        t_eval: Sequence[float],
        parameter_values: Mapping[str, float] | Any | None = None,
        initial_state: Mapping[str, float] | Sequence[float] | None = None,
        method: str = "LSODA",
        **kwargs: Any,
    ):
        if parameter_values is not None:
            self.set_parameters(parameter_values)
        if not self.params:
            raise ValueError("Model parameters are not set.")

        times = np.asarray(t_eval, dtype=float)
        y0 = (
            self.initial_state
            if initial_state is None
            else self._coerce_initial_state(initial_state)
        )
        time_span = (min(0.0, float(times.min())), float(times.max()))

        return solve_ivp(
            self.rhs,
            time_span,
            y0,
            t_eval=times,
            method=method,
            **kwargs,
        )

    def get_state_trajectories(
        self,
        t_eval: Sequence[float],
        parameter_values: Mapping[str, float] | Any | None = None,
        initial_state: Mapping[str, float] | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        solution = self.simulate(
            t_eval,
            parameter_values,
            initial_state,
            **kwargs,
        )
        trajectories = pd.DataFrame(solution.y.T, columns=self.state_names)
        trajectories.insert(0, "time", solution.t)
        return trajectories

    def get_observables(
        self,
        t_eval: Sequence[float],
        parameter_values: Mapping[str, float] | Any | None = None,
        observable_map: Mapping[str, Sequence[str]] | None = None,
        initial_state: Mapping[str, float] | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        states = self.get_state_trajectories(
            t_eval,
            parameter_values,
            initial_state,
            **kwargs,
        )
        result = pd.DataFrame({"time": states["time"]})
        selected_observables = observable_map or self.observable_map
        for name, members in selected_observables.items():
            result[name] = states[list(members)].sum(axis=1)
        return result
