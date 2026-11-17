from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_VARIANTS = {
    "int_pre_eff",
    "pre_int_eff",
    "int_eff_pre",
    "pre_eff_int",
    "eff_int_pre",
    "eff_pre_int",
}

DEFAULT_FIT_OBSERVABLE_MAP: dict[str, tuple[str, ...]] = {
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


def observable_map_for_variant(model_variant: str) -> dict[str, tuple[str, ...]]:
    if model_variant in MODEL_VARIANTS:
        return dict(DEFAULT_FIT_OBSERVABLE_MAP)

    known_variants = ", ".join(sorted(MODEL_VARIANTS))
    raise ValueError(
        f"Unknown model variant '{model_variant}'. Known variants: {known_variants}"
    )


class PreparedData:
    """Load and prepare averaged absolute-count data for model fitting."""

    COMBINED_OBSERVABLE_ALIASES: dict[str, str] = {
        "Ki-67+ Tbet+ Bcl6-": "Ki-67+ Tbet+ / inter Bcl6-",
        "Ki-67+ Tbet inter Bcl6-": "Ki-67+ Tbet+ / inter Bcl6-",
        "Ki-67 inter Tbet+ Bcl6-": "Ki-67 inter Tbet+ / inter Bcl6-",
        "Ki-67 inter Tbet inter Bcl6-": "Ki-67 inter Tbet+ / inter Bcl6-",
        "Ki-67- Tbet+ Bcl6-": "Ki-67- Tbet+ / inter Bcl6-",
        "Ki-67- Tbet inter Bcl6-": "Ki-67- Tbet+ / inter Bcl6-",
    }

    RAW_NAME_ALIASES: dict[str, str] = {
        "Ki-67inter Tbet+ Bcl6+": "Ki-67 inter Tbet+ Bcl6+",
    }

    def __init__(
        self,
        base_dir: str | Path | None = None,
        model_variant: str = "int_pre_eff",
    ) -> None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent / "data"
        self.summary_path = Path(base_dir) / "ki67_analysis_detail_absolute_counts_summary.csv"
        self.model_variant = model_variant

    def load_summary_data(self) -> pd.DataFrame:
        return pd.read_csv(self.summary_path)

    def build_fit_dataframe(self) -> pd.DataFrame:
        data = self.load_summary_data().copy()
        data = data[data["condition"] == "three_cond"].copy()
        data = data[data["subpopulation"] != "cell count LN [in mio]"].copy()
        data["subpopulation"] = data["subpopulation"].replace(
            self.RAW_NAME_ALIASES
        )
        data = data.rename(
            columns={
                "subpopulation": "observable",
                "mean_absolute_count": "data_value",
                "sem_absolute_count": "data_sem",
                "day": "fit_day",
            }
        )
        data["fit_day"] = pd.to_numeric(data["fit_day"], errors="coerce")
        data["data_value"] = pd.to_numeric(data["data_value"], errors="coerce")
        data["data_sem"] = pd.to_numeric(data["data_sem"], errors="coerce")
        data = data.dropna(subset=["fit_day", "data_value"])

        # Day 0 determines the initial state and is not included in the residual.
        data = data[data["fit_day"] != 0].copy()
        data["source_dataset"] = "three_cond_summary"

        data["observable"] = data["observable"].replace(
            self.COMBINED_OBSERVABLE_ALIASES
        )
        data = (
            data.groupby(
                ["source_dataset", "condition", "sheet", "fit_day", "observable"],
                as_index=False,
                dropna=False,
            )
            .agg(
                data_value=("data_value", "sum"),
                data_sem=(
                    "data_sem",
                    lambda values: float(
                        np.sqrt(np.nansum(np.square(values.astype(float))))
                    ),
                ),
            )
        )

        # Terminal populations are not measured reliably at day 1.
        terminal_observables = {
            "Ki-67- Tbet- Bcl6-",
            "Ki-67- Tbet+ / inter Bcl6-",
            "Ki-67- Tbet- Bcl6+",
        }
        exclude_day_one = (data["fit_day"] == 1) & data["observable"].isin(
            terminal_observables
        )
        data = data[~exclude_day_one].copy()

        columns = [
            "source_dataset",
            "condition",
            "sheet",
            "fit_day",
            "observable",
            "data_value",
            "data_sem",
        ]
        return data[columns].sort_values(
            ["observable", "fit_day"],
            ignore_index=True,
        )
