"""Data-loading utilities for the socio-climate model."""

from pathlib import Path

import pandas as pd


DEFAULT_EMISSIONS_CSV = Path(__file__).resolve().parent / "global.1751_2017.csv"


def load_emission_rate(csv_path=DEFAULT_EMISSIONS_CSV):
    """Load and preprocess the historical emissions series.

    The transformation exactly follows the original code:
    1. use the second CSV column,
    2. coerce non-numeric values to NaN and drop them,
    3. discard the first 50 numeric observations,
    4. scale by 10^3.
    """
    frame = pd.read_csv(csv_path)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").dropna().to_numpy()
    return values[50:] / 1e3
