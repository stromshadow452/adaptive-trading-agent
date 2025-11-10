# utils/feature_align.py
from __future__ import annotations
from typing import Mapping, Any
import numpy as np

try:
    import pandas as pd  # noqa: F401
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _is_num(x: Any) -> bool:
    try:
        xf = float(x)
        return np.isfinite(xf)
    except Exception:
        return False


def _as_mapping(row: Any) -> Mapping[str, float]:
    """
    Normalize a single-row features object into a {name: float(value)} mapping.
    Accepts dict-like, pandas.Series, or any object exposing .items().
    Missing/invalid values are coerced to 0.0.
    """
    if row is None:
        return {}

    # pandas.Series
    if pd is not None:
        try:
            import pandas as _pd  # local to avoid global hard dep
            if isinstance(row, _pd.Series):
                return {str(k): float(v) if _is_num(v) else 0.0 for k, v in row.to_dict().items()}
        except Exception:
            pass

    # dict-like
    try:
        return {str(k): float(v) if _is_num(v) else 0.0 for k, v in dict(row).items()}
    except Exception:
        return {}


def align_features_for_lgb(live_features_row: Any, booster) -> np.ndarray:
    """
    Align a single-row feature mapping to the LightGBM Booster's trained order.
    Returns shape (1, n_features_trained); missing -> 0.0; extras ignored.
    """
    try:
        trained_names = list(booster.feature_name())
    except Exception:
        mp = _as_mapping(live_features_row)
        vals = [float(v) for v in mp.values()]
        return np.asarray([vals], dtype=float).reshape(1, -1)

    live_map = _as_mapping(live_features_row)
    aligned = []
    for name in trained_names:
        v = live_map.get(name, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        aligned.append(v)
    return np.asarray([aligned], dtype=float)
