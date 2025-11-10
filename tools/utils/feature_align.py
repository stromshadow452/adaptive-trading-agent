from __future__ import annotations

from typing import Mapping, Any
import numpy as np

try:
    import pandas as pd  # optional, only for Series handling
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
            import pandas as _pd  # noqa
            if isinstance(row, _pd.Series):
                return {str(k): float(v) if _is_num(v) else 0.0 for k, v in row.to_dict().items()}
        except Exception:
            pass

    # generic dict-like
    try:
        return {str(k): float(v) if _is_num(v) else 0.0 for k, v in dict(row).items()}
    except Exception:
        return {}


def align_features_for_lgb(live_features_row: Any, booster) -> np.ndarray:
    """
    Align a single-row feature mapping to the LightGBM Booster's trained order.

    Parameters
    ----------
    live_features_row : dict-like or pandas.Series
        A mapping {feature_name: value} for ONE sample (your live row).
    booster : lightgbm.Booster
        Trained booster. Must expose .feature_name().

    Returns
    -------
    np.ndarray
        Shape (1, n_features_trained) aligned to booster.feature_name() order.
        Missing columns -> 0.0; extra columns are ignored.
    """
    # fetch trained feature order from booster
    try:
        trained_names = list(booster.feature_name())
    except Exception:
        # fall back: no names; return 2D array from whatever we got
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

    # ensure proper 2D shape for LightGBM
    x = np.asarray([aligned], dtype=float)
    return x
