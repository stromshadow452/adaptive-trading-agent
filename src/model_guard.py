# src/model_guard.py
"""Model metadata loader & guard + writer.

Usage:
  from src.model_guard import (
      feature_list_hash,
      load_sidecar_meta,
      assert_feature_names,
      write_model_metadata,
  )
"""

from __future__ import annotations
import json
import hashlib
import os
from typing import List, Dict, Any, Tuple


def feature_list_hash(names: List[str]) -> str:
    txt = "\n".join(names)
    return "sha256:" + hashlib.sha256(txt.encode("utf-8")).hexdigest()


def sidecar_path_for(model_path: str) -> str:
    base, ext = os.path.splitext(model_path)
    return f"{base}_metadata.json"


def load_sidecar_meta(model_path: str, meta_path: str | None = None) -> Dict[str, Any]:
    """Load sidecar metadata JSON (UTF-8 with/without BOM)."""
    mp = meta_path or sidecar_path_for(model_path)
    with open(mp, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def assert_feature_names(expected: List[str], actual: List[str]) -> None:
    """Raise clear error if order/content mismatch."""
    if expected == actual:
        return
    missing = [x for x in expected if x not in actual]
    extra   = [x for x in actual if x not in expected]
    raise ValueError(
        "Feature mismatch between model metadata and live features.\n"
        f"Expected ({len(expected)}): {expected}\n"
        f"Actual   ({len(actual)}): {actual}\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}\n"
        f"Expected-hash: {feature_list_hash(expected)}\n"
    )


def write_model_metadata(
    model_path: str,
    model_name: str,
    feature_names: List[str],
    version: str,
    extra: Dict[str, Any] | None = None,
) -> str:
    """Write sidecar metadata next to model: <model>_metadata.json"""
    meta = {
        "model_name": model_name,
        "version": version,
        "feature_names": feature_names,
        "feature_order_hash": feature_list_hash(feature_names),
    }
    if extra:
        meta.update(extra)
    out_path = sidecar_path_for(model_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_path
