# src/strategy_selector.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except Exception:
    yaml = None

Strategy = Dict[str, Any]

# Known condition keys we accept in strategy files
COND_KEYS = {
    "adx_min", "adx_max",
    "vol_ratio_min", "vol_ratio_max",
    "atr_min", "atr_max",
    "impact_min", "impact_max",
    "session",
}

# ---------------- I/O ----------------

def _load_yaml(p: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(f"YAML strategy found but PyYAML not installed: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _load_json(p: Path) -> Dict[str, Any]:
    return json.load(open(p, "r", encoding="utf-8"))

def _normalize_push(bank: List[Strategy], s: Dict[str, Any], name_fallback: str):
    s.setdefault("name", s.get("id", name_fallback))
    s.setdefault("type", "generic")
    s.setdefault("conditions", {})
    s.setdefault("exit", {})
    s.setdefault("confidence_boost", 1.0)
    bank.append(s)

def _looks_like_strategy(obj: Dict[str, Any], stem: str) -> bool:
    # skip obvious non-strategy files
    bad = {"logo", "readme", "license", "setup", "manifest",
           "package", "pyproject", "config", "requirements", "changelog"}
    if stem.lower() in bad:
        return False
    cond = obj.get("conditions", {})
    if not isinstance(cond, dict):
        return False
    # must contain at least one recognized condition key
    if not any(k in cond for k in COND_KEYS):
        return False
    return True

def load_strategy_bank(bank_root: str = "strategy_bank",
                       subdirs: List[str] | None = None) -> List[Strategy]:
    """
    Load strategies from:
      strategy_bank/normalized_yaml/*.yaml|*.yml
      strategy_bank/*.yaml|*.yml|*.json
      strategy_bank/external_raw/*.yaml|*.yml|*.json
    """
    root = Path(bank_root)
    patterns: List[Path] = []
    if subdirs:
        for sd in subdirs:
            patterns += [root/sd/"*.yaml", root/sd/"*.yml", root/sd/"*.json"]
    else:
        patterns = [
            root/"normalized_yaml/*.yaml",
            root/"normalized_yaml/*.yml",
            root/"*.yaml",
            root/"*.yml",
            root/"*.json",
            root/"external_raw/*.yaml",
            root/"external_raw/*.yml",
            root/"external_raw/*.json",
        ]

    bank: List[Strategy] = []
    for pat in patterns:
        for p in Path().glob(str(pat)):
            try:
                obj = _load_yaml(p) if p.suffix.lower() in (".yaml", ".yml") else _load_json(p)
                if isinstance(obj, dict):
                    if _looks_like_strategy(obj, p.stem):
                        _normalize_push(bank, obj, p.stem)
                elif isinstance(obj, list):
                    for it in obj:
                        if isinstance(it, dict) and _looks_like_strategy(it, p.stem):
                            _normalize_push(bank, it, p.stem)
            except Exception:
                # ignore malformed files
                pass
    return bank

# ---------------- Matching ----------------

def _in_range(val: float, lo: float | None, hi: float | None) -> bool:
    if lo is not None and val < lo: return False
    if hi is not None and val > hi: return False
    return True

def _score_fit(cand: Dict[str, Any], cond: Dict[str, Any]) -> float:
    """
    Soft fit score. Partial mismatches are penalized.
    Supported condition keys:
      adx_min/max, vol_ratio_min/max, atr_min/max, impact_min/max, session
    """
    adx = float(cand.get("adx", 0.0))
    vol = float(cand.get("vol_ratio", 1.0))
    atr = float(cand.get("atr", 0.0))
    impact = float(cand.get("impact_score", 0.0))
    session = str(cand.get("session", "")).upper()

    score = 0.0
    for key, val in (("adx", adx), ("vol_ratio", vol), ("atr", atr), ("impact", impact)):
        lo = cond.get(f"{key}_min", None)
        hi = cond.get(f"{key}_max", None)
        if lo is None and hi is None:
            continue
        if _in_range(val, lo, hi):
            score += 1.0
        else:
            d = 0.0
            if lo is not None and val < lo: d = lo - val
            if hi is not None and val > hi: d = val - hi
            score -= 0.25 * d

    sess_req = cond.get("session", None)
    if sess_req:
        if isinstance(sess_req, (list, tuple, set)):
            score += 0.5 if session in {str(s).upper() for s in sess_req} else -0.5
        else:
            score += 0.5 if session == str(sess_req).upper() else -0.5

    return float(score)

def pick_best_strategy(cand: Dict[str, Any], bank: List[Strategy]) -> Tuple[Strategy | None, float]:
    if not bank:
        return None, 0.0
    best, best_fit = None, -1e12
    for s in bank:
        fit = _score_fit(cand, s.get("conditions", {}))
        if fit > best_fit:
            best, best_fit = s, fit
    return best, best_fit
