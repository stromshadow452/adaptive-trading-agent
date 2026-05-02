#!/usr/bin/env python3

# src/decision_engine.py

"""

Decision Engine - hybrid merge of screener, meta-selector and (optional) finrl policies.



Usage:

    from src.decision_engine import main

    main(args)  # OR run as module via the tools runner that imports main



This module is intentionally defensive and self-contained;

it doesn't require FinRL to be present to run basic hybrid logic.

"""



from __future__ import annotations



import os

import json

import glob

import argparse

from datetime import datetime, timezone

import logging

from typing import Optional, Dict, Any, List, Tuple

import math

from pathlib import Path



# new imports for execution patch

import csv

from typing import NamedTuple

from utils.finrl_loader import load_finrl_policies

from utils.finrl_adapter import finrl_signal as finrl_signal_adapter  # alias to avoid local name clash

from tools.policy_adapter import finrl_signal_adapter, _unwrap_estimator, SklearnPolicyAdapter
# ---- ADDITIVE: finrl audit helper (pure) ----
def finrl_audit_call(candidate, policy, finrl_cfg, sym, logger):
    est = _unwrap_estimator(policy)
    # best-effort logger + stdout fallback (never break flow)
    def _emit(obj):
        try:
            logger(obj)
        except Exception:
            pass
        try:
            print(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    try:
        _emit({
            "stage": "finrl_signal_audit",
            "symbol": sym,
            "est_type": type(est).__name__,
            "has_proba": hasattr(est, "predict_proba"),
            "has_predict": hasattr(est, "predict"),
        })
    except Exception:
        pass

    side, conf, meta = finrl_signal_adapter(candidate, policy, finrl_cfg)

    try:
        _emit({
            "stage": "finrl_signal",
            "symbol": sym,
            "side": side,
            "conf": conf,
            "source": meta.get("source") if isinstance(meta, dict) else None,
            "feat_dim": meta.get("feat_dim") if isinstance(meta, dict) else None,
        })
    except Exception:
        pass

    return side, conf, meta
# ---- END ADDITIVE: finrl audit helper ----




# Optional deps

try:

    import joblib  # type: ignore

except Exception:  # pragma: no cover

    joblib = None



try:

    import yaml  # type: ignore

except Exception:  # pragma: no cover

    yaml = None



try:

    import numpy as np  # type: ignore

except Exception:  # pragma: no cover

    np = None



# -----------------------------------------------------------------------------

# Logging

# -----------------------------------------------------------------------------

LOG = logging.getLogger("decision_engine")

if not LOG.handlers:

    LOG.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    LOG.addHandler(handler)



# -----------------------------------------------------------------------------

# Helpers

# -----------------------------------------------------------------------------

def safe_float(x: Any, default: float = 0.0) -> float:

    try:

        v = float(x)

        if math.isnan(v) or math.isinf(v):

            return default

        return v

    except Exception:

        return default



def safe_int(x: Any, default: int = 0) -> int:

    try:

        return int(x)

    except Exception:

        return default



# ---- ADDITIVE: meta sanitizer (cheap, deterministic) ----

def _plain(x, _depth=0):

    # fast exits

    if x is None or isinstance(x, (bool, int, float, str)):

        return x

    # numpy scalar

    if hasattr(x, "item") and not hasattr(x, "__len__"):

        try:

            return x.item()

        except Exception:

            return str(x)

    # numpy array / pandas objects implementing tolist

    if hasattr(x, "tolist"):

        try:

            return x.tolist()

        except Exception:

            return str(x)

    # containers (cap depth to avoid pathological objects)

    if _depth < 3 and isinstance(x, dict):

        return {str(k): _plain(v, _depth + 1) for k, v in x.items()}

    if _depth < 3 and isinstance(x, (list, tuple)):

        return [_plain(v, _depth + 1) for v in x]

    return str(x)



# --- Universe filters (PATCHED & EXPANDED) -----------------------------------

FX_CCY       = {"USD","EUR","JPY","GBP","AUD","NZD","CAD","CHF"}

CRYPTO_BASES = {"BTC","ETH","SOL","XRP","BNB","ADA","DOGE","DOT","LTC","XMR","AVAX","SHIB","TRX","LINK","UNI","MATIC"}

METALS       = {"XAU","XAG"}



def _normalize_symbol_for_universe(sym: str) -> str:

    s = (sym or "").upper().strip()

    for ch in ("-", "_", "/"):

        s = s.replace(ch, "")

    # normalize USDT → USD for crypto pairs

    s = s.replace("USDT", "USD")

    return s



def _classify_symbol(sym: str) -> str:

    s = _normalize_symbol_for_universe(sym)

    if len(s) < 6:

        return "other"

    base, quote = s[:3], s[3:6]

    if base in CRYPTO_BASES:

        return "crypto"

    if (base in FX_CCY or base in METALS) and (quote in FX_CCY):

        return "forex"

    return "other"



def in_universe(symbol: str, universe: str | None, exclude: list[str] | None) -> bool:

    s = _normalize_symbol_for_universe(symbol)

    if exclude and s in { _normalize_symbol_for_universe(x) for x in exclude }:

        return False

    if not universe:

        return True

    kind = _classify_symbol(s)

    u = (universe or "").replace(" ", "").lower()

    if u in {"fx+crypto", "forex,crypto", "crypto,forex"}:

        return kind in {"forex","crypto"}

    if u == "forex":

        return kind == "forex"

    if u == "crypto":

        return kind == "crypto"

    return True



# -----------------------------------------------------------------------------

# (PATCH) — MTF helpers

# -----------------------------------------------------------------------------

def _infer_side_from_score(s: float) -> str:

    return "buy" if s >= 0 else "sell"



def mtf_consensus(symbol: str, tf: str, siblings: list[dict], mtf_cfg: dict) -> tuple[str, dict]:

    """

    Return (side|'hold', debug) based on primary+confirm TFs alignment.

    siblings: all candidates for same symbol (various TFs)

    """

    if not mtf_cfg.get("enabled", False):

        me = next((c for c in siblings if (c.get("tf") or "").upper() == (tf or "").upper()), None)

        if not me:

            return "hold", {"reason": "no_primary"}

        s = safe_float(me.get("score"), 0.0)

        return _infer_side_from_score(s), {"reason": "single_tf"}



    primary_tf = (mtf_cfg.get("primary_tf") or "M15").upper()

    confirm_tfs = [t.upper() for t in (mtf_cfg.get("confirm_tfs") or [])]

    agree_min = float(mtf_cfg.get("agree_min", 0.55))

    abstain_gap = float(mtf_cfg.get("abstain_gap", 0.15))



    pri = next((c for c in siblings if (c.get("tf") or "").upper() == primary_tf), None)

    if not pri:

        return "hold", {"reason": "no_primary_present"}



    pri_score = safe_float(pri.get("score"), 0.0)

    pri_side = _infer_side_from_score(pri_score)



    votes: List[str] = []

    for ct in confirm_tfs:

        cc = next((c for c in siblings if (c.get("tf") or "").upper() == ct), None)

        if not cc:

            continue

        sc = safe_float(cc.get("score"), 0.0)

        votes.append(_infer_side_from_score(sc))



    if not votes:

        return pri_side, {"reason": "no_confirms", "primary": pri_side, "pri_score": pri_score}



    agree = sum(1 for v in votes if v == pri_side) / len(votes)

    opp = "sell" if pri_side == "buy" else "buy"

    opp_votes = sum(1 for v in votes if v == opp) / len(votes)



    if opp_votes - agree >= abstain_gap:

        return "hold", {"reason": "conflict_abstain", "agree": agree, "opp": opp_votes}



    if agree >= agree_min:

        return pri_side, {"reason": "confirmed", "agree": agree, "votes": votes}



    return "hold", {"reason": "not_enough_agree", "agree": agree, "votes": votes}



# -----------------------------------------------------------------------------

# I/O helpers

# -----------------------------------------------------------------------------

def find_latest_screener(folder: str = "reports/screener", pattern: str = "*_intraday.json") -> Optional[str]:

    path = os.path.join(folder, pattern)

    files = sorted(glob.glob(path), key=os.path.getmtime)

    if not files:

        files = sorted(glob.glob(os.path.join(folder, "*_swing.json")), key=os.path.getmtime)

    return files[-1] if files else None



def load_candidates(path: str) -> List[Dict[str, Any]]:

    if not path:

        raise ValueError("No candidate path provided")

    LOG.info("Loading candidates from: %s", path)

    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

        if isinstance(data, dict) and "candidates" in data:

            return data["candidates"]

        if isinstance(data, list):

            return data

        raise ValueError("Unexpected candidate file structure")



def load_meta_selector(path: str = "models/meta_selector/meta_selector.joblib"):

    if not joblib:

        LOG.warning("joblib not available, skipping meta selector load.")

        return None

    if not os.path.exists(path):

        LOG.info("meta_selector not found at %s, continuing without it.", path)

        return None

    try:

        model = joblib.load(path)

        LOG.info("Loaded meta selector from %s", path)

        return model

    except Exception as e:

        LOG.exception("Failed to load meta selector: %s", e)

        return None



def load_risk_config(path: str = "config/decision.yaml") -> Dict[str, Any]:

    """

    Loads YAML and merges with safe defaults.

    Accepts either a flat YAML with keys or a root 'decision:' mapping.

    """

    defaults = {

        # entry/selectivity

        "min_score_to_enter": 0.20,

        "min_confidence": 0.0,



        # sizing

        "risk_per_trade": 0.01,

        "default_equity": 10000.0,

        "stop_atr_mult": 1.5,

        "takeprofit_rr": 1.5,

        "min_size": 0.001,

        "max_size": 0.05,



        # pct fallbacks if ATR unavailable

        "sl_pct": 0.01,

        "tp_pct": 0.02,



        # --- 2-layer fusion thresholds ---

        "use_two_layer": True,

        "stats_conf_min": 0.60,

        "rl_conf_min": 0.60,

        "agree_min": 0.55,

        "abstain_gap": 0.15,

    }

    if not yaml:

        LOG.info("PyYAML not installed; using default risk config.")

        return defaults

    if os.path.exists(path):

        try:

            with open(path, "r", encoding="utf-8") as f:

                cfg = yaml.safe_load(f) or {}

            merged = defaults.copy()

            merged.update(cfg.get("decision", cfg) or {})

            LOG.info("Loaded risk config from %s", path)

            return merged

        except Exception as e:

            LOG.exception("Failed to load decision.yaml, using defaults: %s", e)

            return defaults

    LOG.info("No decision.yaml found, using defaults.")

    return defaults



# -----------------------------------------------------------------------------

# Kelly sizing (FIXED & TOP-LEVEL)

# -----------------------------------------------------------------------------

def compute_kelly_size(

    equity: float,

    entry_price: float,

    sl_price: float,

    tp_price: float,

    final_conf: float,              # p in [0,1]

    k_frac: float = 0.25,           # fractional Kelly

    max_pct_equity: float = 0.01,   # hard cap (1% default)

    pip_value_per_stdlot: float = 10.0,

    pip_scale: float = 0.0001,

    min_lots: float = 0.01,

    max_lots: float = 10.0

) -> Tuple[float, Dict[str, Any]]:

    """Deterministic fractional-Kelly position sizing (returns lots, meta)."""

    # 1) distances (absolute)

    risk = abs(entry_price - sl_price)

    reward = abs(tp_price - entry_price)

    if risk <= 0 or reward <= 0:

        return 0.0, {"reason": "invalid_sl_or_tp"}



    # 2) convert to pips (numeric)

    pips_risk = risk / pip_scale

    pips_reward = reward / pip_scale

    R = (pips_reward / pips_risk) if pips_risk > 0 else 0.0



    # 3) raw Kelly

    p = max(0.0, min(1.0, float(final_conf)))

    fstar = (p * (R + 1.0) - 1.0) / R if R > 0 else 0.0

    fstar = max(fstar, 0.0)



    # 4) fractional Kelly and caps

    f = k_frac * fstar

    max_risk_amount = equity * max_pct_equity

    risk_amount = min(f * equity, max_risk_amount)

    if risk_amount <= 0:

        return 0.0, {"reason": "kelly_zero_or_capped", "fstar": fstar, "f": f}



    # 5) risk per standard lot (dollars)

    risk_per_stdlot = pips_risk * pip_value_per_stdlot

    if risk_per_stdlot <= 0:

        return 0.0, {"reason": "invalid_risk_per_stdlot"}



    # 6) lots

    lots_unclamped = risk_amount / risk_per_stdlot

    lots = max(0.0, min(lots_unclamped, max_lots))

    steps = int(lots / min_lots)

    lots = steps * min_lots



    meta = {

        "pips_risk": pips_risk,

        "pips_reward": pips_reward,

        "R": R,

        "p": p,

        "fstar": fstar,

        "f_fractional": f,

        "risk_amount": risk_amount,

        "risk_per_stdlot": risk_per_stdlot,

        "lots_unrounded": lots_unclamped,

        "lots_final": lots

    }

    return lots, meta



# -----------------------------------------------------------------------------

# Optional FinRL detection (presence boosts confidence slightly)

# -----------------------------------------------------------------------------

def try_load_finrl_policy(dir_path: str, symbol: str, tf: str | None = None):

    """

    Back-compat shim: accepts optional tf but still loads the policy file.

    Prefer using utils.load_finrl_policies() once outside the loop.

    """

    import os, joblib

    # honor tf if files are saved as SYMBOL_TF_policy.joblib

    cand_names = [

        f"{symbol}_M15_policy.joblib" if tf is None else f"{symbol}_{tf}_policy.joblib",

        f"{symbol}_policy.joblib",

    ]

    for name in cand_names:

        fpath = os.path.join(dir_path, name)

        if os.path.exists(fpath):

            return joblib.load(fpath)

    return None



# -----------------------------------------------------------------------------

# Meta weight

# -----------------------------------------------------------------------------

def compute_meta_weight(model, features: Dict[str, Any]) -> float:

    """

    Use meta selector model to compute a weight in [0,1] expressing 'trust' in technical signals.

    If model missing or predict_proba not supported, fall back to 0.5.

    """

    if model is None:

        return 0.5

    try:

        keys = ["score", "vol_ratio", "rsi14", "atr14", "adx_proxy", "impact", "surprise"]

        X = []

        for k in keys:

            v = features.get(k)

            X.append(safe_float(v, 0.0))

        if np is None:

            vals = [x for x in X if x > 0]

            return min(1.0, max(0.0, (sum(vals) / (len(vals) or 1)) * 0.1))

        X = np.array(X, dtype=float).reshape(1, -1)

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(X)[0]

            return float(proba[1]) if len(proba) >= 2 else float(proba[0])

        if hasattr(model, "predict"):

            pred = model.predict(X)[0]

            return float(pred)

        return 0.5

    except Exception:

        LOG.exception("meta selector prediction failed; using 0.5")

        return 0.5



# -----------------------------------------------------------------------------

# Two-layer signals (Stats + FinRL) and fusion

# -----------------------------------------------------------------------------

def _sigmoid(x: float) -> float:

    try:

        return 1.0 / (1.0 + math.exp(-x))

    except Exception:

        return 0.5



def stats_signal(candidate: Dict[str, Any], meta_w: float) -> Tuple[str, float, Dict[str, Any]]:

    """

    Statistical layer signal.

    Output: (side, confidence in [0,1], rationale)

    """

    score = safe_float(candidate.get("score"), 0.0)

    regime = (candidate.get("regime") or "").lower()

    trend_strength = safe_float(candidate.get("trend_strength"), 0.0)

    rsi14 = safe_float(candidate.get("rsi14"), 50.0)



    base_conf = _sigmoid(abs(score) * 2.0)



    bonus = 0.0

    if regime in ("trend", "trending"):

        bonus += min(0.10, max(0.0, trend_strength * 0.10))

    if 35.0 <= rsi14 <= 65.0 and regime in ("meanrev", "range"):

        bonus += 0.05



    bonus += (meta_w - 0.5) * 0.20  # ±0.10



    conf = max(0.0, min(1.0, base_conf + bonus))

    side = "buy" if score >= 0 else "sell"

    rationale = {

        "score": score,

        "regime": regime,

        "trend_strength": trend_strength,

        "rsi14": rsi14,

        "meta_w": meta_w,

        "base_conf": base_conf,

        "conf": conf

    }

    return side, conf, rationale



def finrl_signal_local(finrl_policy: Any, candidate: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:

    """

    Try to extract an action + confidence from a FinRL policy or candidate RL hints.

    Output: (side in {'buy','sell','hold'}, confidence in [0,1], rationale)

    """

    state = candidate.get("state")

    side = "hold"

    conf = 0.5

    rationale = {"source": "fallback", "details": None}



    # candidate RL direct fields

    if isinstance(candidate.get("rl_action"), str):

        a = candidate["rl_action"].lower()

        if a in ("buy", "long"):

            side = "buy"

        elif a in ("sell", "short"):

            side = "sell"

        else:

            side = "hold"

        conf = max(0.0, min(1.0, safe_float(candidate.get("rl_conf"), 0.5)))

        return side, conf, _plain({"source": "candidate_rl_fields"})



    if "rl_prob_long" in candidate or "rl_prob_short" in candidate:

        p_long = max(0.0, min(1.0, safe_float(candidate.get("rl_prob_long"), 0.0)))

        p_short = max(0.0, min(1.0, safe_float(candidate.get("rl_prob_short"), 0.0)))

        if max(p_long, p_short) < 1e-9:

            side, conf = "hold", 0.5

        else:

            side = "buy" if p_long >= p_short else "sell"

            conf = abs(p_long - p_short) + 0.5 * max(p_long, p_short)

        return side, max(0.0, min(1.0, conf)), _plain({"source": "candidate_rl_probs", "p_long": p_long, "p_short": p_short})



    # model-driven paths

    if finrl_policy is not None:

        try:

            if hasattr(finrl_policy, "predict_proba"):

                proba = finrl_policy.predict_proba([state] if state is not None else [[safe_float(candidate.get("score"), 0.0)]])[0]

                if len(proba) == 3:

                    p_hold, p_long, p_short = map(float, proba)

                elif len(proba) == 2:

                    p_long, p_short = map(float, proba)

                    p_hold = max(0.0, 1.0 - p_long - p_short)

                else:

                    p_hold = 0.5; p_long = 0.25; p_short = 0.25



                if max(p_long, p_short, p_hold) == p_hold:

                    side, conf = "hold", 1.0 - p_hold

                elif p_long >= p_short:

                    side, conf = "buy", max(0.0, min(1.0, p_long - p_short + 0.5*p_long))

                else:

                    side, conf = "sell", max(0.0, min(1.0, p_short - p_long + 0.5*p_short))



                feat_dim = len(state) if isinstance(state, (list, tuple)) else None

                return side, conf, _plain({"source": "predict_proba", "proba": [p_hold, p_long, p_short], "feat_dim": feat_dim})



            if hasattr(finrl_policy, "predict"):

                y = finrl_policy.predict([state] if state is not None else [[safe_float(candidate.get("score"), 0.0)]])

                # robust scalar pick

                y0 = y

                try:

                    if hasattr(y, "ndim") and getattr(y, "size", 0) >= 1:

                        y0 = y[0]

                    elif isinstance(y, (list, tuple)) and len(y) > 0:

                        y0 = y[0]

                except Exception:

                    pass



                try:

                    y0i = int(y0.item() if hasattr(y0, "item") else y0)

                    side = {1: "buy", -1: "sell", 0: "hold"}.get(y0i, "hold")

                except Exception:

                    y0s = str(y0).lower()

                    side = "buy" if y0s in ("buy", "long") else "sell" if y0s in ("sell", "short") else "hold"



                conf = max(0.0, min(1.0, safe_float(candidate.get("rl_conf"), 0.6)))

                return side, conf, _plain({"source": "predict", "y": y0})

        except Exception as e:

            LOG.warning("finrl_signal: policy call failed: %s", e)

            return "hold", 0.55, _plain({"source": "adapter_error", "error": str(e)})



        # presence only

        return "hold", 0.55, _plain({"source": "policy_presence_only"})



    # no policy and no hints

    return "hold", 0.5, _plain({"source": "none"})



def fuse_two_layer(stats: Tuple[str, float, Dict[str, Any]],

                   rl: Tuple[str, float, Dict[str, Any]],

                   rcfg: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:

    """

    Fusion rule:

      - abstain if RL conf ~ 0.5 ± abstain_gap

      - require both sides to match (buy/buy or sell/sell)

      - require conf >= per-layer mins

      - final conf = min(stats_conf, rl_conf), must be >= agree_min

    Returns (final_side, final_conf, fusion_rationale)

    """

    use_two = bool(rcfg.get("use_two_layer", True))

    if not use_two:

        return stats[0], stats[1], {"mode": "stats_only"}



    stats_side, stats_conf, stats_r = stats

    rl_side, rl_conf, rl_r = rl



    abstain_gap = safe_float(rcfg.get("abstain_gap"), 0.15)

    if abs(rl_conf - 0.5) < abstain_gap or rl_side == "hold":

        return "stats_fallback", stats_conf, {"mode": "stats_fallback_rl_uncertain", "stats": stats_r, "rl": rl_r}



    if stats_side != rl_side:

        return "hold", 0.0, {"mode": "disagree", "stats": stats_r, "rl": rl_r}



    stats_min = safe_float(rcfg.get("stats_conf_min"), 0.6)

    rl_min = safe_float(rcfg.get("rl_conf_min"), 0.6)

    if stats_conf < stats_min or rl_conf < rl_min:

        return "hold", 0.0, {"mode": "below_thresholds", "stats_conf": stats_conf, "rl_conf": rl_conf,

                             "stats_min": stats_min, "rl_min": rl_min, "stats": stats_r, "rl": rl_r}



    final_conf = min(stats_conf, rl_conf)

    agree_min = safe_float(rcfg.get("agree_min"), 0.55)

    if final_conf < agree_min:

        return "hold", 0.0, {"mode": "below_agree_min", "final_conf": final_conf, "agree_min": agree_min,

                             "stats": stats_r, "rl": rl_r}



    return stats_side, final_conf, {"mode": "confirmed", "stats": stats_r, "rl": rl_r, "final_conf": final_conf}



# -----------------------------------------------------------------------------

# Policy (original single-layer helpers)

# -----------------------------------------------------------------------------

def _decide_side(score: float, conf: float, min_score: float, min_conf: float) -> Tuple[str, str]:

    """Return ('buy'|'sell'|'hold', reason)."""

    if abs(score) < min_score or conf < min_conf:

        return "hold", f"below_threshold score={score:.3f} conf={conf:.3f}"

    return ("buy", f"score_pos {score:.3f}") if score > 0 else ("sell", f"score_neg {score:.3f}")



def _size_position(candidate: Dict[str, Any], rcfg: Dict[str, Any]) -> float:

    """

    Risk-based sizing with ATR if present; else percent-stop fallback.

      notional  = equity * risk_per_trade

      stop_dist = ATR*mult  or  price*sl_pct

      size      = notional / (stop_dist * price)

    """

    risk_per_trade = safe_float(rcfg.get("risk_per_trade"), 0.01)

    equity = safe_float(candidate.get("equity"), safe_float(rcfg.get("default_equity"), 10000.0))

    price = safe_float(candidate.get("price"), 0.0)

    atr = safe_float(candidate.get("atr") or candidate.get("atr14"), 0.0)

    stop_mult = safe_float(rcfg.get("stop_atr_mult"), 1.5)

    sl_pct = safe_float(rcfg.get("sl_pct"), 0.01)

    eps = 1e-9



    if price <= eps:

        return 0.0



    if atr > eps:

        stop_dist = max(atr * stop_mult, eps)

    else:

        stop_dist = max(price * sl_pct, eps)



    notional = equity * risk_per_trade

    size = notional / (stop_dist * price)



    size = max(safe_float(rcfg.get("min_size"), 0.001),

               min(safe_float(rcfg.get("max_size"), 0.05), size))

    return max(0.0, size)



def _stops_tp(price: float, atr: float, side: str, rcfg: Dict[str, Any]) -> Tuple[float | None, float | None, str]:

    """Return (sl, tp, sl_type) absolute price levels given ATR-based distances or percent fallback."""

    stop_mult = safe_float(rcfg.get("stop_atr_mult"), 1.5)

    rr = safe_float(rcfg.get("takeprofit_rr"), 1.5)

    eps = 1e-9



    if price <= eps:

        return None, None, "none"



    if atr > eps:

        stop_dist = atr * stop_mult

        tp_dist = stop_dist * rr

        if side == "buy":

            return price - stop_dist, price + tp_dist, "atr"

        if side == "sell":

            return price + stop_dist, price - tp_dist, "atr"



    sl_pct = safe_float(rcfg.get("sl_pct"), 0.01)

    tp_pct = safe_float(rcfg.get("tp_pct"), 0.02)

    if side == "buy":

        return price * (1 - sl_pct), price * (1 + tp_pct), "pct"

    if side == "sell":

        return price * (1 + sl_pct), price * (1 - tp_pct), "pct"

    return None, None, "none"



def decide_action_and_size(candidate: Dict[str, Any], meta_w: float, risk_cfg: Dict[str, Any], finrl_policy_present=False) -> Dict[str, Any]:

    """

    Heuristic decision (single-layer):

     - final_score combines screener score, meta_w and presence of finrl policy

     - threshold check

     - compute position size clipped by min/max

    """

    screener_score = safe_float(candidate.get("score"), 0.0)

    conf = safe_float(candidate.get("confidence"), 1.0)



    finrl_boost = 0.10 if finrl_policy_present else 0.0

    final_score = screener_score * (0.5 + 0.5 * meta_w) + finrl_boost



    dir_guess = candidate.get("direction") or candidate.get("side") or candidate.get("bias")

    if isinstance(dir_guess, str):

        dir_guess = dir_guess.lower()

    if dir_guess in ("sell", "short", "s"):

        implied_side = "sell"

    elif dir_guess in ("buy", "long", "b"):

        implied_side = "buy"

    else:

        implied_side = "buy" if final_score >= 0 else "sell"



    min_entry = safe_float(risk_cfg.get("min_score_to_enter"), 0.20)

    min_conf = safe_float(risk_cfg.get("min_confidence"), 0.0)



    side, reason_txt = _decide_side(final_score, conf, min_entry, min_conf)

    if side == "hold":

        size = 0.0

        sl = tp = None

        sl_type = "none"

    else:

        side = implied_side

        price = safe_float(candidate.get("price"), 0.0)

        atr = safe_float(candidate.get("atr") or candidate.get("atr14"), 0.0)

        size = _size_position(candidate, risk_cfg)

        sl, tp, sl_type = _stops_tp(price, atr, side, risk_cfg)



    return {

        "enter": bool(side in ("buy", "sell")),

        "side": side,

        "final_score": float(final_score),

        "size": float(size),

        "sl": None if sl is None else float(sl),

        "tp": None if tp is None else float(tp),

        "sl_type": sl_type,

        "reason": {

            "screener_score": screener_score,

            "meta_w": meta_w,

            "finrl_boost": finrl_boost,

            "min_entry": min_entry,

            "min_confidence": min_conf,

            "explain": reason_txt,

        }

    }



# -----------------------------------------------------------------------------

# Two-layer decision path

# -----------------------------------------------------------------------------

def decide_action_and_size_two_layer(

    candidate: Dict[str, Any],

    meta_w: float,

    risk_cfg: Dict[str, Any],

    finrl_policy: Any

) -> Dict[str, Any]:

    """

    Two-layer confirmation:

      - stats_signal(candidate, meta_w)

      - finrl_signal_local(policy, candidate)

      - fuse_two_layer(...)

      - if confirmed -> size, stops, etc.

      - else -> hold

    """

    s_side, s_conf, s_r = stats_signal(candidate, meta_w)

    r_side, r_conf, r_r = finrl_signal_local(finrl_policy, candidate)

    side, fused_conf, fusion_r = fuse_two_layer((s_side, s_conf, s_r), (r_side, r_conf, r_r), risk_cfg)



    if side in ("hold", "stats_fallback"):

        if side == "stats_fallback":

            side_sf, conf_sf, r_sf = stats_signal(candidate, meta_w)

            side, fused_conf = side_sf, conf_sf

            fusion_r = {"mode": "stats_only_due_to_rl_uncertain", "stats": r_sf, "rl": {"side": r_side, "conf": r_conf}}

        else:

            return {

                "enter": False, "side": "hold", "final_score": 0.0, "size": 0.0, "sl": None, "tp": None, "sl_type": "none",

                "reason": {"layer": "fusion", "stats": {"side": s_side, "conf": s_conf, **s_r}, "rl": {"side": r_side, "conf": r_conf, **r_r}, "fusion": fusion_r}

            }



    price = safe_float(candidate.get("price"), 0.0)

    atr = safe_float(candidate.get("atr") or candidate.get("atr14"), 0.0)

    size = _size_position(candidate, risk_cfg)

    sl, tp, sl_type = _stops_tp(price, atr, side, risk_cfg)



    return {

        "enter": True,

        "side": side,

        "final_score": fused_conf,  # decision confidence

        "size": float(size),

        "sl": None if sl is None else float(sl),

        "tp": None if tp is None else float(tp),

        "sl_type": sl_type,

        "reason": {

            "layer": "fusion",

            "stats": {"side": s_side, "conf": s_conf, **s_r},

            "rl": {"side": r_side, "conf": r_conf, **r_r},

            "fusion": fusion_r

        }

    }



# -----------------------------------------------------------------------------

# Execution + FinRL-fallback integration (NEW)

# -----------------------------------------------------------------------------



class ExecResult(NamedTuple):

    status: str

    reason: str

    size: float

    sl: float | None

    tp: float | None



# risk check stubs - customize to your rules

def passes_risk_checks(plan: Dict[str, Any], state: Dict[str, Any]) -> bool:

    """

    Basic pre-checks before considering execution.

    - zero size or price -> fail

    - max exposure (example): don't open if size > max_size in config

    """

    price = safe_float(plan.get("price"), 0.0)

    size = safe_float(plan.get("size"), 0.0)

    max_size = safe_float(state.get("risk_cfg", {}).get("max_size", 0.5), 0.5)

    if price <= 0.0:

        LOG.info("passes_risk_checks: price missing/zero.")

        return False

    if size <= 0.0:

        LOG.info("passes_risk_checks: computed size <= 0.")

        return False

    if size > max_size:

        LOG.info("passes_risk_checks: size %.4f > max_size %.4f", size, max_size)

        return False

    return True



def final_risk_checks(size: float, state: Dict[str, Any]) -> bool:

    """

    Final safety checks before executing.

    - ensure size not exceeding absolute cap

    """

    hard_cap = safe_float(state.get("risk_cfg", {}).get("max_size", 0.5), 0.5)

    if size <= 0.0 or size > hard_cap:

        LOG.info("final_risk_checks failed: size=%s hard_cap=%s", size, hard_cap)

        return False

    return True



def compute_sl_from_plan(plan: Dict[str, Any]) -> float | None:

    sl = plan.get("sl")

    if sl:

        try:

            return float(sl)

        except Exception:

            return None

    # fallback: use price - ATR*stop_atr_mult for buy, price+... for sell

    price = safe_float(plan.get("price"), 0.0)

    atr = safe_float(plan.get("atr") or plan.get("atr14"), 0.0)

    rcfg = plan.get("reason", {}).get("rcfg", {}) or {}

    stop_mult = safe_float(rcfg.get("stop_atr_mult"), safe_float(plan.get("stop_atr_mult", 1.5)))

    if atr > 0 and price > 0:

        if plan.get("side") == "buy":

            return price - atr * stop_mult

        if plan.get("side") == "sell":

            return price + atr * stop_mult

    return None



# executor stub - replace with real broker/executor integration

def execute_order(signal: str, symbol: str, size: float, side: str, price: float, sl: float | None, tp: float | None) -> ExecResult:

    LOG.info("EXECUTE_ORDER stub -> %s %s size=%.4f price=%.6f sl=%s tp=%s", signal, symbol, size, price, sl, tp)

    # In production replace with broker adapter; here we pretend a paper fill

    return ExecResult(status="FILLED_PAPER", reason="paper_fill", size=size, sl=sl, tp=tp)



def append_execution_csv(out_csv: str, record: Dict[str, Any]):

    p = Path(out_csv)

    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [

        "ts_utc", "symbol", "tf", "side", "size", "price", "sl", "tp", "status", "reason", "meta_w", "final_score"

    ]

    write_header = not p.exists()

    with open(p, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:

            writer.writeheader()

        writer.writerow({k: record.get(k) for k in fieldnames})



def decide_and_execute(plan: Dict[str, Any],

                       primary_conf_threshold: float = 0.7,

                       finrl_conf_threshold: float = 0.65,

                       size_reduction_factor: float = 0.8,

                       state_extra: Dict[str, Any] | None = None) -> ExecResult:

    """

    Inputs:

      plan: plan produced by build_plan(...) - contains side, final_score (confidence-like), size, price

      state_extra: optionally pass risk_cfg, finrl_policy etc.

    Logic:

      1) primary = plan['final_score'] (interpreted as primary conf)

      2) if primary >= PRIMARY_THRESH -> execute with plan.size

      3) else if finrl present, try to load finrl policy and ask for a confidence

      4) do final risk checks; then execute_order stub

    """

    state = state_extra or {}

    primary_conf = safe_float(plan.get("final_score"), 0.0)

    primary_signal = plan.get("side") if plan.get("enter") else None

    if not primary_signal:

        return ExecResult(status="NO_SIGNAL", reason="plan indicates no entry", size=0.0, sl=None, tp=None)



    # pre-checks

    if not passes_risk_checks(plan, {"risk_cfg": state.get("risk_cfg", {})}):

        return ExecResult(status="REJECTED_BY_RISK", reason="pre_checks", size=0.0, sl=None, tp=None)



    final_conf = primary_conf

    size = safe_float(plan.get("size"), 0.0)

    decision_tag = "NONE"



    if primary_conf >= primary_conf_threshold:

        # accept primary

        final_conf = primary_conf

        decision_tag = "EXECUTE_PRIMARY"

    else:

        # attempt finrl fallback

        finrl_policy = state.get("finrl_policy")  # pass in from caller if available

        finrl_conf = 0.0

        if finrl_policy:

            try:

                if hasattr(finrl_policy, "predict_proba"):

                    proba = finrl_policy.predict_proba([plan.get("candidate", {})])[0]

                    finrl_conf = max(proba) if isinstance(proba, (list, tuple)) else float(proba)  # type: ignore[arg-type]

                elif hasattr(finrl_policy, "predict"):

                    _ = finrl_policy.predict([plan.get("candidate", {})])

                    finrl_conf = safe_float(plan.get("candidate", {}).get("rl_conf"), 0.0)

                else:

                    finrl_conf = safe_float(plan.get("candidate", {}).get("rl_conf"), 0.0)

            except Exception as e:

                LOG.warning("decide_and_execute: finrl_policy call failed: %s", e)

                finrl_conf = safe_float(plan.get("candidate", {}).get("rl_conf"), 0.0)



        if finrl_conf >= finrl_conf_threshold:

            final_conf = finrl_conf

            size = size * size_reduction_factor

            decision_tag = "EXECUTE_FINRL_FALLBACK"

        else:

            # Soft gating: allow low confidence with reduced size instead of hard skip
            LOG.info("SOFT_GATE: primary_conf=%.3f finrl_conf=%.3f thresholds (%.3f, %.3f) - reducing size by 50%%", primary_conf, finrl_conf, primary_conf_threshold, finrl_conf_threshold)

            final_conf = primary_conf * 0.5  # Reduce confidence weighting

            size = size * 0.5  # 50% size for low confidence

            decision_tag = "EXECUTE_SOFT_GATED"



    # final safety checks

    if not final_risk_checks(size, {"risk_cfg": state.get("risk_cfg", {})}):

        return ExecResult(status="REJECTED_FINAL_RISK", reason="size_check_failed", size=0.0, sl=None, tp=None)



    # compute sl/tp if missing

    sl = plan.get("sl") or compute_sl_from_plan(plan)

    tp = plan.get("tp")

    price = safe_float(plan.get("price"), 0.0)

    symbol = plan.get("symbol")

    side = plan.get("side")



    # execute (paper)

    exec_res = execute_order(decision_tag, symbol, size, side, price, sl, tp)



    # append to executions CSV for traceability

    rec = {

        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),

        "symbol": symbol,

        "tf": plan.get("tf"),

        "side": side,

        "size": float(exec_res.size),

        "price": price,

        "sl": exec_res.sl,

        "tp": exec_res.tp,

        "status": exec_res.status,

        "reason": exec_res.reason,

        "meta_w": safe_float(plan.get("meta_w"), 0.5),

        "final_score": final_conf

    }

    append_execution_csv("reports/executions/executions.csv", rec)

    return exec_res



# -----------------------------------------------------------------------------

# Plan assembly

# -----------------------------------------------------------------------------

def save_plan(out_dir: str, symbol: str, tf: str, plan: Dict[str, Any]):

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    fname = f"{symbol}_{tf}_plan_{ts}.json"

    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, fname)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(plan, f, indent=2, default=str)

    LOG.info("Saved plan: %s", path)

    return path



def build_plan(candidate: Dict[str, Any], finrl_policy, meta_model, risk_cfg) -> Dict[str, Any]:

    # identifiers (PATCH: normalize symbol and USDT->USD)

    raw_symbol = (candidate.get("pair") or candidate.get("symbol") or candidate.get("asset") or "UNKNOWN").upper()

    symbol = _normalize_symbol_for_universe(raw_symbol)

    tf = candidate.get("tf") or candidate.get("timeframe") or "UNK"



    # meta weight

    features = candidate.copy()

    meta_w = compute_meta_weight(meta_model, features) if meta_model is not None else 0.5



    # finrl presence/model

    finrl_present = finrl_policy is not None

    has_rl_hint   = any(k in candidate for k in ("rl_action", "rl_conf", "rl_prob_long", "rl_prob_short"))

    rl_actionable = policy_is_actionable(finrl_policy)



    want_two = bool(risk_cfg.get("use_two_layer", True))

    use_two  = want_two and (has_rl_hint or rl_actionable)



    # === decision path ===

    if use_two:

        decision = decide_action_and_size_two_layer(candidate, meta_w, risk_cfg, finrl_policy)

    else:

        decision = decide_action_and_size(candidate, meta_w, risk_cfg, finrl_policy_present=finrl_present)



    price = safe_float(candidate.get("price"), 0.0)

    atr = safe_float(candidate.get("atr") or candidate.get("atr14"), 0.0)



    # Optional layer debug

    layer_debug = {}

    if use_two:

        s_side, s_conf, s_r = stats_signal(candidate, meta_w)

        r_side, r_conf, r_r = finrl_signal_local(finrl_policy, candidate)

        layer_debug = {

            "stats": {"side": s_side, "conf": s_conf, **s_r},

            "rl": {"side": r_side, "conf": r_conf, **r_r},

        }



    plan = {

        "symbol": symbol,

        "tf": tf,

        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),

        "candidate": candidate,

        "meta_w": meta_w,

        "finrl_present": bool(finrl_present),

        "two_layer_enabled": bool(use_two),



        # ---- mirrored fields on top-level ----

        "enter": bool(decision["enter"]),

        "side": decision["side"],

        "size": float(decision["size"]),

        "price": price,       # executor can CSV-backfill if price==0.0

        "atr": atr or None,

        "sl": decision["sl"],

        "tp": decision["tp"],

        "sl_type": decision["sl_type"],

        "final_score": float(decision["final_score"]),

        "reason": decision["reason"],



        # diagnostics

        "layer_signals": layer_debug if use_two else None,



        "notes": [

            "Two-layer fusion (Stats + FinRL) enabled" if use_two else "Single-layer (hybrid heuristic) path",

            "Executor may re-check risk gates and price-fill from CSV if price==0.0.",

        ],

        "decision": decision,

    }



    # ADD: csv_root can come from risk_cfg

    if price <= 0.0:

        csv_root = (risk_cfg.get("data", {}) or {}).get("csv_price_dir")

        if csv_root:

            p = backfill_price_from_csv(symbol, tf, csv_root)

            if p > 0.0:

                price = p

                candidate["price"] = p  # keep deterministic



    return plan



# --- add near top of src/decision_engine.py ---

def policy_is_actionable(pol) -> bool:

    try:

        return bool(pol) and (hasattr(pol, "predict_proba") or hasattr(pol, "predict"))

    except Exception:

        return False



# --- add near the top with helpers ---

def _sigmoid01(x: float) -> float:

    try:

        import math

        return 1.0 / (1.0 + math.exp(-x))

    except Exception:

        return 0.5



def _conf_from_plan(plan: dict) -> float:

    """

    Returns a 0..1 confidence for auto-exec:

    - If two-layer: plan['final_score'] is already 0..1

    - Else single-layer: convert signed score -> 0..1 via sigmoid(abs(score)*2)

    """

    fs = safe_float(plan.get("final_score"), 0.0)

    if bool(plan.get("two_layer_enabled", False)):

        # already a decision confidence

        c = fs

    else:

        # single-layer path produces signed score -> normalize

        c = _sigmoid01(abs(fs) * 2.0)

    # clamp

    return max(0.0, min(1.0, c))



# (backfill_price_from_csv is referenced above; if it's in your project elsewhere, keep it.

# If it isn't, you can stub it or remove the CSV backfill block.)

def backfill_price_from_csv(symbol: str, tf: str, root: str) -> float:

    try:

        # very light stub: try a file like <root>/<symbol>_<tf>.csv with a 'close' column last row

        cand = os.path.join(root, f"{symbol}_{tf}.csv")

        if not os.path.exists(cand):

            return 0.0

        import csv as _csv

        last = None

        with open(cand, "r", encoding="utf-8") as f:

            r = _csv.DictReader(f)

            for row in r:

                last = row

        if last and "close" in last:

            return safe_float(last["close"], 0.0)

    except Exception:

        pass

    return 0.0



# -----------------------------------------------------------------------------

# CLI

# -----------------------------------------------------------------------------

def main(args=None):

    parser = argparse.ArgumentParser(prog="decision_engine", description="Hybrid Decision Engine")

    parser.add_argument("--candidates", "-c", help="path to screener JSON (or folder). If omitted uses latest in reports/screener")

    parser.add_argument("--out", "-o", default="reports/daily", help="output folder for plans")

    parser.add_argument("--finrl_policies", "-f", default=None, help="folder with finrl policies (optional)")

    parser.add_argument("--meta_model", "-m", default="models/meta_selector/meta_selector.joblib", help="meta selector joblib path (optional)")

    parser.add_argument("--risk_config", "-r", default="config/decision.yaml", help="decision/risk config YAML")

    # new CLI args for auto-execute / thresholds

    parser.add_argument("--auto_execute", action="store_true", help="If set, attempt to decide+execute plans after building (paper).")

    parser.add_argument("--primary_thresh", type=float, default=0.70, help="primary confidence threshold")

    parser.add_argument("--finrl_thresh", type=float, default=0.65, help="finrl confidence threshold")

    parsed = parser.parse_args(args=args)



    LOG.info("Decision Engine start")



    # candidates

    candidates_path = parsed.candidates

    if not candidates_path:

        latest = find_latest_screener()

        if not latest:

            LOG.error("No screener candidate files found in reports/screener. Exiting.")

            return 2

        candidates_path = latest



    # load data

    try:

        cands = load_candidates(candidates_path)

    except Exception as e:

        LOG.exception("Failed loading candidates: %s", e)

        return 3



    meta_model = load_meta_selector(parsed.meta_model)

    risk_cfg = load_risk_config(parsed.risk_config)



    # ---- Bind FinRL sub-config deterministically from the YAML dict ----

    finrl_cfg: Dict[str, Any] = (risk_cfg.get("finrl") or {})

    def _finrl_log(rec: dict):

        try:

            logger_append(_plain(rec))  # pipeline logger if present (sanitized)

        except Exception:

            print(json.dumps(_plain(rec)))  # fallback to console JSON



    _finrl_log({

        "stage": "finrl_debug_cfg_resolve",

        "picked": "risk_cfg",

        "enable": bool(finrl_cfg.get("enable_finrl_adapter", False))

    })

    _finrl_log({

        "stage": "finrl_debug_cfg_snapshot",

        "keys": sorted(list(finrl_cfg.keys()))

    })



    finrl_dir = parsed.finrl_policies if parsed.finrl_policies else None

    finrl_map = load_finrl_policies(finrl_dir, _finrl_log) if finrl_dir else {}



    # (PATCH) Read filters & pre-filter once

    filters = (risk_cfg.get("filters") or {})

    universe = filters.get("universe")

    exclude = filters.get("exclude") or []



    cands = [

        c for c in cands

        if in_universe((c.get("pair") or c.get("symbol") or c.get("asset") or "UNKNOWN"), universe, exclude)

    ]



    out_dir = parsed.out

    os.makedirs(out_dir, exist_ok=True)



    # ------------------------------------------------------------------

    # Group siblings by symbol for MTF consensus (after filtering)

    # ------------------------------------------------------------------

    by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    for c in cands:

        sym = (c.get("pair") or c.get("symbol") or "UNKNOWN").upper()

        by_symbol.setdefault(sym, []).append(c)



    # Process candidates

    saved: List[str] = []

    for cand in cands:

        try:

            # Normalized symbol used for FinRL policy lookup (single source of truth)

            symbol_norm = _normalize_symbol_for_universe((cand.get("pair") or cand.get("symbol") or "UNKNOWN"))

            tf = cand.get("tf") or cand.get("timeframe") or "UNK"



            # Retrieve FinRL policy for the candidate using normalized key

            policy = finrl_map.get(symbol_norm)

            p_finrl = None



            # Gate FinRL from risk_cfg only (single source of truth)

            yaml_enable = bool(finrl_cfg.get("enable_finrl_adapter", False))

            if not yaml_enable:

                _finrl_log({"stage":"finrl_signal","symbol":symbol_norm,"skip":"disabled_flag"})

            elif not policy:

                _finrl_log({"stage":"finrl_signal","symbol":symbol_norm,"skip":"no_policy_in_map"})

            else:

                # Adapter signature: (candidate, policy, finrl_cfg)

                try:

                    side_ad, conf_ad, meta_ad = finrl_signal_adapter(cand, policy, finrl_cfg)

                    meta_ad = _plain(meta_ad)

                    _finrl_log({"stage":"finrl_signal","symbol":symbol_norm,"side":side_ad,"conf":round(conf_ad,6), **meta_ad})

                    p_finrl = conf_ad

                except Exception as e:

                    _finrl_log({"stage":"finrl_signal","symbol":symbol_norm,"error":str(e)})



            # extra proof if p_finrl missing

            if p_finrl is None:

                _finrl_log({

                    "stage":"finrl_signal",

                    "symbol": symbol_norm,

                    "reason": "no_policy_or_adapter_fail",

                    "have_policy": bool(policy is not None),

                    "tf": tf

                })

            else:

                _finrl_log({"stage":"finrl_signal","symbol":symbol_norm,"p_finrl":float(p_finrl)})



            # Debug FinRL configuration and keys

            _finrl_log({"stage":"finrl_debug_cfg","enable":yaml_enable})

            _finrl_log({"stage":"finrl_debug_keys","have":sorted(list(finrl_map.keys())), "sym":symbol_norm})



            # Extract statistical signal (use meta weight from meta_model if available)

            meta_w_for_stats = compute_meta_weight(meta_model, cand) if meta_model is not None else safe_float(risk_cfg.get("meta_weight", 0.5))

            s_side, p_primary, s_meta = stats_signal(cand, meta_w_for_stats)



            # Fuse primary + optional FinRL confidence for logging/diagnostics

            def fuse_primary_finrl(p_primary: float, p_opt, cfg: dict):

                if p_opt is None or p_opt < cfg.get("p_floor", 0.30):

                    return p_primary, {"used_finrl": False}

                w = cfg.get("w_finrl", 0.35)

                w = max(0.0, min(1.0, float(w)))

                return (1.0 - w) * p_primary + w * p_opt, {"used_finrl": True, "w": w}



            p_fused, fuse_meta = fuse_primary_finrl(p_primary, p_finrl, finrl_cfg)

            _finrl_log({

                "stage":"fusion_primary_finrl","symbol":symbol_norm,

                "p_primary":float(p_primary),

                "p_finrl": (float(p_finrl) if p_finrl is not None else None),

                "p_out":float(p_fused), **fuse_meta

            })



            # Build plan (uses two-layer path internally if RL actionable or hints exist)

            plan = build_plan(cand, policy, meta_model, risk_cfg)



            # Inject MTF result + gentle final_score nudge

            siblings = by_symbol.get((cand.get("pair") or c.get("symbol") or "UNKNOWN").upper(), [])

            mtf_cfg = (risk_cfg.get("mtf")

                       or (risk_cfg.get("decision", {}) or {}).get("mtf", {})

                       or {})

            side_mtf, mtf_dbg = mtf_consensus(symbol_norm, tf, siblings, mtf_cfg)

            plan.setdefault("reason", {})["mtf"] = mtf_dbg

            if mtf_cfg.get("enabled") and side_mtf in ("buy", "sell"):

                plan["side"] = side_mtf

                plan["enter"] = True

                if isinstance(plan.get("final_score"), (int, float)):

                    sgn = 1.0 if side_mtf == "buy" else -1.0

                    plan["final_score"] = float(max(-1.0, min(1.0, plan["final_score"] + 0.05 * sgn)))

            elif mtf_cfg.get("enabled") and side_mtf == "hold":

                plan["enter"] = False

                plan["side"] = "hold"

                plan["reason"]["mtf"]["forced_hold"] = True



            ppath = save_plan(out_dir, plan["symbol"], plan["tf"], plan)

            saved.append(ppath)



            # Optional auto-execute (paper) flow

            if parsed.auto_execute:

                state_for_exec = {

                    "risk_cfg": risk_cfg,

                    "finrl_policy": policy

                }

                exec_res = decide_and_execute(plan,

                                              primary_conf_threshold=parsed.primary_thresh,

                                              finrl_conf_threshold=parsed.finrl_thresh,

                                              size_reduction_factor=0.8,

                                              state_extra=state_for_exec)

                LOG.info("Auto-exec result for %s %s -> %s", plan["symbol"], plan["tf"], exec_res.status)



        except Exception:

            LOG.exception("Failed to build/save plan for candidate: %s", cand)



    LOG.info("Decision Engine done. Plans written: %d", len(saved))

    return 0



if __name__ == "__main__":

    import sys

    sys.exit(main())

