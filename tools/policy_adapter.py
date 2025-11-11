# tools/policy_adapter.py
# Clean, deterministic FinRL policy adapter (public API preserved)
# - No file I/O, no globals
# - Optional in-memory emit hook via finrl_cfg["emit"]
# - Deterministic and idempotent

from typing import Any, Dict, Iterable, List, Tuple

# =========================
# Public lightweight wrapper
# =========================
class SklearnPolicyAdapter:
    """
    Minimal 'policy' wrapper exposing predict_conf(x_vec)->float in [0,1] for BUY.
    Accepts either [feat...] or [[feat...]] and normalizes internally.
    """

    def __init__(self, clf: Any, proba_index: int = 1, meta: Dict[str, Any] | None = None) -> None:
        self.clf = clf
        self.proba_index = int(proba_index)
        self.algo = "adapter_sklearn"
        self.meta = dict(meta or {})

    def predict_conf(self, x_vec: Iterable[float]) -> float:
        """
        Returns a probability-like confidence in [0,1] for 'buy'.
        Handles:
          - predict_proba -> use class index (default 1)
          - decision_function / raw score / logits -> sigmoid
          - predict -> tries float in [0,1], else treats as score->sigmoid,
                       else maps {-1,0,1} or strings
        Deterministic; never raises; clamps to [0,1].
        """
        import math

        # normalize input to one sample
        X = (
            x_vec[0]
            if isinstance(x_vec, (list, tuple))
            and len(x_vec)
            and isinstance(x_vec[0], (list, tuple))
            else x_vec
        )
        X = [float(v) for v in X]
        Xw = [X]

        est = _unwrap_estimator(self.clf)

        try:
            # 1) True probabilities
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(Xw)
                row = proba[0] if hasattr(proba, "__getitem__") else proba
                try:
                    p = float(row[self.proba_index])  # prefer explicit class index
                except Exception:
                    p = float(row)  # single-proba models
                return _clamp01(p)

            # 2) Decision scores / logits -> sigmoid
            if hasattr(est, "decision_function"):
                d = est.decision_function(Xw)
                d0 = d[0] if hasattr(d, "__getitem__") else d
                try:
                    d0 = float(getattr(d0, "item", lambda: d0)())
                except Exception:
                    d0 = float(d0)
                return _clamp01(1.0 / (1.0 + math.exp(-d0)))

            # 3) Generic predict()
            if hasattr(est, "predict"):
                y = est.predict(Xw)
                y0 = y[0] if hasattr(y, "__getitem__") else y

                # float path
                try:
                    y0f = float(getattr(y0, "item", lambda: y0)())
                    if 0.0 <= y0f <= 1.0:
                        return _clamp01(y0f)  # already a probability
                    # treat as raw score/logit
                    return _clamp01(1.0 / (1.0 + math.exp(-y0f)))
                except Exception:
                    pass

                # discrete {-1,0,1}
                try:
                    y0i = int(y0)
                    if y0i == 1:
                        return 1.0
                    if y0i == 0:
                        return 0.5
                    if y0i == -1:
                        return 0.0
                except Exception:
                    pass

                # string mapping
                y0s = str(y0).lower()
                if y0s in ("buy", "long", "1", "true", "yes", "up"):
                    return 1.0
                if y0s in ("hold", "neutral", "0", "none"):
                    return 0.5
                if y0s in ("sell", "short", "-1", "false", "no", "down"):
                    return 0.0

            # 4) Unknown → neutral
            return 0.5

        except Exception:
            return 0.5


# =========================
# Helpers (pure, cheap)
# =========================
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _is_estimator(x: Any) -> bool:
    return any(hasattr(x, k) for k in ("predict_proba", "predict", "__call__"))


def _score_estimator(x: Any) -> int:
    s = 0
    s += 4 if hasattr(x, "predict_proba") else 0
    s += 2 if hasattr(x, "predict") else 0
    s += 1 if callable(x) else 0
    s += 1 if hasattr(x, "classes_") else 0
    s += 1 if hasattr(x, "n_features_in_") else 0
    return s


def _dig_candidates(obj: Any, seen: set[int] | None = None, depth: int = 0, max_depth: int = 3) -> List[Any]:
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > max_depth:
        return []
    seen.add(id(obj))

    out: List[Any] = []
    if _is_estimator(obj):
        out.append(obj)

    # common attributes that may hold estimators
    for attr in ("model", "clf", "estimator", "base_estimator", "policy", "wrapped", "inner", "pipeline", "pipe"):
        if hasattr(obj, attr):
            out += _dig_candidates(getattr(obj, attr), seen, depth + 1, max_depth)

    # pipeline-like containers
    for attr in ("named_steps", "steps"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, dict):
                for v in val.values():
                    out += _dig_candidates(v, seen, depth + 1, max_depth)
            elif isinstance(val, (list, tuple)):
                for pair in val:
                    v = pair[1] if isinstance(pair, (list, tuple)) and len(pair) == 2 else pair
                    out += _dig_candidates(v, seen, depth + 1, max_depth)

    # generic containers
    if isinstance(obj, dict):
        for v in obj.values():
            out += _dig_candidates(v, seen, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _dig_candidates(v, seen, depth + 1, max_depth)

    return out


def _unwrap_estimator(pol: Any) -> Any:
    """
    Returns the best candidate estimator-like object from a wrapper.
    Never raises; falls back to the original object if nothing found.
    """
    try:
        cands = _dig_candidates(pol)
        return max(cands, key=_score_estimator) if cands else pol
    except Exception:
        for k in ("predict_proba", "predict", "__call__"):
            if hasattr(pol, k):
                return pol
        for attr in ("model", "clf", "estimator", "base_estimator", "policy", "wrapped", "inner", "pipeline", "pipe"):
            if hasattr(pol, attr):
                inner = getattr(pol, attr)
                for k in ("predict_proba", "predict", "__call__"):
                    if hasattr(inner, k):
                        return inner
        return pol


def _feat_dim(est: Any) -> int:
    for k in ("n_features_in_", "n_features_"):
        if hasattr(est, k):
            try:
                v = int(getattr(est, k))
                if v > 0:
                    return v
            except Exception:
                pass
    est2 = getattr(est, "estimator", None) or getattr(est, "base_estimator", None)
    if est2 is not None and hasattr(est2, "n_features_in_"):
        try:
            v = int(est2.n_features_in_)
            if v > 0:
                return v
        except Exception:
            pass
    return 16


def _vectorize_candidate(cand: Dict[str, Any], d: int) -> List[float]:
    st = cand.get("state")
    if isinstance(st, (list, tuple)) and len(st) > 0:
        v = list(st)[:d]
        if len(v) < d:
            v += [0.0] * (d - len(v))
        return [float(x) for x in v]
    # tiny fallback set if no state present
    keys = ("score", "rsi14", "adx_proxy", "atr14", "vol_ratio", "impact", "surprise", "confidence")
    v: List[float] = []
    for k in keys:
        try:
            v.append(float(cand.get(k, 0.0)))
        except Exception:
            v.append(0.0)
    if len(v) < d:
        v += [0.0] * (d - len(v))
    return v[:d]


def _proba_to_triplet(est: Any, proba: Iterable[float]) -> Tuple[float, float, float]:
    """
    Map predict_proba outputs to (hold, long, short) using labels when possible.
    """
    proba = [float(x) for x in proba]
    p_hold = p_long = p_short = 0.0
    if hasattr(est, "classes_"):
        cls = list(getattr(est, "classes_"))
        idx = {cls[i]: i for i in range(len(cls))}
        if 1 in idx:
            p_long = proba[idx[1]]
        if -1 in idx:
            p_short = proba[idx[-1]]
        if 0 in idx:
            p_hold = proba[idx[0]]
    if p_hold == p_long == p_short == 0.0:
        if len(proba) == 3:
            p_hold, p_long, p_short = proba
        elif len(proba) == 2:
            p_long, p_short = proba
            p_hold = max(0.0, 1.0 - p_long - p_short)
        else:
            p_hold, p_long, p_short = 0.5, 0.25, 0.25
    return p_hold, p_long, p_short


# =========================
# Main adapter implementation
# =========================
def _finrl_signal_adapter_impl(candidate: Dict[str, Any], policy: Any, finrl_cfg: Dict[str, Any]):
    # optional emitter (typed I/O only)
    emit = finrl_cfg.get("emit") if isinstance(finrl_cfg, dict) else None
    sym = candidate.get("symbol") or candidate.get("sym") or candidate.get("ticker") or candidate.get("pair")
    tf = candidate.get("tf") or candidate.get("timeframe")

    wrapper = policy
    est = _unwrap_estimator(policy)
    d = _feat_dim(est)
    X = [_vectorize_candidate(candidate, d)]

    # Fast-path: wrapper-specific scalar confidence (if available)
    if hasattr(wrapper, "predict_conf"):
        try:
            conf_val = float(wrapper.predict_conf(X))
            # Side selection heuristic (deterministic, no I/O)
            p_long = float(candidate.get("rl_prob_long", 0.0))
            p_short = float(candidate.get("rl_prob_short", 0.0))
            if (p_long + p_short) > 0.0:
                side = "buy" if p_long >= p_short else "sell"
            else:
                side = "buy" if float(candidate.get("score", 0.0)) >= 0.0 else "sell"
            meta = {"source": "predict_conf", "feat_dim": d}
            if callable(emit):
                try:
                    emit({"stage": "finrl_signal", "symbol": sym, "tf": tf, "side": side, "conf": conf_val, "source": "predict_conf", "feat_dim": d})
                except Exception:
                    pass
            return side, _clamp01(conf_val), meta
        except Exception:
            # fall through to other paths
            pass

    # Predict_proba branch
    try:
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            row = proba[0] if hasattr(proba, "__getitem__") else proba
            try:
                row = [float(x) for x in row]
            except Exception:
                row = [0.5, 0.25, 0.25]
            p_hold, p_long, p_short = _proba_to_triplet(est, row)
            if max(p_long, p_short, p_hold) == p_hold:
                side, conf = "hold", max(0.0, 1.0 - p_hold)
            elif p_long >= p_short:
                side, conf = "buy", _clamp01(p_long - p_short + 0.5 * p_long)
            else:
                side, conf = "sell", _clamp01(p_short - p_long + 0.5 * p_short)
            meta = {"source": "predict_proba", "feat_dim": d}
            if callable(emit):
                try:
                    emit({"stage": "finrl_signal", "symbol": sym, "tf": tf, "side": side, "conf": conf, "source": "predict_proba", "feat_dim": d})
                except Exception:
                    pass
            return side, conf, meta
    except Exception:
        # continue to predict branch
        pass

    # Predict branch (guard continuous outputs)
    if hasattr(est, "predict"):
        try:
            y = est.predict(X)
            y0 = y[0] if hasattr(y, "__getitem__") else y
            # numeric path
            try:
                y0f = float(getattr(y0, "item", lambda: y0)())
                # treat as probability if already in [0,1], else as score→sigmoid
                conf = y0f if 0.0 <= y0f <= 1.0 else _clamp01(1.0 / (1.0 + __import__("math").exp(-y0f)))
                # use candidate priors for side, else threshold
                p_long = float(candidate.get("rl_prob_long", 0.0))
                p_short = float(candidate.get("rl_prob_short", 0.0))
                if (p_long + p_short) > 0.0:
                    side = "buy" if p_long >= p_short else "sell"
                else:
                    side = "buy" if conf >= 0.5 else "sell"
                meta = {"source": "predict", "feat_dim": d, "y": y0f}
                if callable(emit):
                    try:
                        emit({"stage": "finrl_signal", "symbol": sym, "tf": tf, "side": side, "conf": conf, "source": "predict", "feat_dim": d})
                    except Exception:
                        pass
                return side, conf, meta
            except Exception:
                # discrete / string mapping
                try:
                    y0i = int(y0)
                    if y0i == 1:
                        side, conf = "buy", 1.0
                    elif y0i == 0:
                        side, conf = "hold", 0.5
                    elif y0i == -1:
                        side, conf = "sell", 1.0
                    else:
                        return "hold", 0.5, {"source": "predict_continuous", "feat_dim": d, "y": y0i}
                    meta = {"source": "predict", "feat_dim": d, "y": y0i}
                    if callable(emit):
                        try:
                            emit({"stage": "finrl_signal", "symbol": sym, "tf": tf, "side": side, "conf": conf, "source": "predict", "feat_dim": d})
                        except Exception:
                            pass
                    return side, conf, meta
                except Exception:
                    y0s = str(y0).lower()
                    if y0s in ("buy", "long", "true", "yes", "up"):
                        side, conf = "buy", 1.0
                    elif y0s in ("sell", "short", "false", "no", "down"):
                        side, conf = "sell", 1.0
                    elif y0s in ("hold", "neutral", "none", "0"):
                        side, conf = "hold", 0.5
                    else:
                        return "hold", 0.5, {"source": "predict_continuous", "feat_dim": d, "y": y0s}
                    meta = {"source": "predict", "feat_dim": d, "y": y0s}
                    if callable(emit):
                        try:
                            emit({"stage": "finrl_signal", "symbol": sym, "tf": tf, "side": side, "conf": conf, "source": "predict", "feat_dim": d})
                        except Exception:
                            pass
                    return side, conf, meta
        except Exception as e:
            return "hold", 0.5, {"source": "predict_error", "feat_dim": d, "error": str(e)}

    # Presence-only bump (deterministic)
    widen_k = float(finrl_cfg.get("widen_k", 0.7)) if isinstance(finrl_cfg, dict) else 0.7
    conf = _clamp01(0.5 + 0.05 * widen_k)
    return "hold", conf, {"source": "policy_presence_only", "feat_dim": d}


# =========================
# Public entry (with optional audit)
# =========================
def finrl_signal_adapter(candidate: Dict[str, Any], policy: Any, finrl_cfg: Dict[str, Any]):
    """
    Public API. Optional in-memory emitter via finrl_cfg['emit'].
    No file I/O. Deterministic. Typed I/O only.
    """
    emit = finrl_cfg.get("emit") if isinstance(finrl_cfg, dict) else None

    sym = candidate.get("symbol") or candidate.get("sym") or candidate.get("ticker") or candidate.get("pair")
    tf = candidate.get("tf") or candidate.get("timeframe")

    # Pre-call audit (best effort, never raises)
    if callable(emit):
        try:
            est = _unwrap_estimator(policy)
            emit({
                "stage": "finrl_signal_audit",
                "symbol": sym,
                "tf": tf,
                "est_type": type(est).__name__,
                "has_proba": hasattr(est, "predict_proba"),
                "has_predict": hasattr(est, "predict"),
            })
        except Exception:
            pass

    side, conf, meta = _finrl_signal_adapter_impl(candidate, policy, finrl_cfg)

    # Post-call audit (best effort)
    if callable(emit):
        try:
            emit({
                "stage": "finrl_signal",
                "symbol": sym,
                "tf": tf,
                "side": side,
                "conf": conf,
                "source": (meta.get("source") if isinstance(meta, dict) else None),
                "feat_dim": (meta.get("feat_dim") if isinstance(meta, dict) else None),
            })
        except Exception:
            pass

    return side, conf, meta
