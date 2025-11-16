from typing import Dict, Any, List, Tuple
from utils.finrl_align import align_features_16

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + (2.718281828459045 ** (-x)))

def _unwrap(policy: Any) -> Any:
    # try common wrapper fields to reach the real estimator
    for attr in ("model", "inner", "estimator", "clf", "base_estimator"):
        if hasattr(policy, attr):
            try:
                return getattr(policy, attr)
            except Exception:
                pass
    return policy

def _proba_from_any(policy: Any, feats16: List[float]) -> Tuple[float, str]:
    # LightGBM Booster
    try:
        import lightgbm  # type: ignore
        if isinstance(policy, lightgbm.Booster):
            y = float(policy.predict([feats16])[0])
            p = _sigmoid(y) if (y < 0.0 or y > 1.0) else max(0.0, min(1.0, y))
            return p, "lgb_booster"
    except Exception:
        pass

    # Unwrap sklearn-like adapters
    core = _unwrap(policy)

    if hasattr(core, "predict_proba"):
        p = core.predict_proba([feats16])[0]
        try:
            return (float(p[1]) if len(p) >= 2 else float(max(p))), "sklearn_predict_proba"
        except Exception:
            try: return float(p), "sklearn_predict_proba_scalar"
            except Exception: return 0.55, "sklearn_predict_proba_err"

    if hasattr(core, "decision_function"):
        m = core.decision_function([feats16])[0]
        return _sigmoid(float(m)), "sklearn_decision_function"

    if hasattr(core, "predict"):
        y = core.predict([feats16])[0]
        try:
            return (1.0 if int(y) == 1 else 0.0), "sklearn_predict_label"
        except Exception:
            return 0.55, "sklearn_predict_label_err"

    # Presence fallback
    return 0.55, "policy_presence_only"

def finrl_signal(candidate: Dict[str, Any], finrl_policy: Any, cfg: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    if not cfg.get("enable_finrl_adapter", False):
        return "hold", 0.55, {"stage":"finrl_signal","source":"disabled"}
    try:
        feats = align_features_16(candidate)
        p_long, src = _proba_from_any(finrl_policy, feats)
        side = "buy" if p_long >= 0.5 else "sell"
        widen_k = float(cfg.get("widen_k", 0.70))
        conf = (abs(p_long - 0.5) * 2.0) * widen_k + (1.0 - widen_k)
        return side, conf, {"stage":"finrl_signal","source":src,"p_long":round(p_long,6),"vec_len":len(feats)}
    except Exception as e:
        return "hold", 0.55, {"stage":"finrl_signal","source":"adapter_exception","error":str(e)[:160]}