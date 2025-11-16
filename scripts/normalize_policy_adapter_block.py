import pathlib, re, sys

p = pathlib.Path("tools/policy_adapter.py")
src = p.read_text(encoding="utf-8")

# Ensure json import present at top
if not re.search(r'(?m)^\s*import\s+json\b', src):
    m = re.search(r'(?m)^(?:from\s+\S+\s+import\s+\S+|import\s+\S+).*\n')
    if m:
        src = src[:m.end()] + "import json\n" + src[m.end():]
    else:
        src = "import json\n" + src

# Find existing adapter def (impl or public)
m_def = re.search(r'(?m)^(?P<indent>[ \t]*)def\s+(?:_finrl_signal_adapter_impl|finrl_signal_adapter)\s*\(\s*candidate\s*,\s*policy\s*,\s*finrl_cfg\s*\)\s*:\s*$', src)
if not m_def:
    print("ERROR: adapter def not found"); sys.exit(1)

indent = m_def.group("indent")

# From first adapter def, find the end = next top-level def/class at same indent
start = m_def.start()
lines = src.splitlines(True)
prefix = src[:start]
start_line = prefix.count("\n")

def is_section_end(line: str) -> bool:
    return line.startswith(indent + "def ") or line.startswith(indent + "class ")

end_line = len(lines)
for i in range(start_line + 1, len(lines)):
    if is_section_end(lines[i]):
        end_line = i
        break

# Clean adapter region template (top-level; we'll prefix indent)
impl_body = """\
wrapper = policy
est = _unwrap_estimator(policy)
d   = _feat_dim(est)
X   = [_vectorize_candidate(candidate, d)]

# Fast-path: wrapper-specific scalar confidence (if available)
if hasattr(wrapper, "predict_conf"):
    try:
        conf_raw = wrapper.predict_conf(X)
        conf_val = float(conf_raw[0] if isinstance(conf_raw, (list, tuple)) else conf_raw)
        p_long  = float(candidate.get("rl_prob_long",  0.0))
        p_short = float(candidate.get("rl_prob_short", 0.0))
        if (p_long + p_short) > 0.0:
            side = "buy" if p_long >= p_short else "sell"
        else:
            side = "buy" if float(candidate.get("score", 0.0)) >= 0.0 else "sell"
        return side, max(0.0, min(1.0, conf_val)), {"source": "predict_conf", "feat_dim": d}
    except Exception:
        pass  # fall through

# Predict_proba branch
try:
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        if hasattr(proba, "__getitem__"):
            proba = proba[0]
        try:
            proba = list(map(float, proba))
        except Exception:
            proba = [0.5, 0.25, 0.25]
        if len(proba) == 3:
            p_hold, p_long, p_short = proba
        elif len(proba) == 2:
            p_long, p_short = proba; p_hold = max(0.0, 1.0 - p_long - p_short)
        else:
            p_hold, p_long, p_short = 0.5, 0.25, 0.25
        if max(p_long, p_short, p_hold) == p_hold:
            side, conf = "hold", max(0.0, 1.0 - p_hold)
        elif p_long >= p_short:
            side, conf = "buy", max(0.0, min(1.0, p_long - p_short + 0.5*p_long))
        else:
            side, conf = "sell", max(0.0, min(1.0, p_short - p_long + 0.5*p_short))
        return side, conf, {"source":"predict_proba", "feat_dim": d}
except Exception:
    pass  # continue to predict branch

# Predict branch (guard continuous outputs)
if hasattr(est, "predict"):
    try:
        y = est.predict(X)
        y0 = y[0] if isinstance(y, (list, tuple)) else y
        try:
            y0i = int(y0)
            if y0i in (-1, 0, 1):
                side = {1:"buy",-1:"sell",0:"hold"}[y0i]
                conf = float(candidate.get("rl_conf", 0.6))
                return side, max(0.0, min(1.0, conf)), {"source":"predict","y": y0i, "feat_dim": d}
            else:
                return "hold", 0.5, {"source":"predict_continuous","y": y0i, "feat_dim": d}
        except Exception:
            y0s = str(y0).lower()
            if y0s in ("buy","long"):
                side = "buy"
            elif y0s in ("sell","short"):
                side = "sell"
            else:
                return "hold", 0.5, {"source":"predict_continuous","y": y0s, "feat_dim": d}
        conf = float(candidate.get("rl_conf", 0.6))
        return side, max(0.0, min(1.0, conf)), {"source":"predict","y": y0, "feat_dim": d}
    except Exception as e:
        return "hold", 0.5, {"source":"predict_error","error": str(e), "feat_dim": d}

# Presence-only bump (deterministic)
widen_k = float(finrl_cfg.get("widen_k", 0.7))
conf = max(0.0, min(1.0, 0.5 + 0.05*widen_k))
return "hold", conf, {"source":"policy_presence_only", "feat_dim": d}
"""

wrapper_body = """\
est = _unwrap_estimator(policy)
# pre-call audit (stdout; never break flow)
try:
    print(json.dumps({
        "stage": "finrl_signal_audit",
        "symbol": candidate.get("symbol"),
        "est_type": type(est).__name__,
        "has_proba": hasattr(est, "predict_proba"),
        "has_predict": hasattr(est, "predict"),
    }, ensure_ascii=False))
except Exception:
    pass

side, conf, meta = _finrl_signal_adapter_impl(candidate, policy, finrl_cfg)

# post-call summary
try:
    src_tag = meta.get("source") if isinstance(meta, dict) else None
    feat_dim = meta.get("feat_dim") if isinstance(meta, dict) else None
    print(json.dumps({
        "stage": "finrl_signal",
        "symbol": candidate.get("symbol"),
        "side": side,
        "conf": conf,
        "source": src_tag,
        "feat_dim": feat_dim
    }, ensure_ascii=False))
except Exception:
    pass

return side, conf, meta
"""

def indent_block(code: str, base: str, spaces: int = 4) -> str:
    pad = base + " " * spaces
    return "".join((pad + ln if ln else pad) + "\n" for ln in code.splitlines())

impl_def = f"{indent}def _finrl_signal_adapter_impl(candidate, policy, finrl_cfg):\n" + indent_block(impl_body, indent)
wrapper_def = f"{indent}# ---- AUDIT WRAPPER: always emit audit lines ----\n" \
              f"{indent}def finrl_signal_adapter(candidate, policy, finrl_cfg):\n" + indent_block(wrapper_body, indent) \
              + f"{indent}# ---- END AUDIT WRAPPER ----\n"

new_region = impl_def + wrapper_def

# Replace region
new_src = "".join(lines[:start_line]) + new_region + "".join(lines[end_line:])

# Final clean: tabs -> spaces
new_src = new_src.replace("\t", "    ")

# Write back
p.write_text(new_src, encoding="utf-8")
print("OK: normalized adapter block")
