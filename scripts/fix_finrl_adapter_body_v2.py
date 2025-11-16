import pathlib, sys

p = pathlib.Path("tools/policy_adapter.py")
text = p.read_text(encoding="utf-8", errors="strict")

# Normalize line endings logic
eol = "\r\n" if "\r\n" in text else "\n"
lines = text.splitlines()

def find_def(lines):
    # Return (decor_start_idx, def_idx, indent_str) for finrl_signal_adapter
    for i, line in enumerate(lines):
        if "def finrl_signal_adapter" in line and line.lstrip().startswith("def finrl_signal_adapter"):
            indent = line[:len(line) - len(line.lstrip())]
            # backtrack decorators with same indent
            j = i - 1
            deco_start = i
            while j >= 0 and lines[j].startswith(indent + "@"):
                deco_start = j
                j -= 1
            return deco_start, i, indent
    return None, None, None

deco_start, def_idx, indent = find_def(lines)
if def_idx is None:
    print("ERROR: finrl_signal_adapter not found"); sys.exit(1)

# find start of body = the next line after def
body_start = def_idx + 1

# find end of body: next line that starts with same indent and begins with 'def ' or 'class ', or EOF
def is_next_top_level(idx):
    if idx >= len(lines): return True
    s = lines[idx]
    return s.startswith(indent + "def ") or s.startswith(indent + "class ")

body_end = len(lines)
for k in range(body_start, len(lines)):
    if is_next_top_level(k):
        body_end = k
        break

# New body content (no leading indent; we will indent it)
new_body_core = """\
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
        # normalize first row
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
    # continue to predict() branch
    pass

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

# Re-indent the new body with def indent + 4 spaces
body_indent = indent + "    "
new_body_lines = [(body_indent + ln if ln else body_indent) for ln in new_body_core.split("\n")]
new_body_block = eol.join(new_body_lines) + eol

# Build output
out_lines = lines[:body_start] + [new_body_block] + lines[body_end:]
new_text = eol.join(out_lines).replace(new_body_block + eol, new_body_block)  # guard double EOL

p.write_text(new_text, encoding="utf-8")
print("OK: rewrote finrl_signal_adapter body with correct indent")
