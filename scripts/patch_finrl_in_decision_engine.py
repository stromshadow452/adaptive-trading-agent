import pathlib, re, sys, json

ROOT = pathlib.Path(".").resolve()
TARGET = ROOT / "src" / "decision_engine.py"

IMPORT_LINE = "from tools.policy_adapter import finrl_signal_adapter, _unwrap_estimator, SklearnPolicyAdapter\n"

AUDIT_HELPER = """\
# ---- ADDITIVE: finrl audit helper (pure) ----
def finrl_audit_call(candidate, policy, finrl_cfg, sym, logger):
    est = _unwrap_estimator(policy)
    try:
        logger({
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
        logger({
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
"""

CALL_RE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)'
    r'(?P<lhs>side\s*,\s*conf\s*,\s*meta\s*=\s*)'
    r'finrl_signal_adapter\s*\(\s*'
    r'(?P<cand>[^,()]+?)\s*,\s*'
    r'(?P<pol>[^,()]+?)\s*,\s*'
    r'(?P<cfg>[^)]+?)\s*\)\s*$'
)

GENERIC_CALL_RE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)'
    r'(?P<lhs>side\s*,\s*conf\s*,\s*meta\s*=\s*)'
    r'finrl_signal_adapter\s*\((?P<args>.+?)\)\s*$'
)

def read(p: pathlib.Path):
    raw = p.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try: return raw.decode(enc)
        except: pass
    raise RuntimeError("encoding failed")

def ensure_import_and_helper(src):
    out = src
    if IMPORT_LINE.strip() not in out:
        lines = out.splitlines(True)
        last = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last = i
        lines.insert(last+1, IMPORT_LINE)
        out = "".join(lines)
    if "def finrl_audit_call(" not in out:
        lines = out.splitlines(True)
        last = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last = i
        lines.insert(last+1, AUDIT_HELPER + "\n")
        out = "".join(lines)
    return out

def rewrite(src):
    changed = 0
    def repl(m):
        nonlocal changed
        indent, lhs = m.group("indent"), m.group("lhs")
        cand, pol, cfg = m.group("cand").strip(), m.group("pol").strip(), m.group("cfg").strip()
        changed += 1
        return (
            f"{indent}policy = SklearnPolicyAdapter(_unwrap_estimator({pol}))\n"
            f"{indent}{lhs}finrl_audit_call({cand}, policy, {cfg}, sym, logger_append)"
        )
    new, n1 = CALL_RE.subn(repl, src)
    changed += n1
    if n1 == 0:
        def repl2(m):
            nonlocal changed
            indent, lhs = m.group("indent"), m.group("lhs")
            parts = [x.strip() for x in m.group("args").split(",")]
            if len(parts) < 3: return m.group(0)
            cand, pol, cfg = parts[0], parts[1], parts[2]
            changed += 1
            return (
                f"{indent}policy = SklearnPolicyAdapter(_unwrap_estimator({pol}))\n"
                f"{indent}{lhs}finrl_audit_call({cand}, policy, {cfg}, sym, logger_append)"
            )
        new, n2 = GENERIC_CALL_RE.subn(repl2, new)
        changed += n2
    return new, changed

def main():
    if not TARGET.exists():
        print("decision_engine.py not found")
        sys.exit(1)
    src = read(TARGET)
    orig = src
    src = ensure_import_and_helper(src)
    src, n = rewrite(src)
    if src != orig:
        TARGET.write_text(src, encoding="utf-8")
        print(f"patched_decision_engine_calls={n}")
    else:
        print("no_changes_needed")

if __name__ == "__main__":
    main()
