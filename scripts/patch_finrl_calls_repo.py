import pathlib
import re
import sys

ROOT = pathlib.Path(".").resolve()

ADAPTER_NAME = "finrl_signal_adapter"
RUNNER_PATH = ROOT / "tools" / "decision_engine_runner.py"
EXCLUDE_FILE = str(ROOT / "tools" / "policy_adapter.py")

SEARCH_DIRS = [ROOT / "tools", ROOT / "src", ROOT / "utils"]  # limit scope to project code

IMPORT_LINE = (
    "from tools.policy_adapter import finrl_signal_adapter, _unwrap_estimator, SklearnPolicyAdapter\n"
)

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
    r'(?P<call>finrl_signal_adapter)\s*\(\s*'
    r'(?P<cand>[^,()]+?)\s*,\s*'
    r'(?P<pol>[^,()]+?)\s*,\s*'
    r'(?P<cfg>[^)]+?)\s*\)\s*$'
)

GENERIC_CALL_RE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)'
    r'(?P<lhs>side\s*,\s*conf\s*,\s*meta\s*=\s*)'
    r'(?P<call>finrl_signal_adapter)\s*\((?P<args>.+?)\)\s*$'
)

def read_text_safely(p: pathlib.Path):
    """Read file with robust fallback; return (text, encoding) or (None, None) if binary/unreadable."""
    try:
        return p.read_text(encoding="utf-8"), "utf-8"
    except Exception:
        data = p.read_bytes()
        # Heuristic: skip likely binaries
        if b"\x00" in data or any(b < 9 and b not in (10, 13) for b in data[:512]):
            return None, None
        for enc in ("cp1252", "latin-1"):
            try:
                return data.decode(enc), enc
            except Exception:
                continue
    return None, None

def write_text_utf8(p: pathlib.Path, text: str):
    p.write_text(text, encoding="utf-8")

def ensure_runner_import_and_helper(src: str) -> str:
    out = src
    if IMPORT_LINE.strip() not in out:
        # insert import after the last import block
        lines = out.splitlines(True)
        last_import = -1
        for i, line in enumerate(lines):
            if re.match(r'^(from\s+\S+\s+import\s+.+|import\s+\S+)', line):
                last_import = i
            elif last_import >= 0 and line.strip() and not line.strip().startswith("#"):
                break
        insert_at = last_import + 1 if last_import >= 0 else 0
        lines.insert(insert_at, IMPORT_LINE)
        out = "".join(lines)

    if "def finrl_audit_call(" not in out:
        # place helper after import block
        lines = out.splitlines(True)
        last_import = -1
        for i, line in enumerate(lines):
            if re.match(r'^(from\s+\S+\s+import\s+.+|import\s+\S+)', line):
                last_import = i
            elif last_import >= 0 and line.strip() and not line.strip().startswith("#"):
                break
        insert_at = last_import + 1 if last_import >= 0 else 0
        if insert_at < len(lines) and lines[insert_at].strip():
            lines.insert(insert_at, "\n"); insert_at += 1
        lines.insert(insert_at, AUDIT_HELPER + "\n")
        out = "".join(lines)
    return out

def apply_runner_rewrite(text: str) -> (str, int):
    if ADAPTER_NAME not in text:
        return text, 0
    changed = 0
    def repl(m):
        nonlocal changed
        indent = m.group("indent")
        lhs = m.group("lhs")
        cand = m.group("cand").strip()
        pol = m.group("pol").strip()
        cfg = m.group("cfg").strip()
        if "finrl_audit_call(" in text[m.start():m.end()+200]:
            return m.group(0)
        changed += 1
        return (
            f"{indent}policy = SklearnPolicyAdapter(_unwrap_estimator({pol}))\n"
            f"{indent}{lhs}finrl_audit_call({cand}, policy, {cfg}, sym, logger_append)"
        )
    new_text, n = CALL_RE.subn(repl, text)
    changed += n
    if n == 0:
        # fallback generic
        def repl2(m):
            nonlocal changed
            indent = m.group("indent")
            lhs = m.group("lhs")
            parts = [p.strip() for p in re.split(r',(?![^(]*\))', m.group("args"))]
            if len(parts) < 3:
                return m.group(0)
            cand, pol, cfg = parts[0], parts[1], parts[2]
            changed += 1
            return (
                f"{indent}policy = SklearnPolicyAdapter(_unwrap_estimator({pol}))\n"
                f"{indent}{lhs}finrl_audit_call({cand}, policy, {cfg}, sym, logger_append)"
            )
        new_text, n2 = GENERIC_CALL_RE.subn(repl2, new_text)
        changed += n2
    return new_text, changed

def apply_wrap_only_rewrite(text: str) -> (str, int):
    if ADAPTER_NAME not in text:
        return text, 0
    changed = 0
    def repl(m):
        nonlocal changed
        indent = m.group("indent")
        lhs = m.group("lhs")
        cand = m.group("cand").strip()
        pol = m.group("pol").strip()
        cfg = m.group("cfg").strip()
        changed += 1
        return (
            f"{indent}policy = SklearnPolicyAdapter(_unwrap_estimator({pol}))\n"
            f"{indent}{lhs}{ADAPTER_NAME}({cand}, policy, {cfg})"
        )
    new_text, n = CALL_RE.subn(repl, text)
    changed += n
    if n == 0:
        def repl2(m):
            nonlocal changed
            indent = m.group("indent")
            lhs = m.group("lhs")
            parts = [p.strip() for p in re.split(r',(?![^(]*\))', m.group("args"))]
            if len(parts) < 3:
                return m.group(0)
            cand, pol, cfg = parts[0], parts[1], parts[2]
            changed += 1
            return (
                f"{indent}policy = SklearnPolicyAdapter(_unwrap_estimator({pol}))\n"
                f"{indent}{lhs}{ADAPTER_NAME}({cand}, policy, {cfg})"
            )
        new_text, n2 = GENERIC_CALL_RE.subn(repl2, text)
        changed += n2
    return new_text, changed

def ensure_import(src: str) -> str:
    if IMPORT_LINE.strip() in src:
        return src
    lines = src.splitlines(True)
    last_import = -1
    for i, line in enumerate(lines):
        if re.match(r'^(from\s+\S+\s+import\s+.+|import\s+\S+)', line):
            last_import = i
        elif last_import >= 0 and line.strip() and not line.strip().startswith("#"):
            break
    insert_at = last_import + 1 if last_import >= 0 else 0
    lines.insert(insert_at, IMPORT_LINE)
    return "".join(lines)

def iter_candidate_files():
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for pyf in base.rglob("*.py"):
            # Skip known non-project or generated paths
            p = str(pyf).lower()
            if any(skip in p for skip in (r"site-packages", r"venv", r".venv", r"__pycache__")):
                continue
            yield pyf

def main():
    total_changed = 0
    files_changed = 0
    wrapped_only = 0
    wrapped_audit = 0

    # Runner: ensure import + helper + rewrite with audit
    if RUNNER_PATH.exists():
        runner_src, enc = read_text_safely(RUNNER_PATH)
        if runner_src is None:
            print(f"skip_runner_unreadable: {RUNNER_PATH}")
        else:
            runner_src2 = ensure_runner_import_and_helper(runner_src)
            runner_src3, n_runner = apply_runner_rewrite(runner_src2)
            if runner_src3 != runner_src:
                write_text_utf8(RUNNER_PATH, runner_src3)
                print(f"patched_runner_calls={n_runner}")
                total_changed += n_runner
                files_changed += 1
                wrapped_audit += n_runner

    # Other files: wrap-only
    for pyf in iter_candidate_files():
        p = str(pyf)
        if p == str(RUNNER_PATH) or p == EXCLUDE_FILE:
            continue
        src, enc = read_text_safely(pyf)
        if src is None:
            # unreadable/binary: skip
            continue
        if ADAPTER_NAME not in src or "finrl_audit_call(" in src:
            continue
        src2 = ensure_import(src)
        src3, n = apply_wrap_only_rewrite(src2)
        if n > 0 and src3 != src:
            write_text_utf8(pyf, src3)
            print(f"patched_wrap_only {pyf}: calls={n} (was {enc})")
            total_changed += n
            files_changed += 1
            wrapped_only += n

    print(f"SUMMARY: files_changed={files_changed}, total_calls_patched={total_changed}, wrap_only={wrapped_only}, wrap_audit={wrapped_audit}")

if __name__ == "__main__":
    sys.exit(main())
