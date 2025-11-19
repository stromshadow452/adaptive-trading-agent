# tools/patch_require_ts.py
from pathlib import Path
import sys, shutil

fp = Path("tools/executor.py")
if not fp.exists():
    print("tools/executor.py not found; abort.")
    sys.exit(2)

bak = fp.with_suffix(".require_ts.bak")
shutil.copy2(fp, bak)
print(f"Backup -> {bak}")

txt = fp.read_text(encoding="utf-8")

old_fn_start = "def _require_exec_timestamp(exec_row):"
if old_fn_start not in txt:
    print("Could not find original _require_exec_timestamp signature. Aborting.")
    sys.exit(3)

# New function: non-fatal by default; strict mode via env STRICT_NO_CSV_TS=1
new_block = '''
def _require_exec_timestamp(exec_row, *, raise_on_missing: bool = False):
    """
    Validate an execution row has a timestamp.

    Non-fatal by default: if timestamp is missing we log/print a warning and return False.
    If strict behavior is required, set environment var STRICT_NO_CSV_TS=1 or call with
    raise_on_missing=True to raise SystemExit like older behavior.

    Returns True if valid timestamp present, False otherwise.
    """
    try:
        ts = exec_row.get("timestamp") if isinstance(exec_row, dict) else None
        if not ts:
            try:
                # best-effort logger append (no failure if logger unavailable)
                logger_append({"stage": "execution_reflex", "error": "missing timestamp"})
            except Exception:
                pass

            # decide whether to raise or warn
            strict_env = (str(os.environ.get("STRICT_NO_CSV_TS", "")).strip() == "1")
            if raise_on_missing or strict_env:
                raise SystemExit("[FAIL:Execution Reflex Engine] missing timestamp")
            else:
                # non-fatal: print a controlled warning so people using dry-run don't get a hard abort
                try:
                    print("[WARN:Execution Reflex Engine] missing timestamp (non-fatal).")
                except Exception:
                    pass
                return False
        return True
    except SystemExit:
        # re-raise intentionally
        raise
    except Exception as _e:
        try:
            logger_append({"stage":"execution_reflex","error":str(_e)})
        except Exception:
            pass
        if raise_on_missing:
            raise
        try:
            print(f"[WARN:Execution Reflex Engine] timestamp validation error: {_e}")
        except Exception:
            pass
        return False
'''

# Replace first occurrence of the old function body (simple approach)
# Find the start index, then find next "def " after it to estimate function end.
si = txt.find(old_fn_start)
if si == -1:
    print("signature not found after all; abort")
    sys.exit(4)

# find next top-level definition after this function to cut replacement region
# crude: find "\ndef " after si+1
next_def_idx = txt.find("\ndef ", si+1)
if next_def_idx == -1:
    # fallback: replace from signature to end-of-file
    new_txt = txt[:si] + new_block
else:
    new_txt = txt[:si] + new_block + txt[next_def_idx:]

fp.write_text(new_txt, encoding="utf-8")
print("Patched _require_exec_timestamp -> non-fatal default (STRICT_NO_CSV_TS=1 preserves old behavior).")
