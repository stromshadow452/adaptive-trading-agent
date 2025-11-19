import os
import tempfile
import json
import time

_LOCK_SUFFIX = ".lock"

def atomic_append_json_lines(path, records, max_attempts=10, sleep=0.1):
    """
    Append JSON-lines (records: list[dict]) to `path` atomically using tmp file + rename.
    Uses a simple lock file to avoid concurrent writers.
    """
    dirpath = os.path.dirname(path) or "."
    lock = path + _LOCK_SUFFIX
    attempt = 0
    while attempt < max_attempts:
        try:
            # acquire lock (naive)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(sleep)
            attempt += 1
    else:
        raise RuntimeError(f"Could not acquire lock for {path}")

    try:
        # write to temp file in same dir
        fd, tmp = tempfile.mkstemp(prefix=".tmp_exec_", dir=dirpath, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # if target exists, copy existing content first
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as old:
                    for line in old:
                        f.write(line)
            # append new records as JSON-lines
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        # atomic replace
        os.replace(tmp, path)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass
