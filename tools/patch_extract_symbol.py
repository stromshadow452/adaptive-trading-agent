#!/usr/bin/env python3
# tools/patch_extract_symbol.py
# Small surgical fixer: replace fragile regex-based main-guard insertion
# with a simple, robust literal-search insertion block.
#
# Usage:
#   python tools/patch_extract_symbol.py [path_to_file_to_fix]
#
# If no path is provided it will operate on 'tools/patch_extract_symbol.py' itself.

from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import Optional

# --- exact replacement block requested by user (must be copied exactly) ---
SAFE_INSERTION_BLOCK = r'''# --- safe insertion logic (replace broken regex-based main_guard block with this) ---
# attempt to insert replacement before the module main guard (robust, avoids fragile regex)
if pattern.search(orig):
    new = pattern.sub(replacement + "\n", orig, count=1)
    FP.write_text(new, encoding="utf-8")
    print("Patched existing _extract_symbol_from_plan block.")
else:
    # safer: look for literal main guard strings instead of a complicated regex
    idx = orig.find("if __name__ == '__main__':")
    if idx == -1:
        idx = orig.find('if __name__ == \"__main__\":')
    if idx != -1:
        # insert before the first occurrence we found
        new = orig[:idx] + replacement + "\n\n" + orig[idx:]
        FP.write_text(new, encoding="utf-8")
        print("Inserted helper block above main() guard.")
    else:
        # no main guard found — append at EOF
        FP.write_text(orig + "\n\n" + replacement, encoding="utf-8")
        print("Appended helper block at EOF.")
# --- end safe insertion logic ---
'''

# Note: SAFE_INSERTION_BLOCK contains the exact text you asked to insert.
# We will search for the fragile branch and replace it with this block.

# Heuristics to find the fragile "regex-based main_guard" branch in the target file.
# We'll look for a common pattern starting point and then attempt to find the
# corresponding block to replace. This is intentionally conservative to avoid
# accidental damage.

SEARCH_START_PATTERNS = [
    r"main_guard\s*=\s*re\.search\(",        # the explicit broken line you mentioned
    r"if\s+pattern\.search\(\s*orig\s*\)\s*:",  # another common variant
    r"#\s*broken\s*regex-based\s*main_guard",   # comments that might exist
]

def find_existing_block_bounds(text: str) -> Optional[tuple[int,int]]:
    """
    Try to find the existing fragile block to replace.
    Returns (start_idx, end_idx) in the text if found, else None.
    Heuristic approach:
      - find first occurrence of any SEARCH_START_PATTERNS
      - then expand until we hit a following blank line + a non-indented line
        or until an 'else:' followed by a dedented block ends; fallback to
        replacing up to the next two blank lines to be safe.
    """
    for pat in SEARCH_START_PATTERNS:
        m = re.search(pat, text)
        if not m:
            continue
        start = m.start()
        # From start, find a reasonable end: search for a marker that probably
        # denotes the end of that insertion branch. We'll look for:
        #   - a line that starts with "# --- end" (explicit marker)
        #   - or a blank line followed by a non-indented line (dedent)
        #   - or the end of file
        # We operate line-wise for easier heuristics.
        lines = text.splitlines(keepends=True)
        # Determine which line index the start char is on
        acc = 0
        start_line_idx = 0
        for i, ln in enumerate(lines):
            acc += len(ln)
            if acc > start:
                start_line_idx = i
                break
        # Now scan forward
        end_line_idx = start_line_idx
        for j in range(start_line_idx, len(lines)):
            ln = lines[j]
            # explicit marker
            if re.search(r"#\s*---\s*end\s*safe\s*insertion\s*logic", ln, re.IGNORECASE):
                end_line_idx = j
                # include this line
                end_line_idx = min(end_line_idx + 1, len(lines)-1)
                break
            # blank line followed by dedent
            if ln.strip() == "":
                # lookahead for a dedented, non-empty line
                if j+1 < len(lines):
                    nxt = lines[j+1]
                    if nxt and (not nxt.startswith((" ", "\t"))):
                        end_line_idx = j
                        break
            # also stop if we see "if __name__ == '__main__':" which likely follows
            if "if __name__" in ln:
                end_line_idx = j
                break
            # also stop if we see another top-level def or class
            if re.match(r"^(def |class )", ln):
                end_line_idx = j
                break
        # compute char indices for replacement
        prefix = "".join(lines[:start_line_idx])
        suffix_from = "".join(lines[end_line_idx:])
        start_char = len(prefix)
        end_char = len(text) - len(suffix_from)
        if start_char < end_char:
            return (start_char, end_char)
    return None

def apply_replacement(fp: Path, replacement_block: str) -> None:
    """
    Read file, attempt to replace fragile branch with replacement_block.
    If we can't find the fragile branch, attempt safer insertions:
      - find literal main guard lines and insert before them
      - otherwise append at EOF
    Prints one of the exact messages you asked to show.
    """
    orig = fp.read_text(encoding="utf-8")
    # Try a direct targeted replacement using our heuristics
    bounds = find_existing_block_bounds(orig)
    if bounds:
        s, e = bounds
        new_text = orig[:s] + replacement_block + "\n" + orig[e:]
        fp.write_text(new_text, encoding="utf-8")
        # Choose printed message consistent with expected outputs
        print("Patched existing _extract_symbol_from_plan block.")
        return

    # If we couldn't locate, attempt literal searches for pattern / main guard
    # We'll use variables similar to the block's variables so the block matches context.
    # Emulate what the block would have expected: pattern, orig, replacement, FP
    # But since we're editing the file, just perform safe literal insertion.
    # Try: find main guard literal
    idx = orig.find("if __name__ == '__main__':")
    if idx == -1:
        idx = orig.find('if __name__ == "__main__":')
    if idx != -1:
        new = orig[:idx] + replacement_block + "\n\n" + orig[idx:]
        fp.write_text(new, encoding="utf-8")
        print("Inserted helper block above main() guard.")
        return

    # As a last resort append
    fp.write_text(orig + "\n\n" + replacement_block, encoding="utf-8")
    print("Appended helper block at EOF.")

def main():
    # default to operating on this file itself if no arg given,
    # so you can run the script once to ensure the safe block is present.
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__)
    if not target.exists():
        print(f"Target file does not exist: {str(target)}", file=sys.stderr)
        sys.exit(2)

    # For cleanliness: ensure we don't accidentally double-insert the block if it already exists.
    text = target.read_text(encoding="utf-8")
    if SAFE_INSERTION_BLOCK.strip() in text:
        print("Safe insertion block already present; nothing to do.")
        return

    # We'll call apply_replacement which prints one of the specific messages
    apply_replacement(target, SAFE_INSERTION_BLOCK)

if __name__ == "__main__":
    main()
