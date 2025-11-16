# tools/mt5_to_canonical.py
import sys, csv, pathlib as P, glob, re

# Accept: <DATE>+<TIME> OR single <TIME> that already has date+time
# Also accept headers without angle brackets (Date,Time,Open,...) and 'datetime'/'timestamp'

NAME_MAP = {
    "date": "date",
    "time": "time",
    "datetime": "datetime",
    "timestamp": "datetime",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
}

def norm_header(h):
    h = h.strip().lower()
    h = h.strip("<>")  # remove angle brackets if present
    h = h.replace(" ", "")
    return NAME_MAP.get(h, h)

def sniff_reader(f):
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
    except Exception:
        dialect = csv.excel_tab  # MT5 is usually tab-delimited
    return csv.reader(f, dialect)

def to_iso(s):
    """Make best-effort ISO: handle 'YYYY.MM.DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS'."""
    s = s.strip()
    # Replace dot dates -> dashes
    s = re.sub(r"(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", s)
    # Replace space between date/time with 'T'
    s = s.replace(" ", "T")
    # If there's no trailing Z or timezone, append Z
    if not re.search(r"[zZ]|[+\-]\d{2}:\d{2}$", s):
        s = s + "Z"
    return s

def convert_one(path: P.Path):
    p = P.Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = sniff_reader(f)
        try:
            raw_hdr = next(rdr)
        except StopIteration:
            raise RuntimeError(f"{p.name}: empty file")

        hdr = [norm_header(h) for h in raw_hdr]
        idx = {c:i for i,c in enumerate(hdr)}

        has_dt = ("datetime" in idx)
        has_date = ("date" in idx)
        has_time = ("time" in idx)
        need_prices = all(k in idx for k in ("open","high","low","close"))
        if not need_prices:
            raise RuntimeError(f"{p.name}: missing price columns; have {hdr}")

        rows_out = []
        for row in rdr:
            if not row or all((c or "").strip()=="" for c in row):
                continue
            try:
                if has_dt:
                    ts = row[idx["datetime"]]
                elif has_date and has_time:
                    ts = f"{row[idx['date']]} {row[idx['time']]}"
                elif has_time:
                    # MT5 variant where <TIME> contains date+time already
                    ts = row[idx["time"]]
                else:
                    # no usable time fields
                    continue

                iso = to_iso(ts)
                o = float(row[idx["open"]]); h = float(row[idx["high"]])
                l = float(row[idx["low"]]);  c = float(row[idx["close"]])
            except Exception:
                continue
            rows_out.append((iso, o, h, l, c))

    if not rows_out:
        raise RuntimeError(f"{p.name}: no valid rows after normalization")

    tmp = p.with_suffix(".canon.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(["time","open","high","low","close"])
        wr.writerows(rows_out)
    tmp.replace(p)

def expand_inputs(args):
    paths = []
    for a in args:
        pa = P.Path(a)
        if pa.is_dir():
            paths += [P.Path(x) for x in glob.glob(str(pa / "**" / "*_M15_*.csv"), recursive=True)]
        else:
            matched = glob.glob(a, recursive=True)
            paths += [P.Path(x) for x in matched] if matched else [pa]
    # de-dup keep order
    seen, out = set(), []
    for p in paths:
        s = str(P.Path(p))
        if s not in seen:
            seen.add(s); out.append(P.Path(p))
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/mt5_to_canonical.py <file|dir|glob> ...")
        sys.exit(2)
    for f in expand_inputs(sys.argv[1:]):
        convert_one(f)
