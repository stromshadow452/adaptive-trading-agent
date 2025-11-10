from pathlib import Path
import re
import pandas as pd

SRC  = Path("data/raw/forex_kaggle")        # input folder
DEST = Path("data/raw/forex_kaggle_norm")   # output folder
DEST.mkdir(parents=True, exist_ok=True)

# Compact line pattern: SYMBOL,YYYYMMDD,HHMMSS,OPEN,HIGH,LOW,CLOSE,VOLUME
COMPACT_RX = re.compile(
    r"^[A-Z]{3,8},\d{8},\d{6},-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,\d+(?:\.\d+)?$"
)

def is_compact_line(s: str) -> bool:
    s = s.strip()
    if s.count(",") != 7:   # exactly 8 fields
        return False
    return bool(COMPACT_RX.match(s))

def normalize_compact(src: Path, dst: Path) -> bool:
    # Stream-parse lines (no pandas read_csv)
    dates, opens, highs, lows, closes, vols = [], [], [], [], [], []
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not is_compact_line(line):
                # skip non-data lines (headers/junk)
                continue
            sym, ymd, hms, o, h, l, c, v = line.split(",", 7)
            dates.append(ymd + hms)
            opens.append(float(o))
            highs.append(float(h))
            lows.append(float(l))
            closes.append(float(c))
            # volume can be float/int; default to 0 if empty
            try:
                vols.append(float(v))
            except Exception:
                vols.append(0.0)

    if not dates:
        print(f"⚠️  No compact rows detected in {src.name}")
        return False

    dt = pd.to_datetime(pd.Series(dates), format="%Y%m%d%H%M%S", errors="coerce")
    out = pd.DataFrame({
        "Date": dt,
        "Open": opens,
        "High": highs,
        "Low":  lows,
        "Close":closes,
        "Volume": vols,
    }).dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date")

    if out.empty:
        print(f"⚠️  Empty after compact parse: {src.name}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return True

def normalize_file(src: Path, dst: Path) -> bool:
    # Quick sniff: first 50 non-empty lines—if majority look compact, use compact parser
    compact_hits, total_seen = 0, 0
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total_seen += 1
            if is_compact_line(s):
                compact_hits += 1
            if total_seen >= 50:
                break
    if total_seen == 0:
        print(f"⚠️  File empty: {src.name}")
        return False

    if compact_hits >= max(3, total_seen // 2):  # majority compact -> compact parser
        return normalize_compact(src, dst)

    # If not compact, try a very light generic (semicolon/comma with Date,Time or Datetime)
    # Read small chunks to avoid slow inference
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(src, sep=sep, engine="python", dtype=str)
        except Exception:
            continue
        cols = [c.strip().lower().replace(" ", "") for c in df.columns]
        def pick(*names):
            for n in names:
                if n.lower() in cols:
                    return df.columns[cols.index(n.lower())]
            return None
        date = pick("date","gmttime","localtime")
        time = pick("time")
        dtc  = pick("datetime","timestamp")
        if date and time:
            s = (df[date].astype(str).str.strip() + " " + df[time].astype(str).str.strip())
            # try common formats
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y.%m.%d %H:%M","%d.%m.%Y %H:%M","%Y/%m/%d %H:%M","%d/%m/%Y %H:%M"):
                try:
                    dt = pd.to_datetime(s, format=fmt, errors="raise"); break
                except Exception:
                    dt = None
            if dt is None:
                dt = pd.to_datetime(s, errors="coerce")
        elif dtc:
            s = df[dtc].astype(str).str.strip()
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y.%m.%d %H:%M","%d.%m.%Y %H:%M","%Y/%m/%d %H:%M","%d/%m/%Y %H:%M","%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = pd.to_datetime(s, format=fmt, errors="raise"); break
                except Exception:
                    dt = None
            if dt is None:
                dt = pd.to_datetime(s, errors="coerce")
        else:
            continue

        def pickcol(name):
            if name in cols:
                return df.columns[cols.index(name)]
            return None

        O,H,L,C = (pickcol("open"), pickcol("high"), pickcol("low"), pickcol("close"))
        V = pickcol("volume") or pickcol("vol")
        if not all([O,H,L,C]):
            continue

        out = pd.DataFrame({
            "Date": dt,
            "Open": pd.to_numeric(df[O], errors="coerce"),
            "High": pd.to_numeric(df[H], errors="coerce"),
            "Low":  pd.to_numeric(df[L], errors="coerce"),
            "Close":pd.to_numeric(df[C], errors="coerce"),
            "Volume": pd.to_numeric(df[V], errors="coerce") if V else 0,
        }).dropna(subset=["Date","Open","High","Low","Close"]).sort_values("Date")

        if not out.empty:
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(dst, index=False)
            return True

    print(f"❌ Could not normalize (unknown format): {src.name}")
    return False

def main():
    files = (sorted(SRC.glob("*_M1.csv")) or 
             sorted(SRC.glob("*.csv")) or
             sorted(SRC.glob("*.txt")))
    if not files:
        print(f"⚠️  No files in {SRC}")
        return
    ok = 0
    for f in files:
        dst = DEST / (f.stem + ".csv")
        print(f"↳ normalize {f.name} → {dst.name}")
        if normalize_file(f, dst):
            ok += 1
    print(f"\n✅ Done: {ok}/{len(files)} files normalized.\nOutput: {DEST.resolve()}")

if __name__ == "__main__":
    main()
