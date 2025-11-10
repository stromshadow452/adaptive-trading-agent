# --- add near top ---
CRYPTO = r"data/raw/crypto"

# Accept 5m/15m/1h/4h/1d variants like "BTC-USDT_15m_2024.csv"
CR_TF_MAP = {"5M":"5m","15M":"15m","1H":"1h","4H":"4h","1D":"1d"}

def infer_crypto_sym_tf(path):
    fn = os.path.basename(path).upper()
    # BTC-USDT_15m_2024.csv  OR ADA-USDT_1D_2022.csv
    m = re.match(r"^([A-Z0-9]+-[A-Z0-9]+)_(5M|15M|1H|4H|1D)_\d{4}\.CSV$", fn)
    if not m:
        # be a bit liberal if needed (e.g., lowercase tf)
        m = re.match(r"^([A-Z0-9]+-[A-Z0-9]+)_([0-9A-Z]+)_.+\.CSV$", fn)
        if not m:
            raise ValueError(f"bad crypto name: {fn}")
    sym = m.group(1)          # e.g., BTC-USDT
    tf  = CR_TF_MAP.get(m.group(2).upper(), m.group(2).lower())
    return sym, tf

def read_crypto_csv(fp, sym, tf):
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]
    # common names: timestamp or open_time; fallback to first col
    ts_col = "timestamp" if "timestamp" in df.columns else list(df.columns)[0]
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    need = ["open","high","low","close"]
    for n in need:
        if n not in df.columns:
            # some datasets use capitalized
            if n.capitalize() in df.columns:
                df.rename(columns={n.capitalize(): n}, inplace=True)
    if "volume" not in df.columns:
        # if quote_volume/base_volume exist, pick base_volume or set 0
        for cand in ["base_volume","quote_volume","volume_(btc)","vol"]:
            if cand in df.columns:
                df.rename(columns={cand:"volume"}, inplace=True)
                break
        if "volume" not in df.columns:
            df["volume"] = 0.0

    df = df.dropna(subset=["timestamp","open","high","low","close"])
    df = df[df["high"] >= df["low"]].sort_values("timestamp")
    df["symbol"] = sym
    df["timeframe"] = tf
    df["asset"] = "crypto"
    return df[["timestamp","open","high","low","close","volume","symbol","timeframe","asset"]]

def write_year_splits_any(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    for y, d in df.groupby("year", sort=True):
        fp = os.path.join(out_dir, f"{y}.parquet")
        if os.path.exists(fp):
            old = pd.read_parquet(fp)
            d = pd.concat([old, d], ignore_index=True)
        d = d.sort_values("timestamp").drop_duplicates(subset=["timestamp","symbol","timeframe"])
        d.drop(columns=["year"], inplace=True, errors="ignore")
        d.to_parquet(fp, index=False)

def ingest_crypto(root):
    print(f"[ingest] {root} (crypto)")
    files = glob.glob(os.path.join(root, "*.csv"))
    for fp in files:
        try:
            sym, tf = infer_crypto_sym_tf(fp)
            df = read_crypto_csv(fp, sym, tf)
            out_dir = os.path.join(OUT, sym, tf)
            write_year_splits_any(df, out_dir)
        except Exception as e:
            print("skip:", fp, "->", e)
