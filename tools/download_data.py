import os, argparse
import pandas as pd
import yfinance as yf

OUT_DIR = "data/master"
os.makedirs(OUT_DIR, exist_ok=True)

def download_yf(sym, start, end, tf, out):
    yf_sym = f"{sym}=X" if len(sym)==6 and not sym.endswith("USDT") else sym
    df = yf.download(yf_sym, start=start, end=end, interval=tf, progress=False)
    if df.empty:
        raise RuntimeError("No data from yfinance for "+yf_sym)
    df.reset_index().rename(columns=str.title).to_csv(out, index=False)
    print("✅ Saved:", out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--tf', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()
    out = os.path.join(OUT_DIR, f"{args.symbol.upper()}_{args.tf}.csv")
    download_yf(args.symbol, args.start, args.end, args.tf, out)
# download_data.py
