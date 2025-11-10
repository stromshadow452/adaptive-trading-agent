# make_dataset# src/make_dataset.py
from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

def build_dataset(cfg):
    logger = get_logger(cfg, name="make_dataset")
    raw_dir = Path(cfg['data']['raw_dir'])
    processed_dir = Path(cfg['data']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    tf = cfg['data'].get('default_tf','1d')
    symbols = cfg.get('symbols', [])
    lookahead = cfg.get('data', {}).get('label_lookahead', 1)

    for s in symbols:
        infile = raw_dir / f"{s}_{tf}.csv"
        if not infile.exists():
            logger.warning("Missing raw file %s", infile)
            continue
        df = pd.read_csv(infile, index_col=0, parse_dates=True).sort_index()
        df['close'] = df['Close']
        # simple EMAs
        df['ema_fast'] = df['close'].ewm(span=cfg['strategy']['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=cfg['strategy']['ema_slow'], adjust=False).mean()
        # label: next bar pct change
        df['future_return'] = df['close'].pct_change(periods=lookahead).shift(-lookahead)
        df['label'] = (df['future_return'] > 0).astype(int)
        df = df.dropna()
        out = processed_dir / f"{s}_{tf}_processed.csv"
        df.to_csv(out)
        logger.info("Processed saved %s rows to %s", len(df), out)

