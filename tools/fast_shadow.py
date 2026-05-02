"""
tools/fast_shadow.py
=====================
SCOPUS Fast Shadow Simulator — Week 17.

Replays historical EURUSD H1 bars through PipelineV2 to generate
the log files needed by shadow_report.py and live_readiness.py
WITHOUT waiting 30 real days.

Default:  720 H1 bars  ≈  30 calendar days  ≈  15-30 seconds runtime.

Output files (all appended, safe to re-run):
  logs/shadow/fills.jsonl
  logs/shadow/psi_daily.jsonl
  logs/shadow/ensemble.jsonl

Usage:
    python tools/fast_shadow.py                   # 30 days EURUSD H1
    python tools/fast_shadow.py --days 60
    python tools/fast_shadow.py --dry-run          # no log files written
    python tools/fast_shadow.py --data-file data/raw/forex_backup_2020_2025/EURUSD_H1_2024_to_2025.csv
    python tools/fast_shadow.py --symbol XAGUSD
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import hashlib

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    def _tqdm(it, **kw):   # noqa: E301  bare fallback — no progress bar
        return it

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
LOG = logging.getLogger("fast_shadow")


def file_sha256(path: str) -> Optional[str]:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

# Default data paths (searched in order)
_DEFAULT_DATA_CANDIDATES = [
    "data/raw/forex_backup_2020_2025/EURUSD_H1_2024_to_2025.csv",
    "data/raw/forex_backup_2020_2025/EURUSD_H1_2023_to_2024.csv",
    "data/raw/forex_kaggle_multiTF/EURUSD_H1.csv",
    "data/raw/M15_only/EURUSD_M15_2024_to_2025.csv",
]

BARS_PER_DAY = {"H1": 24, "M15": 96, "M5": 288, "H4": 6, "D1": 1}

# ---------------------------------------------------------------------------
# Multi-symbol data registry
# ---------------------------------------------------------------------------

# Ordered candidates (first existing file wins per symbol).
# Backup files (2024-2025) are preferred for multi-symbol mode because they
# all share the same calendar period. Kaggle files used as fallback only.
SYMBOL_DATA_CANDIDATES: Dict[str, List[str]] = {
    "EURUSD": [
        "data/raw/forex_backup_2020_2025/EURUSD_H1_2024_to_2025.csv",
        "data/raw/forex_kaggle_multiTF/EURUSD_H1.csv",
    ],
    "GBPUSD": [
        "data/raw/forex_backup_2020_2025/GBPUSD_H1_2024_to_2025.csv",
        "data/raw/forex_kaggle_multiTF/GBPUSD_H1.csv",
    ],
    "USDJPY": [
        "data/raw/forex_backup_2020_2025/USDJPY_H1_2024_to_2025.csv",
        "data/raw/forex_kaggle_multiTF/USDJPY_H1.csv",
    ],
    "USDCAD": [
        "data/raw/forex_backup_2020_2025/USDCAD_H1_2024_to_2025.csv",
        "data/raw/forex_kaggle_multiTF/USDCAD_H1.csv",
    ],
    "AUDUSD": [
        "data/raw/forex_kaggle_multiTF/AUDUSD_H1.csv",
    ],
    "NZDUSD": [
        "data/raw/forex_kaggle_multiTF/NZDUSD_H1.csv",
    ],
    "XAGUSD": [
        "data/raw/forex_kaggle_multiTF/XAGUSD_H1.csv",
    ],
    "XAUUSD": [
        "data/raw/forex_kaggle_multiTF/XAUUSD_H1.csv",
        "data/raw/forex_backup_2020_2025/XAUUSD_H1_2024_to_2025.csv",
    ],
}

ALL_SYMBOLS: List[str] = list(SYMBOL_DATA_CANDIDATES.keys())


# ---------------------------------------------------------------------------
# Bar namedtuple-compatible object
# ---------------------------------------------------------------------------

def _scalar(v) -> float:
    """Safely convert a pandas scalar, Series, or array element to float."""
    try:
        return float(v)
    except TypeError:
        try:
            return float(v.item())
        except Exception:
            return float(v.iloc[0] if hasattr(v, "iloc") else v)


class _Bar:
    """Minimal bar object matching PipelineV2 expectations."""
    __slots__ = ("open", "high", "low", "close", "volume", "time")

    def __init__(self, open, high, low, close, volume, time=None):
        self.open   = _scalar(open)
        self.high   = _scalar(high)
        self.low    = _scalar(low)
        self.close  = _scalar(close)
        self.volume = _scalar(volume)
        self.time   = time


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load and normalise an OHLCV CSV file.

    Handles:
    - MT4/MT5 tab-separated exports: <DATE>\t<TIME>\t<OPEN>\t<HIGH>...
    - Standard comma-separated CSVs with open/high/low/close columns
    - BOMs and mixed case headers
    """
    LOG.info(f"[fast_shadow] Loading {path}")

    # Detect separator by peeking at the first line
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        first_line = fh.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)

    # Strip BOM / angle brackets / whitespace from column names
    df.columns = [
        c.strip().lstrip("\ufeff").strip("<>").lower().replace(" ", "_")
        for c in df.columns
    ]

    # MT4/MT5: separate date and time columns → combine into index
    if "date" in df.columns and "time" in df.columns:
        try:
            dt_str = df["date"].astype(str).str.strip() + " " + \
                     df["time"].astype(str).str.strip()
            df.index = pd.to_datetime(dt_str, dayfirst=False, errors="coerce")
            df.index = df.index.tz_localize("UTC")
            df = df.drop(columns=["date", "time"], errors="ignore")
        except Exception as e:
            LOG.warning(f"[fast_shadow] Date/time parse failed: {e}")

    # Kaggle-style: single 'date' or 'datetime' column (no separate time col)
    elif "date" in df.columns:
        try:
            df.index = pd.to_datetime(
                df["date"].astype(str).str.strip(), errors="coerce", utc=True
            )
            df = df.drop(columns=["date"], errors="ignore")
        except Exception as e:
            LOG.warning(f"[fast_shadow] Single date column parse failed: {e}")
    elif "datetime" in df.columns:
        try:
            df.index = pd.to_datetime(
                df["datetime"].astype(str).str.strip(), errors="coerce", utc=True
            )
            df = df.drop(columns=["datetime"], errors="ignore")
        except Exception as e:
            LOG.warning(f"[fast_shadow] Datetime column parse failed: {e}")

    # Drop unnamed index column if present
    for col in list(df.columns):
        if col.startswith("unnamed"):
            df = df.drop(columns=[col])

    # Rename aliases → standard names
    alias = {
        "o": "open",   "h": "high",  "l": "low", "c": "close",
        "v": "volume", "vol": "volume", "tickvol": "volume",
        "tick_volume": "volume", "spread": None,  # discard spread
    }
    rename = {}
    drop   = []
    for col in df.columns:
        if col in alias:
            if alias[col] is None:
                drop.append(col)
            else:
                rename[col] = alias[col]
    df = df.drop(columns=drop, errors="ignore").rename(columns=rename)

    # If index not yet a datetime, try the first column
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        except Exception:
            pass

    required = {"open", "high", "low", "close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}\n"
            f"  Available: {list(df.columns)}\n"
            f"  File: {path}\n"
            f"  Tip: expected standard OHLCV or MT4/MT5 tab-exported CSV."
        )

    if "volume" not in df.columns:
        df["volume"] = 1000.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    if df.empty:
        raise ValueError(f"No valid rows after parsing: {path}")

    LOG.info(f"[fast_shadow] Loaded {len(df):,} bars from {Path(path).name}")
    return df


def find_data_file(symbol: str, tf: str, user_path: Optional[str]) -> str:
    """Find a suitable CSV data file."""
    if user_path and Path(user_path).exists():
        return user_path

    # Try auto-detection
    candidates = _DEFAULT_DATA_CANDIDATES.copy()
    if symbol != "EURUSD":
        for base in _DEFAULT_DATA_CANDIDATES:
            candidates.insert(0, base.replace("EURUSD", symbol))

    for c in candidates:
        if Path(c).exists():
            return c

    raise FileNotFoundError(
        f"No data file found for {symbol} {tf}.\n"
        f"Tried: {candidates[:3]}\n"
        f"Use --data-file to point to your CSV."
    )


# ---------------------------------------------------------------------------
# Rolling window DataSource wrapper (for PipelineV2 get_history)
# ---------------------------------------------------------------------------

class _CSVDataSource:
    """
    Minimal DataSourceABC-compatible wrapper used by PipelineV2.
    Provides get_history() returning the last n_bars as a DataFrame.
    """
    def __init__(self, df: pd.DataFrame, symbol: str = "EURUSD"):
        self._df    = df
        self.symbol = symbol
        self._cursor = 0    # current bar index

    def advance(self, idx: int):
        self._cursor = idx

    def get_history(self, symbol: str, timeframe, n_bars: int = 250) -> pd.DataFrame:
        end = self._cursor + 1
        start = max(0, end - n_bars)
        return self._df.iloc[start:end].copy()

    def stream(self, symbol: str, timeframe):
        for i in range(len(self._df)):
            self.advance(i)
            row = self._df.iloc[i]
            yield _Bar(row.open, row.high, row.low, row.close, row.volume,
                       time=self._df.index[i])


# ---------------------------------------------------------------------------
# boll_z helper (for portfolio scoring — reuses raw data, no new indicator)
# ---------------------------------------------------------------------------

def _compute_boll_z(df: pd.DataFrame, period: int = 20) -> float:
    """
    Compute Bollinger band z-score from the last row of an OHLCV DataFrame.
    Mirrors the calculation in src/features/common_features.py.
    Used ONLY for portfolio ranking — pipeline computes it internally as well.
    """
    if len(df) < period:
        return 0.0
    close = df["close"]
    mid   = close.rolling(period).mean().iloc[-1]
    std   = close.rolling(period).std().iloc[-1]
    if std < 1e-10:
        return 0.0
    return float((close.iloc[-1] - mid) / std)


def _compute_atr_pctile(df: pd.DataFrame, period: int = 14, window: int = 100) -> float:
    """
    ATR percentile rank of the current bar within the last `window` bars.
    Mirrors the atr_pctile feature in src/features/common_features.py.
    Used ONLY for portfolio edge filtering — no new indicator introduced.
    """
    if len(df) < period + 1:
        return 0.0
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr    = tr.rolling(period).mean()
    tail   = atr.dropna().tail(window)
    if len(tail) < 2:
        return 0.5
    current = float(atr.iloc[-1])
    if current != current:   # NaN guard
        return 0.0
    return float((tail < current).sum() / len(tail))


# ---------------------------------------------------------------------------
# Dry-run executor (no file writes)
# ---------------------------------------------------------------------------

class _DryRunExecutor:
    def check_sl_tp(self, *a, **kw): pass
    def get_open_positions(self): return {}
    def execute(self, signal): return None
    def get_summary(self):
        return {"equity": 10000, "net_pnl_usd": 0, "n_trades": 0,
                "win_rate": 0, "profit_factor": 0, "drawdown_pct": 0,
                "avg_slippage_pips": 0}


def _make_capture_exec(shared_executor, capture_list: list):
    """
    Returns a capturing executor that intercepts execute() calls.
    Stores the signal in capture_list without executing.
    SL/TP and position queries delegate to the real shared executor.
    """
    class _CaptureExec:
        def check_sl_tp(self, s, h, l):  pass
        def get_open_positions(self):     return shared_executor.get_open_positions()
        def execute(self, sig):           capture_list.append(dict(sig)); return None
        def get_summary(self):            return shared_executor.get_summary()
    return _CaptureExec()


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_fast_shadow(
    data_path:  str,
    symbol:     str     = "EURUSD",
    tf:         str     = "H1",
    days:       int     = 30,
    cfg_path:   str     = "config/weapon_system.yaml",
    dry_run:    bool    = False,
    verbose:    bool    = True,
) -> dict:
    """
    Run the fast-forward shadow simulation.

    Returns:
        dict with final metrics summary.
    """
    import yaml
    bars_per_day = BARS_PER_DAY.get(tf.upper(), 24)
    max_bars     = days * bars_per_day

    # Load config
    cfg = {}
    cfg_hash = file_sha256(cfg_path)
    if Path(cfg_path).exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    # Load data
    df = load_csv(data_path)
    if len(df) < 200:
        raise ValueError(f"Not enough bars ({len(df)}) — need ≥ 200")

    # Use last `max_bars` only
    total_available = len(df)
    df_sim = df.iloc[-max_bars:].copy()
    LOG.info(f"[fast_shadow] Simulating {len(df_sim):,} bars ({days} days at {tf})")

    # Build PipelineV2
    from src.pipeline.pipeline_v2 import PipelineV2, PipelineConfig
    pcfg     = PipelineConfig.from_cfg(cfg)
    pipeline = PipelineV2(cfg=pcfg, cfg_yaml=cfg)

    # Build executor
    if dry_run:
        executor = _DryRunExecutor()
        LOG.info("[fast_shadow] DRY RUN — no files will be written")
    else:
        Path("logs/shadow").mkdir(parents=True, exist_ok=True)
        fills_log = cfg.get("fills_log", "logs/shadow/fills.jsonl")
        from src.broker.paper_executor import PaperExecutor
        executor = PaperExecutor(log_path=fills_log, starting_equity=10_000.0)

    # Wrap data source
    ds = _CSVDataSource(df, symbol)

    # ── Simulation loop ───────────────────────────────────────────────────────
    bar_count    = 0
    trade_count  = 0
    error_count  = 0
    t_start      = time.time()
    pending_signal: Optional[dict] = None   # signal from bar T → executes at bar T+1 open

    full_df_for_history = df.copy()
    ds_full = _CSVDataSource(full_df_for_history, symbol)

    # Find where our simulation window starts in the full df
    sim_start_ts  = df_sim.index[0]
    sim_start_idx = 0
    for i, ts in enumerate(full_df_for_history.index):
        if ts >= sim_start_ts:
            sim_start_idx = i
            break

    for sim_i in range(len(df_sim)):
        abs_i = sim_start_idx + sim_i
        ds_full.advance(abs_i)
        row = df_sim.iloc[sim_i]
        bar = _Bar(row.open, row.high, row.low, row.close, row.volume,
                   time=df_sim.index[sim_i])
        raw_df = ds_full.get_history(symbol, tf, n_bars=250)

        # ── Execute PENDING signal from previous bar at THIS bar's open ────
        # Strict next-bar open fill (correct walk-forward execution).
        if pending_signal is not None and not dry_run:
            pending_signal["price"] = float(row.open)   # override to next-bar open
            pending_signal.setdefault("metadata", {})
            if isinstance(pending_signal.get("metadata"), dict):
                pending_signal["metadata"]["bar_time"] = bar.time.isoformat() if hasattr(bar.time, "isoformat") else str(bar.time)
            try:
                fill = executor.execute(pending_signal)
                if fill is not None:
                    trade_count += 1
                    if verbose:
                        LOG.info(
                            f"  FILL [{bar_count:4d}] {symbol} {pending_signal['side'].upper()} "
                            f"@ {fill.fill_px:.5f}  "
                            f"slip={fill.slippage_pips:.2f}pip  "
                            f"regime={pending_signal.get('regime','?')} [next-open]"
                        )
            except Exception as e:
                LOG.debug(f"[fast_shadow] Pending execute error: {e}")
            pending_signal = None

        bar_count += 1
        try:
            # Use a capturing executor: pipeline generates the signal dict,
            # we intercept it and store as pending — real fill happens at next-bar open.
            if dry_run:
                result = pipeline.process_bar(symbol, bar, raw_df, executor)
            else:
                # Temporarily wrap executor so SL/TP runs on real positions,
                # but new execute() calls are captured, not filled at current close.
                class _CapturingExecutor:
                    """Intercepts execute() — defers fill to next-bar open."""
                    def check_sl_tp(self, sym, hi, lo, bar_time=None):
                        executor.check_sl_tp(sym, hi, lo, bar_time=bar_time)
                    def get_open_positions(self): return executor.get_open_positions()
                    def execute(self, sig):
                        nonlocal pending_signal
                        # Only capture if no position open and no pending already
                        if pending_signal is None and sym not in executor.get_open_positions():
                            pending_signal = dict(sig)
                        return None
                    def get_summary(self): return executor.get_summary()
                sym = symbol
                result = pipeline.process_bar(symbol, bar, raw_df, _CapturingExecutor())

            if result.errors:
                error_count += 1

        except Exception as e:
            error_count += 1
            LOG.debug(f"[fast_shadow] Bar {bar_count} error: {e}")

        if verbose and bar_count % 200 == 0:
            elapsed   = time.time() - t_start
            remaining = (len(df_sim) - bar_count) / max(bar_count, 1) * elapsed
            LOG.info(
                f"[fast_shadow] Bar {bar_count:4d}/{len(df_sim)}  "
                f"trades={trade_count}  errors={error_count}  "
                f"elapsed={elapsed:.1f}s  ETA={remaining:.0f}s"
            )

    elapsed = time.time() - t_start

    # Final summary
    summary = executor.get_summary()
    summary.update({
        "symbol":        symbol,
        "timeframe":     tf,
        "days_simulated": days,
        "config_sha256": cfg_hash,
        "bars":          bar_count,
        "trades":        trade_count,
        "errors":        error_count,
        "features":      pipeline.feature_count,
        "elapsed_s":     round(elapsed, 2),
        "bars_per_sec":  round(bar_count / max(elapsed, 0.1), 1),
        "pipeline":      "v2",
    })

    LOG.info("=" * 60)
    LOG.info(f"[fast_shadow] SIMULATION COMPLETE")
    LOG.info(f"  Symbol:     {symbol} {tf}")
    LOG.info(f"  Bars:       {bar_count:,}  ({days} days)")
    LOG.info(f"  Trades:     {trade_count}")
    LOG.info(f"  Features:   {pipeline.feature_count}")
    LOG.info(f"  Win rate:   {summary.get('win_rate', 0):.1%}")
    LOG.info(f"  PF:         {summary.get('profit_factor', 0):.3f}")
    LOG.info(f"  Net PnL:    ${summary.get('net_pnl_usd', 0):+,.2f}")
    LOG.info(f"  Drawdown:   {summary.get('drawdown_pct', 0):.2f}%")
    LOG.info(f"  Speed:      {summary['bars_per_sec']} bars/sec")
    LOG.info(f"  Elapsed:    {elapsed:.1f}s")
    LOG.info("=" * 60)

    return summary


df_for_history_idx = 0   # module-level (used in main)


# ---------------------------------------------------------------------------
# TRUE Multi-symbol synchronized simulation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 1.1 helpers — Dual-Speed Loss Detection + Regime Persistence
# ---------------------------------------------------------------------------

# Only pre-approved symbols are allowed to generate trades.
PRE_APPROVED_SYMBOLS = frozenset({"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD"})


class _StrategyBlacklist:
    """
    DUAL-SPEED loss detection (Fast + Slow) — Per Strategy.

    FAST signal:  3 consecutive losses on a strategy → 24-bar cooldown
                  Reacts within hours. Prevents streak damage.
    SLOW signal:  Rolling PF over last 15 trades < 0.80 → hard block
                  Reacts over days. Catches chronic losers in specific regimes.

    Combined: fast catches streaks, slow catches structural failure.
    Tracking is per (symbol, strategy) so one bad strategy doesn't kill the symbol.
    """

    def __init__(self, slow_window: int = 15, pf_floor: float = 0.80,
                 fast_consec: int = 3, cooldown_bars: int = 24):
        self.slow_window  = slow_window
        self.pf_floor     = pf_floor
        self.fast_consec  = fast_consec
        self.cooldown_bars = cooldown_bars
        self._trades: Dict[str, list] = {}          # key → [pnl, ...]
        self._consec_losses: Dict[str, int] = {}    # key → consecutive loss count
        self._cooldown_until: Dict[str, int] = {}   # key → bar_count when cooldown ends

    def record_trade(self, sym: str, strategy: str, pnl: float) -> None:
        key = f"{sym}_{strategy}"
        self._trades.setdefault(key, []).append(pnl)
        if len(self._trades[key]) > self.slow_window:
            self._trades[key] = self._trades[key][-self.slow_window:]

        # Fast consecutive-loss tracking
        if pnl < 0:
            self._consec_losses[key] = self._consec_losses.get(key, 0) + 1
        else:
            self._consec_losses[key] = 0  # reset on win

    def is_blacklisted(self, sym: str, strategy: str, current_bar: int = 0) -> str:
        """
        Returns reason string if blocked, empty string if OK.
        Order: Cooldown expiry → Fast kill → Active cooldown → Slow PF.
        """
        key = f"{sym}_{strategy}"
        # STEP 1: Clear expired cooldowns FIRST (before any kill check)
        if key in self._cooldown_until:
            if current_bar >= self._cooldown_until[key]:
                # Cooldown served → clean slate
                del self._cooldown_until[key]
                self._consec_losses[key] = 0
            else:
                # Still in cooldown
                return "COOLDOWN"

        # STEP 2: FAST kill — 3 consecutive losses → start new cooldown
        if self._consec_losses.get(key, 0) >= self.fast_consec:
            self._cooldown_until[key] = current_bar + self.cooldown_bars
            return "FAST_KILL"

        # STEP 3: SLOW — rolling PF check
        trades = self._trades.get(key, [])
        if len(trades) >= 5:
            gp = sum(p for p in trades if p > 0) or 1e-9
            gl = abs(sum(p for p in trades if p < 0)) or 1e-9
            if gp / gl < self.pf_floor:
                return "SLOW_PF"

        return ""   # OK

    def status(self) -> Dict[str, str]:
        out = {}
        for sym, trades in self._trades.items():
            consec = self._consec_losses.get(sym, 0)
            n = len(trades)
            if n < 5:
                pf_str = f"({n} trades)"
            else:
                gp = sum(p for p in trades if p > 0) or 1e-9
                gl = abs(sum(p for p in trades if p < 0)) or 1e-9
                pf_str = f"PF={gp/gl:.2f}"
            out[sym] = f"{pf_str} consec_L={consec}"
        return out


class _RegimePersistence:
    """
    Tracks per-symbol regime_score history.
    A trade is only allowed if score > threshold for N consecutive bars.
    """

    def __init__(self, required_bars: int = 2, threshold: float = 0.35):
        self.required    = required_bars
        self.threshold   = threshold
        self._history: Dict[str, list] = {}   # sym → [score, score, ...]

    def record(self, sym: str, score: float) -> None:
        self._history.setdefault(sym, []).append(score)
        # Keep a small fixed window so 2-of-3 logic has enough history.
        if len(self._history[sym]) > 3:
            self._history[sym] = self._history[sym][-3:]

    def is_confirmed(self, sym: str) -> bool:
        """
        True if persistence is satisfied.

        Phase 1 unblock:
          - required_bars is kept at 2 (minimum)
          - confirmation uses a 2-of-3 rule when possible to avoid brittle blocking
        """
        h = self._history.get(sym, [])
        if len(h) < self.required:
            return False
        # Require at least `required` observations, but score confirmation on up to 3 bars.
        if len(h) < self.required:
            return False
        last3 = h[-3:]
        return sum(1 for s in last3 if s > self.threshold) >= 2


def _tiered_size_mult(regime_score: float) -> float:
    """
    Tiered conviction sizing — replaces score^1.5 curve.

    The continuous curve penalised the middle of the distribution
    (mean score ~0.48 → only 33% of base). Tiers preserve the profitable middle.

    Tier A: score ≥ 0.60 → 100% (strong MR environment)
    Tier B: score ≥ 0.45 → 80%  (decent conditions — most EURUSD trades)
    Tier C: score ≥ 0.35 → 55%  (marginal — reduced but not killed)
    Below:  → 30%  (minimum viable — persistence guard should catch most)
    """
    if regime_score >= 0.60:
        return 1.00
    if regime_score >= 0.45:
        return 0.80
    if regime_score >= 0.35:
        return 0.55
    return 0.30


def run_multi_symbol_shadow(
    symbols:  List[str] = None,
    tf:       str       = "H1",
    days:     int       = 90,
    cfg_path: str       = "config/weapon_system.yaml",
    dry_run:  bool      = False,
    verbose:  bool      = True,
) -> dict:
    """
    TRUE multi-symbol portfolio simulation over a SHARED timeline.

    Phase 1 upgrades (portfolio layer only — zero pipeline changes):
        • Symbol blacklist:     rolling PF < 0.80 over last 20 trades → skip
        • Score-based sizing:   size *= regime_score^1.5
        • Regime persistence:   score > 0.65 for 3 consecutive bars → allow

    Architecture per bar T:
        1. Execute pending signals (from T-1) at T's open price
        2. Check SL/TP for all open positions
        3. Collect signals from ALL symbols via unchanged PipelineV2
        4. Rank + correlation guard + select top 3
        5. Store as pending  →  fill at T+1 open

    Zero changes to PipelineV2, PaperExecutor, entry logic, SL/TP.
    """
    import yaml
    from src.pipeline.pipeline_v2 import PipelineV2, PipelineConfig
    from src.broker.paper_executor import PaperExecutor
    from src.risk.circuit_breaker import CircuitBreaker
    from src.risk.risk_engine import RealTimeRiskEngine
    from tools.portfolio_selector import select_signals, compute_score, get_group
    from src.pipeline.regime_classifier import compute_regime_score

    if symbols is None:
        symbols = ALL_SYMBOLS

    # ── Pre-approval filter: only trade symbols with PROVEN edge ──────
    # Symbols not in PRE_APPROVED_SYMBOLS are silently dropped.
    # To expand: add symbol after it passes symbol_audit.py with PF ≥ 1.10
    symbols = [s for s in symbols if s.upper() in PRE_APPROVED_SYMBOLS]
    if not symbols:
        symbols = list(PRE_APPROVED_SYMBOLS)  # fallback to approved set
    LOG.info(f"[multi] Pre-approved filter → {symbols}")

    bars_per_day = BARS_PER_DAY.get(tf.upper(), 24)
    max_bars     = days * bars_per_day
    HISTORY_BARS = 250    # bars of history required by pipeline

    # Load config
    cfg = {}
    if Path(cfg_path).exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    # ── 1. Load data for all requested symbols ────────────────────────────
    # Alignment: all symbols must have data in this window.
    # Symbols without data in [ALIGN_START, ALIGN_END] are auto-skipped.
    ALIGN_START = pd.Timestamp("2023-01-01", tz="UTC")
    ALIGN_END   = pd.Timestamp("2023-12-31", tz="UTC")

    sym_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sym_up = sym.upper()
        for cand in SYMBOL_DATA_CANDIDATES.get(sym_up, []):
            if Path(cand).exists():
                try:
                    raw = load_csv(cand)

                    # Period alignment filter ─────────────────────────────
                    aligned = raw[(raw.index >= ALIGN_START) &
                                  (raw.index <= ALIGN_END)]

                    if len(aligned) < HISTORY_BARS + 10:
                        LOG.warning(
                            f"[multi] {sym_up}: only {len(aligned)} bars in "
                            f"{ALIGN_START.date()} – {ALIGN_END.date()} "
                            f"(from {Path(cand).name}) — skipping"
                        )
                        continue   # try next candidate file

                    sym_dfs[sym_up] = aligned
                    LOG.info(
                        f"[multi] {sym_up}: {len(aligned):,} bars  "
                        f"({aligned.index[0].date()} → {aligned.index[-1].date()})  "
                        f"[{Path(cand).name}]"
                    )
                    break
                except Exception as e:
                    LOG.warning(f"[multi] Failed to load {cand}: {e}")
        if sym_up not in sym_dfs:
            LOG.warning(f"[multi] {sym_up}: no data in 2024–2025 → skipping")

    valid_symbols = list(sym_dfs.keys())
    if len(valid_symbols) < 1:
        raise ValueError(f"No symbol data found. Tried: {symbols}")

    LOG.info(f"[multi] Loaded {len(valid_symbols)} symbols: {valid_symbols}")

    # ── 2. Build COMMON timeline (timestamp intersection) ─────────────────
    common_idx = sym_dfs[valid_symbols[0]].index
    for sym in valid_symbols[1:]:
        common_idx = common_idx.intersection(sym_dfs[sym].index)

    common_idx = common_idx.sort_values()
    LOG.info(f"[multi] Common timeline: {len(common_idx):,} bars across {len(valid_symbols)} symbols")

    if len(common_idx) < HISTORY_BARS + 10:
        raise ValueError(
            f"Only {len(common_idx)} common bars — need ≥ {HISTORY_BARS + 10}.\n"
            "Symbols may have mismatched date ranges. "
            "Check that all data files cover the same period."
        )

    # Trim to simulation window (last max_bars of the common timeline)
    sim_ts: pd.DatetimeIndex = (
        common_idx[-max_bars:] if len(common_idx) > max_bars else common_idx
    )
    LOG.info(
        f"[multi] Simulation window: {len(sim_ts):,} bars  "
        f"({sim_ts[0]} → {sim_ts[-1]})"
    )

    # Build fast timestamp→integer-index maps per symbol
    sym_ts_idx: Dict[str, Dict] = {
        sym: {ts: i for i, ts in enumerate(sym_dfs[sym].index)}
        for sym in valid_symbols
    }

    # ── 3. Build per-symbol PipelineV2 instances ──────────────────────────
    pcfg = PipelineConfig.from_cfg(cfg)
    sym_pipelines: Dict[str, PipelineV2] = {
        sym: PipelineV2(cfg=pcfg, cfg_yaml=cfg)
        for sym in valid_symbols
    }

    # ── 4. Build ONE shared executor ──────────────────────────────────────
    if dry_run:
        executor = _DryRunExecutor()
        LOG.info("[multi] DRY RUN — no files written")
    else:
        Path("logs/shadow").mkdir(parents=True, exist_ok=True)
        fills_log = cfg.get("fills_log_multi", "logs/shadow/fills_multi.jsonl")
        executor  = PaperExecutor(log_path=fills_log, starting_equity=10_000.0)

    # ── 5. Simulation loop (ONE loop — synchronized) ──────────────────────
    bar_count         = 0
    trade_count       = 0
    error_count       = 0
    blacklist_blocks  = 0
    persist_blocks    = 0
    trades_per_sym:   Dict[str, int] = {sym: 0 for sym in valid_symbols}
    ml_veto_blocks    = 0
    trades_per_group: Dict[str, int] = {"USD": 0, "JPY": 0, "METALS": 0, "OTHER": 0}
    pending_signals:  Dict[str, dict] = {}   # sym -> signal (fill at next-bar open)

    # ── Phase 1 / JARVIS tracking logic ──────────────────────────────────
    # Use an isolated risk state file for historical simulation runs.
    # The default shared `logs/risk_state.json` can contain cooldown timestamps
    # in the (real) future relative to backtest timestamps, causing all trades
    # to be blocked.
    # Use a per-run isolated risk state file to avoid timeline skew between
    # historical bar timestamps (e.g., 2023) and wall-clock timestamps that can
    # appear in persisted state from other runs.
    risk_engine = RealTimeRiskEngine(state_file=f"logs/shadow/risk_state_sim_{int(time.time())}.json")

    # Phase 1.1 components (dual-speed detection + loosened persistence)
    blacklist     = _StrategyBlacklist(
        slow_window=15, pf_floor=0.80,
        fast_consec=3, cooldown_bars=24,
    )
    base_persist_required = 2
    base_persist_threshold = 0.25
    persistence   = _RegimePersistence(required_bars=base_persist_required, threshold=base_persist_threshold)
    fast_blocks   = 0
    slow_pf_blocks = 0     # counted but NOT hard-blocked for approved symbols

    # Phase 2 components (JARVIS adaptive engine)
    from src.strategies.strategy_bank import StrategyBank
    from src.pipeline.strategy_router import StrategyRouter
    from src.ml.ml_engine import MLEngine, TradeMemory
    from src.rl.rl_controller import RLController

    strategy_bank = StrategyBank()
    strategy_router = StrategyRouter()
    ml_engine = MLEngine(memory_path="logs/shadow/trade_memory_phase2.jsonl")
    rl_controller = RLController(window=20)
    router_skip_blocks = 0
    frequency_gap_blocks = 0
    prev_boll_z = {}   # per-symbol scalar cache for reversal confirmation
    reversal_blocks = 0
    risk_skips = 0
    risk_block_reasons = {}
    adaptive_relax_active = 0
    # Symbol health control (rolling PF on last 10-15 trades)
    sym_recent_pnls = {}          # sym -> deque[pnl_usd]
    # NOTE: no hard blocking/cooldowns in health filter (sizing only).
    sym_cooldown_until_bar = {}   # kept for compatibility (no longer used)
    sym_health_window = 30
    last_trade_bar = {sym: -999 for sym in valid_symbols}
    sym_bos_state = {sym: {"level": None, "dir": 0, "active": False} for sym in valid_symbols}

    # Phase 2 adaptive relaxation: approximate trades/day over last ~3 days (H1 → 72 bars).
    from collections import deque
    recent_trade_bars = deque(maxlen=72)

    t_start = time.time()

    # ── Silence noisy sub-loggers so tqdm bar renders cleanly ────────────
    _noisy = ["pipeline_v2", "src.decision", "src.risk",
              "src.features", "src.broker", "src.monitoring"]
    _saved_levels = {n: logging.getLogger(n).level for n in _noisy}
    if verbose:
        for n in _noisy:
            logging.getLogger(n).setLevel(logging.ERROR)

    pbar = _tqdm(
        sim_ts,
        total=len(sim_ts),
        desc="SCOPUS Multi-Symbol",
        unit="bar",
        dynamic_ncols=True,
        disable=not verbose,
    )

    for ts in pbar:
        bar_count += 1
        recent_trade_bars.append(0)

        # ── Phase A: Fill pending signals at THIS bar's open ─────────────
        for sym, psig in list(pending_signals.items()):
            idx = sym_ts_idx[sym].get(ts)
            if idx is None:
                continue
            psig["price"] = float(sym_dfs[sym].iloc[idx]["open"])
            psig.setdefault("metadata", {})
            if isinstance(psig.get("metadata"), dict):
                psig["metadata"]["bar_time"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            if not dry_run:
                try:
                    fill = executor.execute(psig)
                    if fill is not None:
                        trade_count += 1
                        last_trade_bar[sym] = bar_count
                        try:
                            recent_trade_bars[-1] = int(recent_trade_bars[-1]) + 1
                        except Exception:
                            pass
                        trades_per_sym[sym] = trades_per_sym.get(sym, 0) + 1
                        grp = get_group(sym) or "OTHER"
                        trades_per_group[grp] = trades_per_group.get(grp, 0) + 1
                        if verbose:
                            LOG.info(
                                f"  FILL [{bar_count:4d}] {sym} "
                                f"{psig['side'].upper()} "
                                f"@ {psig['price']:.5f}  "
                                f"score={psig.get('_score', 0):.3f}  "
                                f"regime={psig.get('regime', '?')} [next-open]"
                            )
                except Exception as e:
                    LOG.debug(f"[multi] fill error ({sym}): {e}")

        # ── Phase A2: Record closed-trade PnL for blacklist tracking ─────
        if hasattr(executor, 'state'):
            closed = executor.state.closed_trades
            cursor = getattr(executor, '_bl_cursor', 0)
            while cursor < len(closed):
                ct = closed[cursor]
                pnl = ct.pnl_usd if ct.pnl_usd is not None else 0.0
                strat = ct.metadata.get("strategy", "MEAN_REVERSION") if ct.metadata else "MEAN_REVERSION"
                
                blacklist.record_trade(ct.symbol, strat, pnl)

                # --- Symbol health update (rolling PF per symbol) ---
                sym_u = str(ct.symbol or "").upper()
                if sym_u:
                    dq = sym_recent_pnls.get(sym_u)
                    if dq is None:
                        dq = deque(maxlen=sym_health_window)
                        sym_recent_pnls[sym_u] = dq
                    dq.append(float(pnl))

                
                # Update Risk Engine
                risk_engine.update_post_trade(ct)

                # Phase 2: Record to RL and ML
                rl_controller.record_trade(pnl, strat)

                features = ct.metadata if ct.metadata else {}
                ml_engine.record_trade(TradeMemory(
                    symbol=ct.symbol,
                    strategy=strat,
                    side=ct.side,
                    adx=features.get("adx", 0.0),
                    atr_pctile=features.get("atr_pctile", 0.0),
                    boll_z=features.get("boll_z", 0.0),
                    regime_score=features.get("regime_score", 0.0),
                    ret_std=features.get("ret_std", 0.0),
                    hour=features.get("hour", 12),
                    bar_index=bar_count,
                    pnl_usd=pnl,
                    r_multiple=0.0,
                    won=pnl > 0,
                    ml_confidence=features.get("ml_confidence", 0.0),
                ))

                cursor += 1
            executor._bl_cursor = cursor

        # Keep RL drawdown updated
        if hasattr(executor, 'get_summary'):
            rl_controller.set_drawdown(executor.get_summary().get("drawdown_pct", 0.0))

        pending_signals.clear()

        # ── Phase B: SL/TP check for all open positions ──────────────────
        for sym in valid_symbols:
            idx = sym_ts_idx[sym].get(ts)
            if idx is None:
                continue
            row = sym_dfs[sym].iloc[idx]
            try:
                executor.check_sl_tp(sym, float(row["high"]), float(row["low"]), bar_time=ts)
            except Exception as e:
                LOG.debug(f"[multi] sl_tp error ({sym}): {e}")

        # ── Phase C: Collect signals from ALL symbols ────────────────────
        open_pos   = executor.get_open_positions()
        candidates = []

        for sym in valid_symbols:
            if sym in open_pos:
                continue   # already has open position

            # Symbol health filter is sizing-only (no pre-skip).

            idx = sym_ts_idx[sym].get(ts)
            if idx is None:
                continue

            # Build current bar object
            row = sym_dfs[sym].iloc[idx]
            bar = _Bar(
                row["open"], row["high"], row["low"], row["close"],
                row.get("volume", 1000.0),
                time=ts,
            )

            # Get rolling history window for features
            hist_start = max(0, idx - HISTORY_BARS + 1)
            raw_df     = sym_dfs[sym].iloc[hist_start: idx + 1].copy()

            if len(raw_df) < pcfg.min_bars:
                continue   # warm-up incomplete

            # ── PHASE 1: Structure Tracking (BOS & Liquidity Sweeps)
            if len(raw_df) < 22:
                continue

            current_high = float(getattr(bar, "high", getattr(bar, "High", getattr(bar, "h", 0.0))) or 0.0)
            current_low = float(getattr(bar, "low", getattr(bar, "Low", getattr(bar, "l", 0.0))) or 0.0)
            current_close = float(getattr(bar, "close", getattr(bar, "Close", getattr(bar, "c", 0.0))) or 0.0)
            current_open = float(getattr(bar, "open", getattr(bar, "Open", getattr(bar, "o", 0.0))) or 0.0)
            
            recent_highs = raw_df["high"].iloc[-21:-1]
            recent_lows = raw_df["low"].iloc[-21:-1]
            highest_20 = float(recent_highs.max())
            lowest_20 = float(recent_lows.min())
            
            # BOS Detection
            state = sym_bos_state[sym]
            
            # ATR for sizing/SL
            _atr_val = float((raw_df["high"] - raw_df["low"]).rolling(14).mean().iloc[-1])
            
            if current_high > highest_20 and current_close > highest_20:
                breakout_size = current_close - highest_20
                if breakout_size >= _atr_val * 0.3:
                    state["level"] = highest_20
                    state["dir"] = 1
                    state["active"] = True
            elif current_low < lowest_20 and current_close < lowest_20:
                breakout_size = lowest_20 - current_close
                if breakout_size >= _atr_val * 0.3:
                    state["level"] = lowest_20
                    state["dir"] = -1
                    state["active"] = True

            # Cooldown check
            if bar_count - last_trade_bar[sym] <= 10:
                continue

            # ── PHASE 2: Signal Generation
            mr_sig = None
            tp_sig = None

            # 2. TREND PULLBACK (BOS Pullback + Rejection)
            # H4 Trend Filter approximation (EMA80 on H1)
            ema_up = True
            ema_down = True
            if len(raw_df) > 80:
                ema80 = raw_df["close"].ewm(span=80, adjust=False).mean()
                ema_up = float(ema80.iloc[-1]) > float(ema80.iloc[-2])
                ema_down = float(ema80.iloc[-1]) < float(ema80.iloc[-2])
            
            if state["active"]:
                if state["dir"] == 1 and ema_up:
                    if current_low <= state["level"] and current_close > state["level"] and current_close > current_open:
                        sl = current_low - _atr_val * 0.5
                        sl_dist = current_close - sl
                        tp = current_close + 2.0 * sl_dist
                        tp_sig = {
                            "symbol": sym,
                            "strategy": "TREND_PULLBACK",
                            "side": "buy",
                            "price": current_close,
                            "sl": sl,
                            "tp": tp,
                            "size": pcfg.base_risk_pct,
                            "metadata": {"strategy": "TREND_PULLBACK"}
                        }
                        state["active"] = False
                elif state["dir"] == -1 and ema_down:
                    if current_high >= state["level"] and current_close < state["level"] and current_close < current_open:
                        sl = current_high + _atr_val * 0.5
                        sl_dist = sl - current_close
                        tp = current_close - 2.0 * sl_dist
                        tp_sig = {
                            "symbol": sym,
                            "strategy": "TREND_PULLBACK",
                            "side": "sell",
                            "price": current_close,
                            "sl": sl,
                            "tp": tp,
                            "size": pcfg.base_risk_pct,
                            "metadata": {"strategy": "TREND_PULLBACK"}
                        }
                        state["active"] = False

            # --- Execution Filter (Spread + Slippage cost vs expected move) ---
            pip_val = 0.01 if "JPY" in sym else 0.0001
            total_cost = (1.5 + 1.0) * 2 * pip_val
            
            if tp_sig:
                expected_move = abs(tp_sig["tp"] - tp_sig["price"])
                if expected_move < total_cost * 4:
                    tp_sig = None
            if mr_sig:
                expected_move = abs(mr_sig["tp"] - mr_sig["price"])
                if expected_move < total_cost * 4:
                    mr_sig = None

            # --- Select signal (MR priority) ---
            sig = None
            score = 1.0
            chosen_strategy = None
            
            if mr_sig:
                sig, score, chosen_strategy = mr_sig, 1.0, "MEAN_REVERSION"
            elif tp_sig:
                sig, score, chosen_strategy = tp_sig, 1.0, "TREND_PULLBACK"

            if sig:
                # ──────────────────────────────────────────────────────────
                # SINGLE ENTRY DECISION POINT
                # All entry logic consolidated here. Nothing after this
                # block may reject a valid entry (except Risk Engine).
                #
                # ──────────────────────────────────────────────────────────
                
                size_mult = 1.0
                original_size = sig.get("size", 0.01)
                sig["size"] = round(original_size * size_mult, 4)

                sig["_score"]        = score
                sig["_regime_score"] = 0.5
                sig["_size_mult"]    = round(size_mult, 3)
                sig["_size_tier"]    = "UNKNOWN"
                
                sig["metadata"] = {
                    "strategy": chosen_strategy or "UNKNOWN",
                    "adx": 0.0,
                    "atr": _atr_val,
                    "atr_pctile": 0.0,
                    "boll_z": 0.0,
                    "regime_score": 0.5,
                    "ret_std": 0.0,
                    "ml_confidence": 0.0,
                    "hour": 12
                }
                
                rl_controller.set_last_trade_bar(bar_count)
                
                candidates.append({
                    "symbol": sym,
                    "signal": sig,
                    "score":  score,
                    "boll_z": 0.0,
                    "regime_score": 0.5,
                })

            prev_boll_z[sym] = 0.0

        # ── Phase D: Portfolio selection (rank + guard + cap) ────────────
        if candidates:
            selected = select_signals(candidates, open_pos, max_trades=3)
            for item in selected:
                sym = item["symbol"]
                sig = item["signal"]
                
                # Evaluate Risk Engine BEFORE routing
                try:
                    allow, size_mult, reason = risk_engine.evaluate_pre_trade(sig, open_pos, ts)
                except Exception as e:
                    LOG.error(f"[fast_shadow] Risk engine error: {e}")
                    allow, size_mult, reason = True, 1.0, "error_fallback"
                
                if allow:
                    sig["size"] = sig.get("size", 1.0) * size_mult
                    pending_signals[sym] = sig
                else:
                    LOG.debug(f"[Risk Engine] Blocked {sym}: {reason}")
                    risk_skips += 1
                    risk_block_reasons[reason] = risk_block_reasons.get(reason, 0) + 1

        # Live postfix update (tqdm handles rate/ETA itself)
        if _TQDM_AVAILABLE and verbose and bar_count % 10 == 0:
            pbar.set_postfix(trades=trade_count, active=len(open_pos), refresh=False)

    elapsed = time.time() - t_start

    # Restore logger levels and close progress bar
    if verbose:
        for n, lvl in _saved_levels.items():
            logging.getLogger(n).setLevel(lvl)
    if _TQDM_AVAILABLE:
        pbar.close()

    # ── 6. Aggregate results ──────────────────────────────────────────────

    raw_summary = executor.get_summary()

    # ── Export to Brain journal (append-only JSONL) ──────────────────────
    try:
        import dataclasses as _dc
        from pathlib import Path as _Path
        brain_path = _Path("logs/brain/trade_journal.jsonl")
        brain_path.parent.mkdir(parents=True, exist_ok=True)
        with brain_path.open("a", encoding="utf-8") as _fh:
            for _t in executor.state.closed_trades:
                _row = _dc.asdict(_t)
                _row["event"] = "TRADE_CLOSED"
                _fh.write(json.dumps(_row, default=str) + "\n")
        LOG.info(f"[brain] Exported {len(executor.state.closed_trades)} trades to {brain_path}")
    except Exception as _e:
        LOG.debug(f"[brain] Journal export skipped: {_e}")

    group_pct = {
        g: round(cnt / max(trade_count, 1) * 100, 1)
        for g, cnt in trades_per_group.items()
    }
    sym_pct = {
        sym: round(cnt / max(trade_count, 1) * 100, 1)
        for sym, cnt in trades_per_sym.items()
    }
    # Validation check
    max_sym_pct = max(sym_pct.values()) if sym_pct else 0.0

    # Calculate strategy breakdown and per-strategy PF
    strat_counts = {}
    strat_pnls = {}    # strategy → [pnl, pnl, ...]
    strat_sym_pnls = {}  # (strategy, symbol) → [pnl, ...]
    if hasattr(executor, 'state'):
        for t in executor.state.closed_trades:
            s_name = t.metadata.get("strategy", "UNKNOWN") if t.metadata else "UNKNOWN"
            s_sym = str(t.symbol or "UNKNOWN").upper()
            strat_counts[s_name] = strat_counts.get(s_name, 0) + 1
            strat_pnls.setdefault(s_name, []).append(float(t.pnl_usd or 0))
            strat_sym_pnls.setdefault((s_name, s_sym), []).append(float(t.pnl_usd or 0))
        for t in executor.state.open_positions.values():
            s_name = t.metadata.get("strategy", "UNKNOWN") if t.metadata else "UNKNOWN"
            strat_counts[s_name] = strat_counts.get(s_name, 0) + 1

    # Compute per-strategy profit factor
    strat_pf = {}
    for s_name, pnls in strat_pnls.items():
        gross_profit = sum(p for p in pnls if p > 0) or 1e-9
        gross_loss = abs(sum(p for p in pnls if p < 0)) or 1e-9
        strat_pf[s_name] = round(gross_profit / gross_loss, 3)

    # Compute per-strategy-per-symbol PF
    strat_sym_pf = {}
    for (s_name, s_sym), pnls in strat_sym_pnls.items():
        gross_profit = sum(p for p in pnls if p > 0) or 1e-9
        gross_loss = abs(sum(p for p in pnls if p < 0)) or 1e-9
        strat_sym_pf[(s_name, s_sym)] = {
            "pf": round(gross_profit / gross_loss, 3),
            "trades": len(pnls),
            "wins": sum(1 for p in pnls if p > 0),
        }

    summary = {
        **raw_summary,
        "mode":              "multi_symbol",
        "symbols":           valid_symbols,
        "timeframe":         tf,
        "days_simulated":    days,
        "bars":              bar_count,
        "trades":            trade_count,
        "errors":            error_count,
        "elapsed_s":         round(elapsed, 2),
        "bars_per_sec":      round(bar_count / max(elapsed, 0.1), 1),
        "trades_per_symbol": trades_per_sym,
        "trades_per_group":  trades_per_group,
        "group_pct":         group_pct,
        "sym_pct":           sym_pct,
        "max_sym_pct":       max_sym_pct,
        "strat_counts":      strat_counts,
        "router_skips":      router_skip_blocks,
        "freq_blocks":       frequency_gap_blocks,
        "adaptive_relax_active_bars": adaptive_relax_active,
    }

    # ── 7. Print report ───────────────────────────────────────────────────
    sep  = "=" * 65
    dash = "-" * 65
    pf   = raw_summary.get("profit_factor", 0)
    wr   = raw_summary.get("win_rate", 0)
    dd   = raw_summary.get("drawdown_pct", 0)
    pnl  = raw_summary.get("net_pnl_usd", 0)
    f_spread = raw_summary.get("friction_spread_usd", 0)
    f_slip   = raw_summary.get("friction_slip_usd", 0)
    f_pct    = raw_summary.get("friction_loss_pct", 0)

    LOG.info(sep)
    LOG.info("[multi] MULTI-SYMBOL SIMULATION COMPLETE (Phase 2 JARVIS)")
    LOG.info(f"  Symbols:    {', '.join(valid_symbols)}")
    LOG.info(f"  Bars:       {bar_count:,}  ({days} days @ {tf})")
    LOG.info(f"  Trades:     {trade_count}")
    LOG.info(f"  Win rate:   {wr:.1%}")
    LOG.info(f"  PF:         {pf:.3f}")
    LOG.info(f"  Net PnL:    ${pnl:+,.2f}")
    LOG.info(f"  Drawdown:   {dd:.2f}%")
    LOG.info(f"  Friction:   Spread ${f_spread:,.2f} | Slippage ${f_slip:,.2f}")
    LOG.info(f"  Friction %: {f_pct:.1f}% (PnL lost to execution cost)")
    LOG.info(f"  Speed:      {summary['bars_per_sec']} bars/sec")
    LOG.info(f"  Elapsed:    {elapsed:.1f}s")
    LOG.info(dash)
    LOG.info("  Phase 2 Subsystems:")
    LOG.info(f"    Router skips (chop logic): {router_skip_blocks}")
    LOG.info(f"    RL Freq gap blocks:        {frequency_gap_blocks}")
    LOG.info(f"    Strategy Breakdown:        {strat_counts}")
    if strat_pf:
        LOG.info("    Per-Strategy PF:")
        for s_name in sorted(strat_pf.keys()):
            s_trades = len(strat_pnls.get(s_name, []))
            s_wins = sum(1 for p in strat_pnls.get(s_name, []) if p > 0)
            s_wr = s_wins / max(s_trades, 1)
            s_pnl = sum(strat_pnls.get(s_name, []))
            s_avg = s_pnl / max(s_trades, 1)
            LOG.info(f"      {s_name:20s}: PF={strat_pf[s_name]:.3f}  trades={s_trades}  WR={s_wr:.1%}  Avg PnL=${s_avg:+.2f}")
        LOG.info("    Per-Strategy-Symbol PF:")
        for (s_name, s_sym) in sorted(strat_sym_pf.keys()):
            info = strat_sym_pf[(s_name, s_sym)]
            s_wr = info["wins"] / max(info["trades"], 1)
            LOG.info(f"      {s_sym:8s} {s_name:20s}: PF={info['pf']:.3f}  trades={info['trades']}  WR={s_wr:.1%}")
    LOG.info(dash)
    LOG.info("  Dual-speed filters:")
    LOG.info(f"    Fast blocks (3-consec):    {fast_blocks}")
    LOG.info(f"    Slow blocks (PF<0.80):     {blacklist_blocks}")
    LOG.info(f"    Slow PF size-penalty bars:  {slow_pf_blocks}")
    LOG.info(f"    Persistence blocks:        {persist_blocks}")
    LOG.info(f"    Reversal blocks:           {reversal_blocks}")
    LOG.info(f"    Risk engine blocks:        {risk_skips}")
    if risk_skips > 0:
        for reason, count in risk_block_reasons.items():
            LOG.info(f"      - {reason}: {count}")
    bl_status = blacklist.status()
    if bl_status:
        LOG.info("    Symbol health:")
        for sym, st in bl_status.items():
            LOG.info(f"      {sym:8s}: {st}")
    LOG.info(dash)
    LOG.info("  Trades per symbol:")
    for sym, cnt in sorted(trades_per_sym.items(), key=lambda x: -x[1]):
        p = sym_pct.get(sym, 0)
        flag = "  ← >40% concentration" if p > 40 else ""
        LOG.info(f"    {sym:8s}: {cnt:3d}  ({p:.1f}%){flag}")
    LOG.info("  Trades per group:")
    for grp, cnt in trades_per_group.items():
        LOG.info(f"    {grp:8s}: {cnt:3d}  ({group_pct.get(grp, 0):.1f}%)")
    LOG.info(dash)
    # Validation gate
    pass_fail = (
        trade_count >= 80
        and pf >= 1.30
        and dd <= 10.0
        and max_sym_pct <= 40.0
    )
    LOG.info(f"  VALIDATION: {'PASS ✓' if pass_fail else 'FAIL ✗'}")
    LOG.info(f"    trades≥80: {'✓' if trade_count >= 80 else '✗'} ({trade_count})")
    LOG.info(f"    PF≥1.30:   {'✓' if pf >= 1.30 else '✗'} ({pf:.3f})")
    LOG.info(f"    DD≤10%:    {'✓' if dd <= 10.0 else '✗'} ({dd:.2f}%)")
    LOG.info(f"    sym≤40%:   {'✓' if max_sym_pct <= 40.0 else '✗'} ({max_sym_pct:.1f}%)")
    LOG.info(sep)

    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="SCOPUS Fast Shadow Simulator")
    p.add_argument("--symbol",  default="EURUSD",
                   help="Single symbol for single-symbol mode (ignored if --symbols set).")
    p.add_argument("--symbols", default=None,
                   help="Comma-separated symbols or ALL. Enables multi-symbol mode. "
                        "Example: --symbols ALL   or   --symbols EURUSD,GBPUSD")
    p.add_argument("--tf",        default="H1", choices=list(BARS_PER_DAY))
    p.add_argument("--days",      default=30, type=int)
    p.add_argument("--data-file", default=None, dest="data_file")
    p.add_argument("--cfg",       default="config/weapon_system.yaml")
    p.add_argument("--dry-run",   action="store_true", dest="dry_run")
    p.add_argument("--json",      action="store_true",
                   help="Emit JSON summary at end")
    args = p.parse_args(argv)

    cfg_hash = file_sha256(args.cfg)
    if cfg_hash:
        LOG.info(f"[fast_shadow] Config fingerprint sha256={cfg_hash[:12]}.. ({args.cfg})")

    # ── Multi-symbol mode ────────────────────────────────────────────────
    if args.symbols is not None:
        syms = (
            ALL_SYMBOLS
            if args.symbols.strip().upper() == "ALL"
            else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        )
        summary = run_multi_symbol_shadow(
            symbols  = syms,
            tf       = args.tf,
            days     = args.days,
            cfg_path = args.cfg,
            dry_run  = args.dry_run,
            verbose  = not args.json,
        )

    # ── Single-symbol mode (UNCHANGED) ──────────────────────────────────
    else:
        data_path = find_data_file(args.symbol, args.tf, args.data_file)
        summary   = run_fast_shadow(
            data_path = data_path,
            symbol    = args.symbol,
            tf        = args.tf,
            days      = args.days,
            cfg_path  = args.cfg,
            dry_run   = args.dry_run,
            verbose   = not args.json,
        )

    if args.json:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
