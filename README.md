# Adaptive Trading Agent

**Private Repository** | **v1.0**

---

## Overview

Single-asset (EURUSD M15) trading agent with a 13-stage pipeline.
- **Philosophy:** Risk-first, explainable, production-grade
- **Current Performance:** 30% return, 3.3% max DD, 57% win rate

---

## Quick Start

### 1. Clone the Repository
```bash
git clone git@github.com:YOUR_USERNAME/adaptive-trading-agent.git
cd adaptive-trading-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Data Locally
> ⚠️ **Data is intentionally excluded from Git**

Create the data directory and add your files:
```bash
mkdir -p data/raw
# Copy your EURUSD M15 CSV files to data/raw/
```

Expected structure:
```
data/
├── raw/
│   ├── EURUSD_M15.csv
│   └── forex_kaggle_multiTF/
├── training/
└── history/
```

### 5. Copy Config Template
```bash
cp config/config.example.yaml config/local.yaml
# Edit config/local.yaml with your settings
```

### 6. Run Backtest
```bash
python tools/phase4_quality_scorer_test.py
```

---

## Project Structure

```
adaptive-trading-agent/
├── src/                    # Core source code
│   ├── backtest/           # Backtesting engine
│   ├── market_data/        # Data loaders
│   ├── ml/                 # ML components
│   ├── risk/               # Risk management
│   └── stages/             # Pipeline stages
├── tools/                  # CLI tools and scripts
├── tests/                  # Unit and integration tests
├── config/                 # Configuration files
│   ├── config.example.yaml # Template (committed)
│   └── local.yaml          # Local settings (ignored)
├── models/                 # ML models (ignored by default)
├── data/                   # Market data (NEVER COMMITTED)
├── logs/                   # Runtime logs (ignored)
└── docs/                   # Documentation
```

---

## Configuration

### Environment Variables
Create a `.env` file (not committed):
```bash
# API Keys (if using live data)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Data paths
DATA_DIR=./data/raw
```

### Config Files
- `config.example.yaml` - Template (committed)
- `config/local.yaml` - Your local settings (ignored)
- `config/production.yaml` - Production settings (ignored)

---

## Data Requirements

This agent expects OHLCV data in CSV format:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.1050,1.1055,1.1045,1.1052,1000
```

### Supported Pairs (with data available locally)
- EURUSD ✅ (primary)
- GBPUSD, USDJPY, USDCAD, AUDUSD, etc.

---

## Key Commands

| Command | Description |
|---------|-------------|
| `python tools/phase4_quality_scorer_test.py` | Run backtest with Quality Scorer |
| `python tools/phase3_longrun_test.py` | Run baseline backtest |
| `pytest tests/` | Run all tests |
| `python -m src.backtest.execution_core` | Direct execution core test |

---

## Security Notes

- ❌ **Never commit data/** - Contains market data
- ❌ **Never commit .env** - Contains API keys
- ❌ **Never commit models/** - Large binary files
- ✅ **Commit only code** - All trading logic is versioned

---

## Architecture

```
[Data] → [Features] → [Regime] → [ML Brain] → [Quality Scorer]
    → [AdaptiveState] → [RSI Gate] → [Risk Brain] → [Execute]
```

See `docs/pipeline.md` for detailed architecture.

---

## Changelog

### v1.0 (2026-01-24)
- Quality Scorer integrated
- Regime-specific aggression
- 30% return, 3.3% max DD

---

*Private repository - Do not share*