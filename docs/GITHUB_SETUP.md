# GitHub Repository Setup Guide

## Quick Checklist

### Step 1: Clean up before first commit
```bash
# Navigate to project
cd "e:\adaptive-trading-agent (2)\adaptive-trading-agent (2)"

# Remove any existing git history (fresh start)
rm -rf .git

# Initialize new repository
git init

# Add all files (respects .gitignore)
git add .

# Verify what will be committed
git status
```

### Step 2: Verify NO data is staged
```bash
# This should return NOTHING:
git status | findstr ".csv"
git status | findstr "data/"
git status | findstr ".joblib"
```

### Step 3: Create GitHub private repository
1. Go to github.com → New Repository
2. Name: `adaptive-trading-agent`
3. **Select: Private** ✅
4. DO NOT add README (you have one)
5. Create repository

### Step 4: Push code
```bash
git commit -m "Initial commit: Trading agent v1.0"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/adaptive-trading-agent.git
git push -u origin main
```

---

## What Gets Committed vs Ignored

| Committed ✅ | Ignored ❌ |
|-------------|-----------|
| `src/` (all code) | `data/` (all market data) |
| `tools/` (scripts) | `*.csv` (any CSV) |
| `tests/` (tests) | `models/` (ML models) |
| `config/config.example.yaml` | `config/local.yaml` |
| `.gitignore` | `.env` |
| `README.md` | `logs/` |
| `requirements.txt` | `__pycache__/` |

---

## Security Best Practices

### 1. Pre-commit check (manual)
Before any commit, run:
```bash
git status | findstr ".csv .joblib .env"
# Should return nothing
```

### 2. Add pre-commit hook (optional)
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Block commits containing sensitive files
if git diff --cached --name-only | grep -E '\.(csv|joblib|pkl|env)$'; then
    echo "ERROR: Attempting to commit data/model/env files!"
    exit 1
fi
```

### 3. Folder naming convention
Use names that make it obvious what's ignored:
- `data/` → Obviously data
- `models/` → Obviously ML models
- `logs/` → Obviously logs

---

## Data Access Strategy

### Local-only data
```
data/
├── raw/                    # Your CSV files (ignored)
│   ├── EURUSD_M15.csv
│   └── forex_kaggle_multiTF/
├── training/               # Training datasets (ignored)
└── history/                # Backtest results (ignored)
```

### Config points to local paths
```yaml
# config/local.yaml
data:
  raw_dir: "./data/raw"
```

### Optional: API-based data
```python
# Future: Add API loaders
from src.market_data.api_loader import load_from_broker
df = load_from_broker("EURUSD", "M15")
```

---

## Summary

1. ✅ `.gitignore` created - blocks all data
2. ✅ `README.md` created - setup instructions
3. ✅ `config.example.yaml` created - template (no secrets)
4. ⚠️ Verify no data in staging before commit
5. 🔒 Create PRIVATE repository on GitHub
6. 🚀 Push code only
