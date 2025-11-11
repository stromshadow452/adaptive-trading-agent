#!/usr/bin/env bash
# run daily smoke for EURUSD auto tf
python -u -m src.main --mode smoke --symbol EURUSD --tf auto
