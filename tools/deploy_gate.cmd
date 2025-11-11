python tools\auto_find_latest.py
if ERRORLEVEL 1 exit /b 1
python tools\promote_model.py --require_predeploy_pass
python tools\aggregate_returns.py --equity_csv reports\equity.csv
