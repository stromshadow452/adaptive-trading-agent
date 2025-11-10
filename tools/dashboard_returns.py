#!/usr/bin/env python3
import json, pandas as pd, streamlit as st, plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Adaptive Agent — Returns", layout="wide")

reports = Path("reports")
daily_path   = reports / "returns_daily.csv"
monthly_path = reports / "returns_monthly.csv"
summary_path = reports / "returns_summary.json"
equity_path  = reports / "equity.csv"   # optional, if you keep it here

# Load
daily   = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame(columns=["timestamp","ret"])
monthly = pd.read_csv(monthly_path) if monthly_path.exists() else pd.DataFrame(columns=["timestamp","ret","month"])
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

# Header
st.title("📈 Adaptive Trading Agent — PnL & Returns Dashboard")

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Days", summary.get("days", len(daily)))
c2.metric("Cum Return", f"{summary.get('cum_return', 0)*100:0.2f}%")
c3.metric("Ann. Sharpe", f"{summary.get('ann_sharpe', 0):0.2f}")
c4.metric("Max Drawdown", f"{summary.get('max_drawdown', 0)*100:0.2f}%")

# Equity curve (if available)
if equity_path.exists():
    eq = pd.read_csv(equity_path)
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    fig_eq = px.line(eq, x="timestamp", y="equity", title="Equity Curve")
    st.plotly_chart(fig_eq, use_container_width=True)

# Daily returns
if not daily.empty:
    daily["timestamp"] = pd.to_datetime(daily["timestamp"])
    fig_d = px.bar(daily, x="timestamp", y="ret", title="Daily Returns", labels={"ret":"Return"})
    st.plotly_chart(fig_d, use_container_width=True)

# Monthly returns heatmap-style by year-month
if not monthly.empty:
    monthly["timestamp"] = pd.to_datetime(monthly["timestamp"])
    monthly["Year"] = monthly["timestamp"].dt.year
    monthly["Month"] = monthly["timestamp"].dt.strftime("%b")
    pivot = monthly.pivot_table(index="Year", columns="Month", values="ret", aggfunc="sum")
    st.subheader("Monthly Returns (Table)")
    st.dataframe((pivot*100).round(2))

st.caption("Data sources: returns_daily.csv, returns_monthly.csv, returns_summary.json, equity.csv (optional).")
