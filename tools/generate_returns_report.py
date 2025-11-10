# tools/generate_returns_report.py (drop this into your repo)
from pathlib import Path
import json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def load_or_empty(path, cols):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=cols)

def make_report(out_path="reports/returns_report.pdf"):
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    summary_p = reports / "returns_summary.json"
    daily_p   = reports / "returns_daily.csv"
    monthly_p = reports / "returns_monthly.csv"
    equity_p  = reports / "equity.csv"

    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    daily   = load_or_empty(daily_p, ["timestamp","ret"])
    monthly = load_or_empty(monthly_p, ["timestamp","ret"])
    equity  = load_or_empty(equity_p, ["timestamp","equity"])

    for df, col in [(daily,"timestamp"), (monthly,"timestamp"), (equity,"timestamp")]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col])

    def pct(x): return f"{x*100:.2f}%"

    def page_title(title, lines):
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.5, 0.92, title, ha="center", va="center", fontsize=20, fontweight="bold")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        plt.text(0.5, 0.88, f"Generated: {now}", ha="center", va="center", fontsize=10)
        y = 0.80
        for line in lines:
            plt.text(0.1, y, line, ha="left", va="top", fontsize=12)
            y -= 0.05
        return fig

    def chart_equity(df):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        if not df.empty:
            ax.plot(df["timestamp"], df["equity"])
            ax.set_title("Equity Curve"); ax.set_xlabel("Date"); ax.set_ylabel("Equity")
        fig.tight_layout(); return fig

    def chart_daily(df):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        if not df.empty:
            ax.bar(df["timestamp"], df["ret"])
            ax.set_title("Daily Returns"); ax.set_xlabel("Date"); ax.set_ylabel("Return")
        fig.tight_layout(); return fig

    def chart_monthly(df):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        if not df.empty:
            df2 = df.copy(); df2["Month"] = pd.to_datetime(df2["timestamp"]).dt.to_period("M").astype(str)
            ax.bar(df2["Month"], df2["ret"])
            ax.set_title("Monthly Returns"); ax.set_xlabel("Month"); ax.set_ylabel("Return")
            for lab in ax.get_xticklabels(): lab.set_rotation(45)
        fig.tight_layout(); return fig

    lines = [
        f"Days: {summary.get('days', 0)}",
        f"Cumulative Return: {pct(summary.get('cum_return', 0.0))}",
        f"Annualized Sharpe: {summary.get('ann_sharpe', 0.0):.2f}",
        f"Hit Rate: {pct(summary.get('hit_rate', 0.0))}",
        f"Max Drawdown: {pct(summary.get('max_drawdown', 0.0))}",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with PdfPages(out_path) as pdf:
        pdf.savefig(page_title("Adaptive Agent — Returns Report", lines)); plt.close()
        pdf.savefig(chart_equity(equity)); plt.close()
        pdf.savefig(chart_daily(daily)); plt.close()
        pdf.savefig(chart_monthly(monthly)); plt.close()

if __name__ == "__main__":
    make_report()
