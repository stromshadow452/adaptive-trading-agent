#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.style.use("seaborn-v0_8-whitegrid")

EXEC = "reports/executions/executions_mark2close.csv"
OUT  = "reports/aggregate/equity_by_symbol_stats.png"

def main():
    df = pd.read_csv(EXEC)
    df = df[df["executed"].eq(True) & df["side"].isin(["buy", "sell"])].copy()
    if df.empty:
        raise SystemExit("No executed buy/sell rows found.")
    df["pnl"] = df["pnl"].astype(float)
    symbols = sorted(df["symbol"].unique(), key=lambda s: df.loc[df["symbol"]==s, "pnl"].sum(), reverse=True)

    n = len(symbols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 3), constrained_layout=True)
    if n == 1: axes = [axes]

    for ax, sym in zip(axes, symbols):
        sub = df[df["symbol"] == sym].copy()
        sub["cum_pnl"] = sub["pnl"].cumsum()

        ax.plot(sub.index, sub["cum_pnl"], label=sym, linewidth=2)
        ax.scatter(sub.index[sub["pnl"] > 0], sub["cum_pnl"][sub["pnl"] > 0], color="green", s=80, label="Win")
        ax.scatter(sub.index[sub["pnl"] <= 0], sub["cum_pnl"][sub["pnl"] <= 0], color="red", s=80, label="Loss")

        ax.set_title(sym, fontsize=13)
        ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL")
        wins = (sub["pnl"]>0).sum(); losses = (sub["pnl"]<=0).sum()
        winrate = 100.0 * wins / max(wins+losses, 1)
        gp = sub.loc[sub["pnl"]>0, "pnl"].sum(); gl = -sub.loc[sub["pnl"]<0, "pnl"].sum()
        pf = gp/gl if gl>0 else math.inf
        color = "green" if winrate >= 50 else "red"
        text = (f"Trades: {wins+losses}\n"
                f"Wins: {wins}  Losses: {losses}\n"
                f"Winrate: {winrate:.1f}%\n"
                f"PF: {pf:.2f}" if math.isfinite(pf) else
                f"Trades: {wins+losses}\nWins: {wins}  Losses: {losses}\nWinrate: {winrate:.1f}%\nPF: inf")
        ax.text(0.02, 0.05, text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.3"),
                color=color, ha="left", va="bottom")
        total_pnl = sub["pnl"].sum()
        ax.text(0.95, 0.9, f"ΣPnL: {total_pnl:+.3f}", transform=ax.transAxes,
                fontsize=9, ha="right", color=("green" if total_pnl>0 else "red"))
        ax.legend()

    plt.suptitle("Equity by Symbol with Stats", fontsize=15)
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"[OK] Saved detailed symbol chart -> {OUT}")

if __name__ == "__main__":
    main()
