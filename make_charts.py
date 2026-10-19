"""Static PNG charts for slides (outputs/charts/). Run after run_test.py runs exist."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY, DARK = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})


def load_vw(run):
    f = os.path.join("outputs", run, "returns_deciles_vw.csv")
    return pd.read_csv(f, index_col=0, parse_dates=True) if os.path.exists(f) else None


def decile_chart(out):
    paper = pd.read_csv("paper_benchmarks.csv")
    paper = paper[(paper.table == "VI_alpha_pct") & (paper.portfolio.isin([str(i) for i in range(1, 11)]))]
    runs = sorted(glob.glob("outputs/base_seed*_trueinputs/returns_deciles_vw.csv"))
    runs = [r for r in runs if os.path.basename(os.path.dirname(r)).count("_") == 2]  # skip perturbation runs
    rf = 0.05 / 12
    means = np.mean([pd.read_csv(r, index_col=0)[[str(i) for i in range(1, 11)]].mean().values - rf for r in runs], axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(1, 11)
    ax.bar(x, means, color=GREY, width=0.6, label=f"Dummy data, true inputs (avg of {len(runs)} simulated markets)")
    ax.plot(x, paper["Mean"].values, "o", color=BLUE, ms=7, label="Paper, real US data 1980-2014 (Table VI)")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x, ["1\nmost\nundervalued"] + [str(i) for i in range(2, 10)] + ["10\nmost\novervalued"])
    ax.set_ylabel("Avg monthly excess return (%)")
    ax.set_title("Returns fall from undervalued to overvalued deciles", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "decile_returns.png"), dpi=200); plt.close(fig)


def cumulative_chart(out):
    groups = [("base_seed{}_trueinputs", "Mispriced market, true inputs", BLUE),
              ("base_seed{}", "Mispriced market, paper-style calibrated inputs", ORANGE),
              ("null_seed{}", "No mispricing (control)", GREY)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for pat, label, c in groups:
        ws = [load_vw(pat.format(s)) for s in (7, 8, 9, 10)]
        ws = [w["1-10"] for w in ws if w is not None]
        if not ws:
            continue
        avg = pd.concat(ws, axis=1).mean(axis=1)          # equal mix of the simulated markets each month
        cum = np.exp(np.log1p(avg.fillna(0)).cumsum())
        ax.plot(cum.index, cum.values, color=c, lw=2, label=f"{label} ({len(ws)} markets)")
    ax.axhline(1, color="#c3c2b7", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Long D1 / short D10, value-weighted, averaged across simulated markets", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "cumulative_long_short.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    out = os.path.join("outputs", "charts")
    os.makedirs(out, exist_ok=True)
    decile_chart(out)
    cumulative_chart(out)
    print("charts ->", out)
