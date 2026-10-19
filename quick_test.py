"""
quick_test.py
-------------
One command that checks the whole kit works (about 2-3 minutes on a laptop):

    python quick_test.py

It (1) checks the model's math against the paper, (2) generates two small fresh
dummy markets, one with planted mispricing and one without, (3) runs the full
strategy on both, and (4) runs the real-data path on Bloomberg-style files.
Every check prints PASS or FAIL; a summary comes at the end.
Scratch files go to outputs/quick_test/ (safe to delete).
"""
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import lmz_model as M
import make_dummy_data as D
import lmz_pipeline as P
import run_test as R

OUT = os.path.join("outputs", "quick_test")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def to_bloomberg_style(src, dst):
    """Rewrite WRDS-style dummy files into the simpler columns a Bloomberg pull produces."""
    os.makedirs(dst, exist_ok=True)
    t = R.load_tables(src)
    c = t["crsp_monthly"]
    c["me"] = c["prc"].abs() * c["shrout"] / 1e3
    c[["gvkey", "date", "me", "ret"]].assign(gvkey=lambda d: "TCK" + d.gvkey + " US Equity") \
        .to_csv(os.path.join(dst, "crsp_monthly.csv"), index=False)
    a = t["compustat_annual"]
    a["ebitda"] = a["sale"] - a["cogs"] - a["xsga"]
    a["debt"] = a["dlc"] + a["dltt"]
    a[["gvkey", "datadate", "gind", "gsector", "ppegt", "ebitda", "dp", "debt"]] \
        .assign(gvkey=lambda d: "TCK" + d.gvkey + " US Equity").to_csv(os.path.join(dst, "compustat_annual.csv"), index=False)
    q = t["compustat_quarterly"]
    q[["gvkey", "datadate", "saleq"]].assign(gvkey=lambda d: "TCK" + d.gvkey + " US Equity") \
        .to_csv(os.path.join(dst, "compustat_quarterly.csv"), index=False)
    i = t["ibes_ltg"]
    i["meanltg"] = i["meanltg"] + 9.0            # make forecasts as optimistic as real ones (~10-15%)
    i[["gvkey", "statpers", "meanltg"]].assign(gvkey=lambda d: "TCK" + d.gvkey + " US Equity") \
        .to_csv(os.path.join(dst, "ibes_ltg.csv"), index=False)
    t["ff_factors"].to_csv(os.path.join(dst, "ff_factors.csv"), index=False)
    pd.DataFrame({"date": t["ff_factors"]["date"], "r": 0.045}).to_csv(os.path.join(dst, "risk_free.csv"), index=False)


def main():
    t0 = time.time()
    shutil.rmtree(OUT, ignore_errors=True)

    print("\n1. Model math (paper equations 3-5)")
    th, eta, mu, sg, lam, r = 0.25, 1.0, 0.01, 0.2, 0.1, 0.05
    K = 10.0
    X = M.investment_boundary(K, th, eta, mu, sg, lam, r)
    V = lambda k, x: M.firm_value(k, x, th, eta, mu, sg, lam, r, adjust=False)
    h = 1e-4
    VK = (V(K * (1 + h), X) - V(K * (1 - h), X)) / (2 * K * h)
    check("marginal value of capital = eta at the investment boundary", abs(VK - eta) < 1e-5, f"{VK:.6f}")
    x = 0.7 * X
    Vx = (V(K, x * (1 + h)) - V(K, x * (1 - h))) / (2 * x * h)
    Vxx = (V(K, x * (1 + h)) - 2 * V(K, x) + V(K, x * (1 - h))) / (x * h) ** 2
    VKi = (V(K * (1 + h), x) - V(K * (1 - h), x)) / (2 * K * h)
    pde = r * V(K, x) - ((1 - M.TAU) * x * K**th + mu * x * Vx + 0.5 * sg**2 * x**2 * Vxx - lam * K * VKi)
    check("valuation formula solves the model's pricing equation", abs(pde) / V(K, x) < 1e-6)
    tb = float(M.theta_upper_bound(0.01, 0.2, 0.05, 0.05))
    check("theta upper bound matches the paper's 0.46", abs(tb - 0.46) < 0.005, f"{tb:.3f}")
    print("\n2. Dummy data (two small markets, 1995-2014)")
    small = dict(n_industries=6, firms_per_industry=(40, 60), start="1995-01", end="2014-12")
    dm, dn = os.path.join(OUT, "data_mispriced"), os.path.join(OUT, "data_null")
    D.main(dm, mispricing="go", seed=11, **small)
    D.main(dn, mispricing="none", seed=11, **small)
    need = R.FILES[:-1] + R.TRUTH
    check("all data files written", all(os.path.exists(os.path.join(dm, f + ".csv")) for f in need))
    tf = pd.read_csv(os.path.join(dm, "truth_firm_month.csv"))
    check("mispricing was planted in the mispriced market", tf["g_GO_mispricing"].std() > 0.1,
          f"sd {tf['g_GO_mispricing'].std():.2f}")
    print("\n3. Full strategy on the mispriced market (true inputs) and the control market")
    print("   (this is the slow part, about a minute each)", flush=True)
    m = R.run(dm, os.path.join(OUT, "run_mispriced"), true_inputs=True, verbose=False)
    n = R.run(dn, os.path.join(OUT, "run_null"), verbose=False)
    sq = m["signal_quality"]["spearman_pooled_avg"]
    check("signal ranks firms in line with the planted mispricing", sq > 0.15, f"rank corr {sq:.2f}")
    fm_m = m["fama_macbeth"][1]["t"]
    check("cheap-by-model stocks earn more (cross-sectional t below -2)", fm_m < -2, f"t = {fm_m:.1f}")
    ls_m = m["vw_alpha_pct"]["Mean"]["1-10"]
    check("long-short return is positive in the mispriced market", ls_m > 0, f"{ls_m:.2f}% a month")
    fm_n = n["fama_macbeth"][1]["t"]
    ls_n_t = n["vw_t"]["Mean"]["1-10"]
    check("control market: no significant return (no look-ahead or bugs)", abs(fm_n) < 2.5 and abs(ls_n_t) < 2.5,
          f"cross-sectional t = {fm_n:.1f}, long-short t = {ls_n_t:.1f}")

    print("\n4. Real-data path on Bloomberg-style files (paper's appendix settings)")
    db = os.path.join(OUT, "data_bloomberg_style")
    to_bloomberg_style(dm, db)
    tb_ = R.load_tables(db)
    cfg = P.REAL_DATA_CONFIG
    panel = P.build_panel(tb_, cfg)
    rf = tb_["risk_free"].assign(date=lambda d: pd.to_datetime(d["date"]) + pd.offsets.MonthEnd(0)).set_index("date")["r"]
    cal = P.calibrate(panel, tb_, cfg, r_series=rf)
    check("Bloomberg-style files load into the firm-month panel", panel["usable"].sum() > 1000, f"{int(panel['usable'].sum())} firm-months")
    g = cal["ltg_perpetual"]
    check("analyst growth adjusted the appendix's way (raw ~10% -> ~2-4%)",
          g.notna().all() and 0.015 < g.median() < 0.045, f"median raw {cal['ltg'].median():.1%} -> {g.median():.1%}")
    p = P.merge_params(panel, cal)
    one = p[p["est_ok"]]["gind"].iloc[0]
    est = P.estimate_theta_eta(p[p["gind"] == one], cfg)
    mis = P.misvaluation(p, est, cfg)
    check("estimation and misvaluation run on the real-data settings", len(mis) > 100 and np.isfinite(mis["log_misval"]).all(),
          f"{len(mis)} firm-months in one industry")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed in {time.time() - t0:.0f} seconds.")
    print("All good." if passed == len(results) else "Something is off: see the FAIL lines above.")


if __name__ == "__main__":
    main()
