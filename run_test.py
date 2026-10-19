"""
run_test.py
-----------
End-to-end test of the LMZ (2020) strategy on synthetic data.

  python run_test.py                         # base world (growth options mispriced)
  python run_test.py --scenario null         # no mispricing anywhere (should find nothing)
  python run_test.py --scenario total        # whole firm mispriced, not just GO
  python run_test.py --perturb mu=+0.02      # deliberately mis-calibrate an input
  python run_test.py --data data/bloomberg   # real data with the same file names (no truth checks);
                                             # uses lmz_pipeline.REAL_DATA_CONFIG (Internet Appendix rules)

Results go to outputs/<run name>/.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import lmz_pipeline as P
import lmz_backtest as B

FILES = ["crsp_monthly", "compustat_annual", "compustat_quarterly", "ibes_ltg", "ff_factors", "risk_free"]
TRUTH = ["truth_industry", "truth_firm_month"]


def load_tables(datadir):
    t = {}
    for k in FILES + TRUTH:
        f = os.path.join(datadir, k + ".csv")
        if os.path.exists(f):
            t[k] = pd.read_csv(f)
            if "gvkey" in t[k]:
                t[k]["gvkey"] = t[k]["gvkey"].astype(str).str.zfill(6)
    return t


def apply_perturbation(cal, perturb):
    """perturb like {'mu': '+0.02', 'sigma': '*1.3'}: simulate calibration errors."""
    cal = cal.copy()
    for k, v in (perturb or {}).items():
        op, val = v[0], float(v[1:])
        cal[k] = cal[k] + val if op == "+" else (cal[k] - val if op == "-" else cal[k] * val)
    cal["mu"] = np.minimum(cal["mu"], cal["r"] - P.CONFIG["mu_max_spread"])
    return cal


def signal_quality(mis, truth_fm, truth_ind, est):
    tf = truth_fm.copy()
    tf["date"] = pd.to_datetime(tf["date"]) + pd.offsets.MonthEnd(0)
    m = mis.merge(tf[["gvkey", "date", "log_true_misval", "g_GO_mispricing", "AP_true", "GO_true"]],
                  on=["gvkey", "date"], how="inner")
    sp = pd.Series([g["log_misval"].corr(g["log_true_misval"], method="spearman") for _, g in m.groupby("date")])
    within = pd.Series([g["log_misval"].corr(g["log_true_misval"], method="spearman") if len(g) > 10 else np.nan
                        for _, g in m.groupby(["date", "gind"])])
    c = np.corrcoef(m["log_misval"], m["log_true_misval"])[0, 1]
    # how well extreme deciles capture true mispricing
    m["dec"] = B.assign_groups(m, "log_misval", 10)
    m["true_dec"] = B.assign_groups(m, "log_true_misval", 10)
    hit1 = (m.loc[m.dec == 1, "true_dec"] <= 3).mean()
    hit10 = (m.loc[m.dec == 10, "true_dec"] >= 8).mean()
    ti = truth_ind.set_index("gind")
    e = est.groupby("gind")[["theta_hat", "eta_hat"]].median().join(ti[["theta", "eta"]])
    m["goap_true"] = m["GO_true"] / m["AP_true"]
    return dict(
        spearman_pooled_avg=float(sp.mean()),
        spearman_within_industry_avg=float(within.mean()),
        pearson_R2=float(c ** 2),
        share_D1_truly_undervalued_bottom3=float(hit1),
        share_D10_truly_overvalued_top3=float(hit10),
        theta_corr_with_truth=float(e["theta_hat"].corr(e["theta"])),
        eta_corr_with_truth=float(e["eta_hat"].corr(e["eta"])),
        goap_rank_corr_model_vs_true=float(m["go_ap"].corr(m["goap_true"], method="spearman")),
    ), e, m


def run(datadir, outdir, perturb=None, reuse=False, true_inputs=False, verbose=True, cfg=None):
    cfg = cfg or P.CONFIG
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    tables = load_tables(datadir)
    log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True) if verbose else None

    panel = P.build_panel(tables, cfg)
    log("panel", panel.shape)
    r_series = None
    if "risk_free" in tables:                 # optional file: date, r (annual decimal), e.g. 10-year Treasury from FRED
        rf = tables["risk_free"].copy()
        rf["date"] = pd.to_datetime(rf["date"]) + pd.offsets.MonthEnd(0)
        r_series = rf.set_index("date")["r"]
    cal = P.calibrate(panel, tables, cfg, r_series=r_series)
    if true_inputs:   # synthetic only: replace calibrated mu/sigma/lambda with the true industry values
        ti = tables["truth_industry"][["gind", "mu_Q", "sigma", "lam"]].rename(columns={"mu_Q": "mu"})
        cal = cal.drop(columns=["mu", "sigma", "lam"]).merge(ti, on="gind")
    cal = apply_perturbation(cal, perturb)
    cal.to_csv(os.path.join(outdir, "calibrated_params.csv"), index=False)
    p = P.merge_params(panel, cal)
    log("calibrated", cal.shape)
    est_file = os.path.join(outdir, "estimated_theta_eta.csv")
    if reuse and os.path.exists(est_file):
        est = pd.read_csv(est_file, parse_dates=["date"])
    else:
        est = P.estimate_theta_eta(p, cfg)
        est.to_csv(est_file, index=False)
    log("estimated theta/eta", est.shape)
    mis = P.misvaluation(p, est, cfg)
    mis["ind_goap"] = mis.groupby(["gind", "date"])["go_ap"].transform("median")
    ap = P.estimate_ap_only(p, cfg)
    mis = mis.merge(ap[["gvkey", "date", "log_misval_ap", "theta_ap"]], on=["gvkey", "date"], how="left")
    keep = ["gvkey", "date", "gind", "me", "V_obs", "ebitda", "ppegt", "debt", "mu", "sigma", "lam", "r",
            "theta_hat", "eta_hat", "V_model", "AP_model", "GO_model", "misval", "log_misval", "go_ap",
            "ind_goap", "log_misval_ap", "ret", "ret_fwd"]
    mis[keep].to_csv(os.path.join(outdir, "misvaluation_panel.csv"), index=False)
    log("misvaluation", mis.shape)

    ff = tables["ff_factors"]
    res = {"run": os.path.basename(outdir), "perturb": perturb, "true_inputs": true_inputs, "n_firm_months": int(len(mis)),
           "first_month": str(mis["date"].min())[:10], "last_month": str(mis["date"].max())[:10]}

    # ---- main decile test (Table VI)
    vw, d = B.decile_portfolios(mis, "log_misval", 10, "me")
    ew, _ = B.decile_portfolios(mis, "log_misval", 10, "equal")
    tab_vw = B.alpha_table(vw, ff); tab_ew = B.alpha_table(ew, ff)
    tab_vw.to_csv(os.path.join(outdir, "alphas_deciles_vw.csv"), index=False)
    tab_ew.to_csv(os.path.join(outdir, "alphas_deciles_ew.csv"), index=False)
    vw.to_csv(os.path.join(outdir, "returns_deciles_vw.csv"))
    a, t = B.pivot_alphas(tab_vw)
    res["vw_alpha_pct"] = a.round(3).to_dict()
    res["vw_t"] = t.round(2).to_dict()
    ae, te = B.pivot_alphas(tab_ew)
    res["ew_spread"] = {"alpha_pct": ae.loc["1-10"].round(3).to_dict(), "t": te.loc["1-10"].round(2).to_dict()}
    res["mean_log_misval_by_decile"] = d.groupby("dec")["log_misval"].mean().round(3).to_dict()
    res["avg_firms_per_decile"] = float(d.groupby(["date", "dec"]).size().mean())
    res["turnover_D1"] = B.turnover(d, "dec", 1)
    res["turnover_D10"] = B.turnover(d, "dec", 10)

    # ---- persistence (Table V)
    tr = B.transition_table(d)
    tr.to_csv(os.path.join(outdir, "transition_table.csv"))
    res["transition"] = tr.round(2).to_dict()

    # ---- double sort on industry GO/AP terciles (Table X)
    ds, _ = B.double_sort(mis, "ind_goap", 3, "log_misval", 10)
    res["double_sort"] = {}
    for k, w in ds.items():
        tb = B.alpha_table(w[["1-10"]], ff, models={m: B.FACTOR_MODELS[m] for m in ["Mean", "CAPM", "FF3"]})
        res["double_sort"][f"GOAP_tercile_{k}"] = {r.model: (round(r.alpha_pct, 3), round(r.t, 2)) for r in tb.itertuples()}

    # ---- counterfactual: AP-only model (Table XI)
    vw_ap, _ = B.decile_portfolios(mis.dropna(subset=["log_misval_ap"]), "log_misval_ap", 10, "me")
    tb = B.alpha_table(vw_ap[["1-10"]], ff, models={m: B.FACTOR_MODELS[m] for m in ["Mean", "CAPM", "FF3"]})
    res["ap_only_spread"] = {r.model: (round(r.alpha_pct, 3), round(r.t, 2)) for r in tb.itertuples()}
    res["corr_full_vs_ap_only_signal"] = float(mis[["log_misval", "log_misval_ap"]].corr(method="spearman").iloc[0, 1])

    # ---- Fama-MacBeth (Table VIII, simplified controls)
    mis["log_me"] = np.log(mis["me"])
    res["fama_macbeth"] = B.fama_macbeth(mis, "ret_fwd", ("log_misval",)).round(4).to_dict("records")
    res["fama_macbeth_ctrl"] = B.fama_macbeth(mis, "ret_fwd", ("log_misval", "log_me")).round(4).to_dict("records")

    # ---- what drives the signal: correlation with simple multiples
    mis["ev_ebitda"] = np.log(mis["V_obs"] / mis["ebitda"])
    mis["ebitda_ppe"] = np.log(mis["ebitda"] / mis["ppegt"])
    mis["ev_ebitda_indrel"] = mis["ev_ebitda"] - mis.groupby(["gind", "date"])["ev_ebitda"].transform("median")
    res["spearman_signal_vs_indrel_ev_ebitda"] = float(mis[["log_misval", "ev_ebitda_indrel"]].corr(method="spearman").iloc[0, 1])
    res["spearman_aponly_vs_indrel_ev_ebitda"] = float(mis[["log_misval_ap", "ev_ebitda_indrel"]].corr(method="spearman").iloc[0, 1])
    res["spearman_signal_vs_ebitda_ppe"] = float(mis[["log_misval", "ebitda_ppe"]].corr(method="spearman").iloc[0, 1])

    res["estimates_summary"] = {"theta_hat_median": float(est["theta_hat"].median()),
                                "eta_hat_median": float(est["eta_hat"].median()),
                                "share_mu_capped": float(cal["mu_capped"].mean()),
                                "goap_median": float(mis["go_ap"].median()),
                                "goap_mean": float(mis["go_ap"].clip(upper=50).mean())}

    # ---- truth checks (synthetic data only)
    if "truth_firm_month" in tables:
        q, e, m = signal_quality(mis, tables["truth_firm_month"], tables["truth_industry"], est)
        res["signal_quality"] = q
        e.to_csv(os.path.join(outdir, "theta_eta_vs_truth.csv"))
        # an "oracle" portfolio sorted on TRUE mispricing = upper bound for the strategy
        vw_or, _ = B.decile_portfolios(m, "log_true_misval", 10, "me")
        tb = B.alpha_table(vw_or[["1-10"]], ff, models={k: B.FACTOR_MODELS[k] for k in ["Mean", "CAPM"]})
        res["oracle_spread"] = {r.model: (round(r.alpha_pct, 3), round(r.t, 2)) for r in tb.itertuples()}
        # misvaluation sort on B/M alone (is the model better than a plain value sort?)
    res["runtime_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    log("done")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="base", choices=["base", "null", "total"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--perturb", nargs="*", default=[], help="e.g. mu=+0.02 sigma=*1.3 lam=+0.03 r=+0.01")
    ap.add_argument("--data", default=None, help="folder with real data in the pipeline's file format, e.g. from bloomberg_pull_template.py")
    ap.add_argument("--name", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse saved theta/eta estimates in the output folder")
    ap.add_argument("--true-inputs", action="store_true", help="synthetic only: use true mu/sigma/lambda instead of calibrating them")
    args = ap.parse_args()
    perturb = dict(kv.split("=") for kv in args.perturb) if args.perturb else None
    cfg = None
    if args.data:
        datadir = args.data
        name = args.name or "real_data"
        cfg = P.REAL_DATA_CONFIG   # paper's Internet Appendix timing and LTG procedure
    else:
        import make_dummy_data as M
        mode = {"base": "go", "null": "none", "total": "total"}[args.scenario]
        datadir = os.path.join("data", f"{args.scenario}_seed{args.seed}")
        if not os.path.exists(os.path.join(datadir, "crsp_monthly.csv")):
            print("generating synthetic data ->", datadir)
            M.main(datadir, mispricing=mode, seed=args.seed)
        name = args.name or (f"{args.scenario}_seed{args.seed}" + ("_trueinputs" if args.true_inputs else "")
                             + ("" if not perturb else "_" + "_".join(f"{k}{v}".replace("*", "x") for k, v in perturb.items())))
    res = run(datadir, os.path.join("outputs", name), perturb, reuse=args.reuse, true_inputs=args.true_inputs, cfg=cfg)
    print(json.dumps({k: res[k] for k in ["vw_alpha_pct", "ew_spread", "double_sort", "ap_only_spread"] if k in res}, indent=1, default=str)[:3000])
