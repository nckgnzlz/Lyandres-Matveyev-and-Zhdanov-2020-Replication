"""
make_dummy_data.py
------------------
Builds a synthetic ("dummy") panel of firms that looks like the Compustat/CRSP/IBES inputs
the paper uses, so the full strategy pipeline can be run and checked against a
known answer before anyone pays for / pulls real data.

How the fake world works
  * Industries have TRUE parameters (theta, eta, mu, sigma, lambda).
  * Each firm follows the paper's model exactly: demand shock x is a GBM,
    capital K depreciates at lambda, and the firm invests optimally whenever
    x hits the investment boundary X(K).
  * Fundamental value = AP + GO from eq. (5) with the firm's own parameters.
  * The MARKET misprices growth options:  market value = AP + GO * exp(g),
    where g is a mean-reverting (AR(1)) firm-level error plus a slower
    industry-level error.  This is the paper's hypothesis made literal.
    `mispricing="none"` gives a null world; `mispricing="total"` misprices the
    whole firm instead of only GO (used to test what the GO/AP sorts detect).
  * Things the econometrician gets wrong on purpose, as in real data:
    firm-specific theta/eta/lambda (model assumes industry-common values),
    annual data released with a lag, EBITDA is a flow over the year while K is
    year-end, noisy analyst growth forecasts, debt from the last balance sheet.

Output files (column names follow Compustat / CRSP / IBES / Ken French):
  compustat_annual.csv, compustat_quarterly.csv, crsp_monthly.csv,
  ibes_ltg.csv, ff_factors.csv
  truth_industry.csv, truth_firm_month.csv   <- the "answer key"
"""
import argparse
import os
import numpy as np
import pandas as pd

from lmz_model import (TAU, beta1, theta_upper_bound, investment_boundary,
                       target_capital, value_AP, value_GO)

DEFAULTS = dict(
    seed=7,
    n_industries=12,
    firms_per_industry=(50, 110),      # uniform range
    start="1980-01", end="2014-12",    # simulation calendar (first 3y = burn-in)
    burn_in_months=36,
    r=0.05,                            # risk-free rate (constant here)
    mrp=0.06, sigma_m=0.16,            # market risk premium / market vol
    mispricing="go",                   # "go" | "total" | "none"
    rho_firm=0.94, sd_firm=0.25,       # firm GO-mispricing AR(1): monthly rho, stationary sd
    rho_ind=0.98, sd_ind=0.15,         # industry GO-mispricing AR(1)
    theta_dev=0.01, eta_dev=0.10, lam_dev=0.15,   # within-industry heterogeneity (model misspecification)
    ltg_coverage=0.7, ltg_firm_sd=0.02, ltg_month_sd=0.01, ltg_bias=0.0,
)


def month_range(start, end):
    return pd.period_range(start, end, freq="M")


def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    months = month_range(cfg["start"], cfg["end"])
    T = len(months)
    dt = 1.0 / 12.0
    r, mrp, sm = cfg["r"], cfg["mrp"], cfg["sigma_m"]

    # ---------------- industries (true parameters) ----------------
    J = cfg["n_industries"]
    ind = pd.DataFrame({
        "gind": [101010 + 10 * j for j in range(J)],   # fake 6-digit GICS industry codes
        "gsector": [(10, 15, 20, 25, 30, 35, 45, 50)[j % 8] for j in range(J)],   # skips 40 (financials) & 55 (utilities)
        "mu_Q": rng.uniform(-0.04, 0.02, J),
        "sigma": rng.uniform(0.08, 0.30, J),
        "lam": rng.uniform(0.06, 0.12, J),
        "eta": rng.uniform(1.00, 1.36, J),
        "a_mkt": rng.uniform(0.20, 0.50, J),   # loading of demand shock on market shock
    })
    tb = theta_upper_bound(ind.mu_Q.values, ind.sigma.values, ind.lam.values, r)
    ind["theta"] = np.minimum(tb - rng.uniform(0.10, 0.22, J), 0.48)
    ind["theta_bound"] = tb
    ind["b_ind"] = 0.45
    # physical drift = risk-neutral drift + demand-beta * MRP
    ind["beta_x"] = ind.sigma * ind.a_mkt / sm
    ind["mu_P"] = ind.mu_Q + ind.beta_x * mrp
    ind["season"] = [rng.normal(0, 0.05, 4) for _ in range(J)]

    # ---------------- firms ----------------
    sizes = rng.integers(cfg["firms_per_industry"][0], cfg["firms_per_industry"][1] + 1, J)
    jj = np.repeat(np.arange(J), sizes)
    N = len(jj)
    I = ind.iloc[jj].reset_index(drop=True)
    th_i = np.clip(I.theta.values + rng.normal(0, cfg["theta_dev"], N), 0.1, I.theta_bound.values - 0.02)
    eta_i = I.eta.values * np.exp(rng.normal(0, cfg["eta_dev"], N))
    lam_i = I.lam.values * np.exp(rng.normal(0, cfg["lam_dev"], N))
    mu_Q, sig, a = I.mu_Q.values, I.sigma.values, I.a_mkt.values
    b = I.b_ind.values
    c = np.sqrt(1 - a**2 - b**2)
    mu_P = I.mu_P.values
    lev = rng.uniform(0.0, 0.35, N)                  # debt / fundamental value
    margin = rng.uniform(0.08, 0.35, N)              # EBITDA / sales
    at_mult = rng.uniform(0.9, 1.5, N)               # book assets / PPE
    covered = rng.random(N) < cfg["ltg_coverage"]
    ltg_firm = rng.normal(0, cfg["ltg_firm_sd"], N)

    K = np.exp(rng.normal(np.log(300), 1.0, N))
    X0 = investment_boundary(K, th_i, eta_i, mu_Q, sig, lam_i, r)
    x = X0 * rng.uniform(0.2, 1.0, N)

    g_f = rng.normal(0, cfg["sd_firm"], N)
    g_j = rng.normal(0, cfg["sd_ind"], J)
    e_f = cfg["sd_firm"] * np.sqrt(1 - cfg["rho_firm"]**2)
    e_j = cfg["sd_ind"] * np.sqrt(1 - cfg["rho_ind"]**2)
    if cfg["mispricing"] == "none":
        g_f[:] = 0; g_j[:] = 0; e_f = 0; e_j = 0

    # storage
    out = {k: np.zeros((T, N)) for k in
           ["x", "K", "pi", "invest", "AP", "GO", "Vtrue", "Vmkt", "D", "E", "div", "ret", "g"]}
    mkt = np.zeros(T)

    D_prev = None
    E_prev = None
    for t in range(T):
        zm = rng.standard_normal()
        zj = rng.standard_normal(J)
        zi = rng.standard_normal(N)
        mkt[t] = mrp * dt + sm * np.sqrt(dt) * zm          # market excess return
        shock = a * zm + b * zj[jj] + c * zi
        x = x * np.exp((mu_P - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * shock)
        K = K * np.exp(-lam_i * dt)
        X = investment_boundary(K, th_i, eta_i, mu_Q, sig, lam_i, r)
        inv = x > X
        Kstar = target_capital(x, th_i, eta_i, mu_Q, sig, lam_i, r)
        dK = np.where(inv, np.maximum(Kstar - K, 0), 0.0)
        K = K + dK
        pi = x * K**th_i                                   # pre-tax EBITDA (annual rate)
        AP = value_AP(K, x, th_i, mu_Q, lam_i, r)
        GO = value_GO(K, x, th_i, eta_i, mu_Q, sig, lam_i, r)
        V = AP + GO
        # mispricing
        g_f = cfg["rho_firm"] * g_f + e_f * rng.standard_normal(N)
        g_j = cfg["rho_ind"] * g_j + e_j * rng.standard_normal(J)
        g = g_f + g_j[jj]
        if cfg["mispricing"] == "total":
            Vm = V * np.exp(g * 0.35)      # same average size of firm-level error, spread over all value
        else:
            Vm = AP + GO * np.exp(g)
        D = lev * V
        E = Vm - D
        cf = (1 - TAU) * pi * dt
        if t == 0:
            div = np.zeros(N); ret = np.full(N, np.nan)
        else:
            div = cf - eta_i * dK - r * dt * D_prev + (D - D_prev)
            ret = (E + div) / E_prev - 1
        for k, v in zip(["x", "K", "pi", "invest", "AP", "GO", "Vtrue", "Vmkt", "D", "E", "div", "ret", "g"],
                        [x, K, pi, eta_i * dK, AP, GO, V, Vm, D, E, div, ret, g]):
            out[k][t] = v
        D_prev, E_prev = D, E

    meta = dict(months=months, jj=jj, N=N, ind=ind, th_i=th_i, eta_i=eta_i, lam_i=lam_i,
                lev=lev, margin=margin, at_mult=at_mult, covered=covered, ltg_firm=ltg_firm,
                mkt=mkt, rng=rng)
    return out, meta


def to_database_tables(out, meta, cfg):
    """Convert simulated paths into Compustat/CRSP/IBES-shaped tables."""
    rng = meta["rng"]
    months = meta["months"]
    T, N = out["K"].shape
    ind = meta["ind"]
    jj = meta["jj"]
    keep = np.arange(T) >= cfg["burn_in_months"]
    gvkey = np.array([f"{100001 + i:06d}" for i in range(N)])
    permno = 10001 + np.arange(N)
    dt = 1.0 / 12.0

    # ---------- CRSP monthly ----------
    shrout = np.maximum(out["E"][cfg["burn_in_months"]] * 1e6 / 30.0 / 1e3, 100.0)  # thousands, ~$30 start price
    rows = []
    for t in np.where(keep)[0]:
        prc = out["E"][t] * 1e6 / (shrout * 1e3)
        rows.append(pd.DataFrame({"permno": permno, "gvkey": gvkey, "date": months[t].to_timestamp("M"),
                                  "prc": prc, "shrout": shrout, "ret": out["ret"][t]}))
    crsp = pd.concat(rows, ignore_index=True)

    # ---------- Compustat annual (fiscal year = calendar year, Dec FYE) ----------
    years = sorted(set(months.year))
    arows = []
    for y in years:
        idx = np.where((months.year == y))[0]
        if idx.max() < cfg["burn_in_months"] or len(idx) < 12:
            continue
        dec = idx[-1]
        ebitda = out["pi"][idx].sum(0) * dt
        dp = (meta["lam_i"] * out["K"][idx]).sum(0) * dt
        sale = ebitda / meta["margin"]
        opex = sale - ebitda
        D = out["D"][dec]
        arows.append(pd.DataFrame({
            "gvkey": gvkey, "fyear": y, "datadate": months[dec].to_timestamp("M"),
            "gind": ind.gind.values[jj], "gsector": ind.gsector.values[jj],
            "at": meta["at_mult"] * out["K"][dec], "ppegt": out["K"][dec],
            "sale": sale, "cogs": 0.75 * opex, "xsga": 0.25 * opex, "dp": dp,
            "dlc": 0.2 * D, "dltt": 0.8 * D, "capx": out["invest"][idx].sum(0)}))
    comp_a = pd.concat(arows, ignore_index=True)

    # ---------- Compustat quarterly sales ----------
    qrows = []
    for y in years:
        for q in range(4):
            idx = np.where((months.year == y) & (((months.month - 1) // 3) == q))[0]
            if len(idx) < 3 or idx.max() < cfg["burn_in_months"]:
                continue
            seas = np.array([ind.season[j][q] for j in range(len(ind))])[jj]
            saleq = out["pi"][idx].sum(0) * dt / meta["margin"] * (1 + seas) * np.exp(rng.normal(0, 0.02, N))
            qrows.append(pd.DataFrame({"gvkey": gvkey, "datadate": months[idx[-1]].to_timestamp("M"),
                                       "fyearq": y, "fqtr": q + 1, "saleq": saleq}))
    comp_q = pd.concat(qrows, ignore_index=True)

    # ---------- IBES long-term growth consensus (percent, like IBES) ----------
    mu_P = ind.mu_P.values[jj]
    lrows = []
    for t in np.where(keep)[0]:
        ltg = mu_P + cfg["ltg_bias"] + meta["ltg_firm"] + rng.normal(0, cfg["ltg_month_sd"], N)
        m = meta["covered"]
        lrows.append(pd.DataFrame({"gvkey": gvkey[m], "statpers": months[t].to_timestamp("M"),
                                   "meanltg": 100 * ltg[m], "numest": rng.integers(1, 12, m.sum())}))
    ibes = pd.concat(lrows, ignore_index=True)

    # ---------- truth ----------
    trows = []
    for t in np.where(keep)[0]:
        trows.append(pd.DataFrame({"gvkey": gvkey, "date": months[t].to_timestamp("M"),
                                   "gind": ind.gind.values[jj], "x": out["x"][t], "K": out["K"][t],
                                   "AP_true": out["AP"][t], "GO_true": out["GO"][t],
                                   "V_true": out["Vtrue"][t], "V_mkt": out["Vmkt"][t],
                                   "g_GO_mispricing": out["g"][t],
                                   "log_true_misval": np.log(out["Vmkt"][t] / out["Vtrue"][t])}))
    truth_fm = pd.concat(trows, ignore_index=True)
    truth_ind = ind.drop(columns=["season"]).copy()

    ff = pd.DataFrame({"date": [m.to_timestamp("M") for m in months[keep]],
                       "mktrf": meta["mkt"][keep], "rf": cfg["r"] / 12.0})
    return dict(crsp_monthly=crsp, compustat_annual=comp_a, compustat_quarterly=comp_q,
                ibes_ltg=ibes, ff_factors=ff, truth_firm_month=truth_fm, truth_industry=truth_ind)


def add_synthetic_ff_factors(tables, lag_months=4):
    """Build SMB, HML and UMD from the synthetic universe (Ken French style,
    simplified to monthly rebalancing) so multi-factor alphas can be run.
    On real data you would download these from Ken French's website instead."""
    crsp = tables["crsp_monthly"].copy()
    crsp["me"] = crsp.prc * crsp.shrout / 1e3
    crsp = crsp.sort_values(["gvkey", "date"])
    crsp["me_lag"] = crsp.groupby("gvkey").me.shift(1)
    crsp["lr"] = np.log1p(crsp.ret)
    # signal for holding month d = cumulative return over months d-12 .. d-2
    crsp["lr_l2"] = crsp.groupby("gvkey").lr.shift(2)
    crsp["mom"] = crsp.groupby("gvkey").lr_l2.transform(lambda s: s.rolling(11, min_periods=11).sum())
    a = tables["compustat_annual"].copy()
    a["be"] = a["at"] - a.dlc - a.dltt
    a["avail"] = a.datadate + pd.offsets.MonthEnd(lag_months)
    a = a.sort_values("avail")
    crsp = crsp.sort_values("date")
    m = pd.merge_asof(crsp, a[["gvkey", "avail", "be"]], left_on="date", right_on="avail", by="gvkey")
    m = m.sort_values(["gvkey", "date"])
    m["bm_lag"] = (m.be / m.me).where(m.be > 0).groupby(m.gvkey).shift(1)
    m["mom_lag"] = m["mom"]
    res = []
    for d, g in m.groupby("date"):
        g = g.dropna(subset=["me_lag", "ret"])
        if len(g) < 50:
            continue
        big = g.me_lag >= g.me_lag.median()
        out = {"date": d}
        for name, col in [("hml", "bm_lag"), ("umd", "mom_lag")]:
            h = g[col]
            if h.notna().sum() < 50:
                out[name] = np.nan
                if name == "hml":
                    out["smb"] = np.nan
                continue
            lo, hi = h.quantile([0.3, 0.7])
            ports = {}
            for sz, sm in [("S", ~big), ("B", big)]:
                for bk, bm in [("L", h <= lo), ("H", h >= hi), ("M", (h > lo) & (h < hi))]:
                    p = g[sm & bm]
                    ports[sz + bk] = np.average(p.ret, weights=p.me_lag) if len(p) else np.nan
            out[name] = 0.5 * (ports["SH"] + ports["BH"]) - 0.5 * (ports["SL"] + ports["BL"])
            if name == "hml":
                out["smb"] = np.nanmean([ports["SL"], ports["SM"], ports["SH"]]) - np.nanmean([ports["BL"], ports["BM"], ports["BH"]])
        res.append(out)
    f = pd.DataFrame(res)
    tables["ff_factors"] = tables["ff_factors"].merge(f, on="date", how="left")
    return tables


def main(outdir="data", **overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    out, meta = simulate(cfg)
    tables = to_database_tables(out, meta, cfg)
    tables = add_synthetic_ff_factors(tables)
    os.makedirs(outdir, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(outdir, name + ".csv"), index=False)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic Compustat/CRSP/IBES-style data for the LMZ strategy test")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--mispricing", default="go", choices=["go", "total", "none"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    t = main(args.outdir, mispricing=args.mispricing, seed=args.seed)
    print({k: v.shape for k, v in t.items()})
