"""
lmz_pipeline.py
---------------
Signal construction for the Lyandres-Matveyev-Zhdanov (2020) strategy:

  1. build_panel()            firm-month panel: market value, EBITDA, PP&E, debt
  2. calibrate()              industry-month mu, sigma, lambda (paper Sec. 3.1)
  3. estimate_theta_eta()     rolling 12-month L1 fit of theta, eta per industry (eq. 7)
  4. misvaluation()           epsilon = observed value / model value (eq. 6)
  5. estimate_ap_only()       counterfactual with growth options shut down (eq. 14)

Input tables use Compustat/CRSP/IBES-style column names, or the simpler
Bloomberg-style columns (see README).  The same code runs on the synthetic data
from make_dummy_data.py or on real data from bloomberg_pull_template.py.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from lmz_model import TAU, firm_value, theta_upper_bound, value_AP

CONFIG = dict(
    tau=TAU,
    mrp=0.06,              # market risk premium (paper: 6% per year)
    lag_months=4,          # months after fiscal year-end before annual data is usable (paper does not say; 4 is a common choice)
    max_staleness=18,      # drop annual data older than this many months (after lag)
    est_window=12,         # months of data used to estimate theta, eta (paper: t-12..t-1)
    beta_window=36, beta_min=24,
    sales_quarters=8,      # quarters of sales used for demand volatility
    dep_years=3,           # years of DP/PPEGT used for depreciation
    ltg_months=3,          # IBES forecasts within the last 3 months
    debt_beta=0.0,         # paper: CAPM debt beta from credit spreads by distress quintile; 0 in the dummy world
    r_annual=0.05,         # risk-free rate if no time series supplied
    mu_max_spread=0.005,   # cap mu at r - 0.5% so the model stays finite
    exclude_sectors=(40, 55),
    grid_theta=16, grid_eta=16,
    eta_range=(1.0, 5.0),  # paper's estimates span 1.00-1.36; the 1.00 minimum suggests a lower bound of 1 ($1 of capital costs >= $1)
    min_obs=30,            # minimum firm-months in an industry window to estimate
    # --- analyst growth forecasts -> demand drift mu
    ltg_method="raw",      # "raw": use LTG as given (dummy data, whose analysts are unbiased)
                           # "lmz": Internet Appendix E procedure (median of firm medians, CKL adjustment,
                           #        5 years at adjusted LTG then 2% forever, converted to one perpetual rate)
    ltg_exclude_current=False,  # True = forecasts from months t-3..t-1 only (Internet Appendix Fig. IA.1)
    ltg_ckl=(0.007264043, 0.408605737),   # Adjusted LTG = a + b * median LTG  (eq. IA.26)
    ltg_terminal=0.02, ltg_years=5,       # eq. IA.27-28
)

# Settings that follow the paper's Internet Appendix exactly; used automatically for real data.
REAL_DATA_CONFIG = dict(
    CONFIG,
    lag_months=6,              # Fig. IA.1: accounting data at least six months old
    ltg_method="lmz",
    ltg_exclude_current=True,  # Fig. IA.1: analyst forecasts from months t-3 .. t-1
)


def terminal_equivalent_growth(g5, rA, gT=0.02, years=5):
    """Perpetual growth rate g' that gives the same value as growing at g5 for
    `years` years and at gT afterwards, discounted at rA (Internet Appendix eq. IA.28).
    The printed IA.27 shows '1 +' inside the bracket and no growth on the terminal
    cash flow; both look like typos (as printed it implies ~9% perpetual growth for a
    15% LTG).  We use the standard two-stage formula, which gives ~3.5% for a 15% LTG,
    in line with the ~3% industry growth rates in Table IA.3."""
    g5, rA = np.asarray(g5, float), np.asarray(rA, float)
    rA = np.maximum(rA, gT + 0.005)
    ratio = (1 + g5) / (1 + rA)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(rA - g5) > 1e-9, (1 - ratio**years) / (rA - g5), years / (1 + rA))
    V = annuity + ratio**years / (rA - gT)
    return rA - 1.0 / V


# --------------------------------------------------------------------------
# 1. Panel
# --------------------------------------------------------------------------
def _month_end(s):
    return pd.to_datetime(s) + pd.offsets.MonthEnd(0)


def build_panel(tables, cfg=CONFIG):
    crsp = tables["crsp_monthly"].copy()
    crsp["date"] = _month_end(crsp["date"])
    if "me" not in crsp.columns:                                   # Bloomberg-style files supply market cap directly
        crsp["me"] = crsp["prc"].abs() * crsp["shrout"] / 1e3      # $ millions
    crsp = crsp.sort_values(["gvkey", "date"])
    crsp["ret_fwd"] = crsp.groupby("gvkey")["ret"].shift(-1)       # next month's return
    nxt = crsp.groupby("gvkey")["date"].shift(-1)
    crsp.loc[nxt != crsp["date"] + pd.offsets.MonthEnd(1), "ret_fwd"] = np.nan

    a = tables["compustat_annual"].copy()
    a["datadate"] = _month_end(a["datadate"])
    a["avail"] = a["datadate"] + pd.offsets.MonthEnd(cfg["lag_months"])
    if "ebitda" not in a.columns:                                  # Bloomberg-style files supply EBITDA directly
        a["ebitda"] = a["sale"] - a["cogs"] - a["xsga"]
    if "debt" not in a.columns:
        a["debt"] = a["dlc"].fillna(0) + a["dltt"].fillna(0)
    if "at" not in a.columns:
        a["at"] = np.nan
    a["be"] = a["at"] - a["debt"]
    keep = ["gvkey", "avail", "datadate", "gind", "gsector", "ppegt", "ebitda", "debt", "be", "at"]
    a = a[keep].sort_values("avail")
    p = pd.merge_asof(crsp.sort_values("date"), a, left_on="date", right_on="avail", by="gvkey")
    stale = (p["date"] - p["avail"]).dt.days / 30.44 > cfg["max_staleness"]
    p = p[~stale & p["avail"].notna()]
    p = p[~pd.to_numeric(p["gsector"], errors="coerce").isin(cfg["exclude_sectors"])]
    p["V_obs"] = p["me"] + p["debt"]
    p["usable"] = (p["ebitda"] > 0) & (p["ppegt"] > 0) & (p["me"] > 0)
    return p.sort_values(["gvkey", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Calibration: mu, sigma, lambda by industry-month
# --------------------------------------------------------------------------
def equity_betas(tables, cfg=CONFIG):
    crsp = tables["crsp_monthly"][["gvkey", "date", "ret"]].copy()
    crsp["date"] = _month_end(crsp["date"])
    ff = tables["ff_factors"].copy()
    ff["date"] = _month_end(ff["date"])
    d = crsp.merge(ff[["date", "mktrf", "rf"]], on="date").sort_values(["gvkey", "date"])
    d["ex"] = d["ret"] - d["rf"]
    w, mn = cfg["beta_window"], cfg["beta_min"]
    g = d.groupby("gvkey")
    d["xy"] = d["ex"] * d["mktrf"]; d["xx"] = d["mktrf"] ** 2
    roll = lambda c: g[c].transform(lambda s: s.rolling(w, min_periods=mn).mean())
    mx, my, mxy, mxx = roll("mktrf"), roll("ex"), roll("xy"), roll("xx")
    d["beta_eq"] = (mxy - mx * my) / (mxx - mx ** 2)
    return d[["gvkey", "date", "beta_eq"]]


def _sales_vol_one_firm(df, nq):
    """Demand volatility from the last `nq` quarters of sales: regress log sales
    on fiscal-quarter dummies, take residual series, std of its quarterly
    changes, annualise x2.  (Paper's description is brief; this is our reading.)"""
    ls = np.log(df["saleq"].values)
    q = df["fqtr"].values
    out = np.full(len(df), np.nan)
    for k in range(nq - 1, len(df)):
        w = slice(k - nq + 1, k + 1)
        y, qq = ls[w], q[w]
        if np.isnan(y).any():
            continue
        seas = np.array([y[qq == j].mean() if (qq == j).any() else 0.0 for j in qq])
        resid = y - seas
        out[k] = np.std(np.diff(resid), ddof=1) * 2.0
    return out


def calibrate(panel, tables, cfg=CONFIG, r_series=None):
    """Returns DataFrame [gind, date, mu, sigma, lam, r] (annual decimals)."""
    months = np.sort(panel["date"].unique())
    firm_ind = panel[["gvkey", "date", "gind", "me", "debt"]]

    # --- lambda: industry median of firm 3-year average DP / lagged PPEGT
    a = tables["compustat_annual"].copy()
    a["datadate"] = _month_end(a["datadate"])
    a = a.sort_values(["gvkey", "datadate"])
    a["dep"] = a["dp"] / a.groupby("gvkey")["ppegt"].shift(1)
    a["dep3"] = a.groupby("gvkey")["dep"].transform(lambda s: s.rolling(cfg["dep_years"], min_periods=1).mean())
    a["avail"] = a["datadate"] + pd.offsets.MonthEnd(cfg["lag_months"])
    lam_f = pd.merge_asof(firm_ind.sort_values("date"), a[["gvkey", "avail", "dep3"]].dropna().sort_values("avail"),
                          left_on="date", right_on="avail", by="gvkey")
    lam = lam_f.groupby(["gind", "date"])["dep3"].median().rename("lam")

    # --- sigma: industry median firm sales volatility
    q = tables["compustat_quarterly"].copy()
    q["datadate"] = _month_end(q["datadate"])
    q = q.sort_values(["gvkey", "datadate"])
    if "fqtr" not in q.columns:                                    # Bloomberg-style files: use calendar quarter
        q["fqtr"] = (q["datadate"].dt.month - 1) // 3 + 1
    q["svol"] = np.nan
    for gv, idx in q.groupby("gvkey").groups.items():
        q.loc[idx, "svol"] = _sales_vol_one_firm(q.loc[idx], cfg["sales_quarters"])
    q["avail"] = q["datadate"] + pd.offsets.MonthEnd(min(cfg["lag_months"], 3))
    sig_f = pd.merge_asof(firm_ind.sort_values("date"), q[["gvkey", "avail", "svol"]].dropna().sort_values("avail"),
                          left_on="date", right_on="avail", by="gvkey")
    sigma = sig_f.groupby(["gind", "date"])["svol"].median().rename("sigma")

    # --- mu: analysts' LTG (physical) minus asset beta * MRP  (risk-neutral drift)
    ib = tables["ibes_ltg"].copy()
    ib["statpers"] = _month_end(ib["statpers"])
    if "numest" not in ib.columns:
        ib["numest"] = 1
    ib = ib.merge(firm_ind[["gvkey", "gind"]].drop_duplicates("gvkey"), on="gvkey")
    lmz = cfg.get("ltg_method", "raw") == "lmz"
    skip = 1 if cfg.get("ltg_exclude_current") else 0
    firm_col = "medltg" if (lmz and "medltg" in ib.columns) else "meanltg"   # firm-level median across analysts
    rows = []
    ib_g = {k: v for k, v in ib.groupby("gind")}
    for gind, df in ib_g.items():
        df = df.sort_values("statpers")
        for t in months:
            hi = t - pd.offsets.MonthEnd(skip) if skip else t
            lo = hi - pd.offsets.MonthEnd(cfg["ltg_months"])
            w = df[(df["statpers"] <= hi) & (df["statpers"] > lo)]
            w = w.sort_values("statpers").drop_duplicates("gvkey", keep="last")
            if len(w):
                v = w[firm_col] / 100.0
                rows.append((gind, t, float(np.median(v)) if lmz else np.average(v, weights=w["numest"])))
    ltg = pd.DataFrame(rows, columns=["gind", "date", "ltg"]).set_index(["gind", "date"])["ltg"]

    b = equity_betas(tables, cfg)
    fb = firm_ind.merge(b, on=["gvkey", "date"], how="left")
    w_e = fb["me"] / (fb["me"] + fb["debt"])
    fb["beta_asset"] = fb["beta_eq"] * w_e + cfg["debt_beta"] * (1 - w_e)
    bavg = fb.groupby(["gind", "date"])["beta_asset"].mean().rename("beta_asset")
    bmed = fb.groupby(["gind", "date"])["beta_asset"].median().rename("beta_asset_median")

    cal = pd.concat([lam, sigma, ltg, bavg, bmed], axis=1).reset_index()
    if r_series is None:
        cal["r"] = cfg["r_annual"]
    else:
        cal = cal.merge(r_series.rename("r"), left_on="date", right_index=True, how="left")
    if lmz:
        a_, b_ = cfg["ltg_ckl"]
        cal["ltg_adjusted"] = a_ + b_ * cal["ltg"]                                   # eq. IA.26
        cal["cost_of_capital"] = cal["r"] + cal["beta_asset_median"] * cfg["mrp"]    # median firm's CAPM cost of capital
        cal["ltg_perpetual"] = terminal_equivalent_growth(cal["ltg_adjusted"], cal["cost_of_capital"],
                                                          cfg["ltg_terminal"], cfg["ltg_years"])
        growth = cal["ltg_perpetual"]
    else:
        growth = cal["ltg"]
    cal["mu_raw"] = growth - cal["beta_asset"] * cfg["mrp"]
    cal["mu"] = np.minimum(cal["mu_raw"], cal["r"] - cfg["mu_max_spread"])
    cal["mu_capped"] = cal["mu_raw"] > cal["mu"]
    return cal.dropna(subset=["mu", "sigma", "lam", "r"])


# --------------------------------------------------------------------------
# 3. Estimation of theta, eta (eq. 7)
# --------------------------------------------------------------------------
def _l1_objective(theta, eta, V, pi, K, mu, sig, lam, r, tau):
    x = pi / K ** theta
    v = firm_value(K, x, theta, eta, mu, sig, lam, r, tau)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.abs(np.log(V / v))
    e = np.where(np.isfinite(e), e, 1e3)
    return e.sum(axis=-1)


def fit_one_window(obs, cfg=CONFIG):
    """obs: DataFrame of firm-months in the estimation window with columns
    V_obs, ebitda, ppegt, mu, sigma, lam, r.  Returns (theta, eta, objective/n)."""
    V, pi, K = obs["V_obs"].values, obs["ebitda"].values, obs["ppegt"].values
    mu, sg, lm, r = obs["mu"].values, obs["sigma"].values, obs["lam"].values, obs["r"].values
    tau = cfg["tau"]
    tmax = float(np.min(theta_upper_bound(mu, sg, lm, r))) - 1e-3
    th_grid = np.linspace(0.03, tmax, cfg["grid_theta"])
    et_grid = np.geomspace(*cfg["eta_range"], cfg["grid_eta"])
    TH, ET = np.meshgrid(th_grid, et_grid, indexing="ij")
    TH, ET = TH.reshape(-1, 1), ET.reshape(-1, 1)
    obj = _l1_objective(TH, ET, V, pi, K, mu, sg, lm, r, tau)
    k = int(np.argmin(obj))
    x0 = np.array([TH[k, 0], np.log(ET[k, 0])])
    lo_eta, hi_eta = cfg["eta_range"]

    def f(z):
        if z[0] <= 0.01 or z[0] >= tmax or not (np.log(lo_eta) <= z[1] <= np.log(hi_eta)):
            return 1e12
        th, et = z[0], np.exp(z[1])
        return float(_l1_objective(th, et, V, pi, K, mu, sg, lm, r, tau))

    res = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=300))
    z = res.x if res.fun <= obj[k] else x0
    return z[0], float(np.exp(z[1])), min(res.fun, obj[k]) / len(V)


def merge_params(panel, cal):
    p = panel.merge(cal[["gind", "date", "mu", "sigma", "lam", "r"]], on=["gind", "date"], how="left")
    p["est_ok"] = p["usable"] & p[["mu", "sigma", "lam", "r"]].notna().all(axis=1)
    return p


def estimate_theta_eta(p, cfg=CONFIG, verbose=False):
    """Rolling estimation: for each industry j and month t, fit on months
    t-12 .. t-1.  Returns DataFrame [gind, date, theta_hat, eta_hat, fit_mad, n_obs]."""
    rows = []
    d = p[p["est_ok"]]
    for gind, df in d.groupby("gind"):
        months = np.sort(df["date"].unique())
        by_m = {m: g for m, g in df.groupby("date")}
        for i in range(cfg["est_window"], len(months)):
            t = months[i]
            lo = t - pd.offsets.MonthEnd(cfg["est_window"])
            win = [by_m[m] for m in months[max(0, i - cfg["est_window"]):i] if m >= lo]
            obs = pd.concat(win)
            if len(obs) < cfg["min_obs"]:
                continue
            th, et, fit = fit_one_window(obs, cfg)
            rows.append((gind, t, th, et, fit, len(obs)))
        if verbose:
            print(f"  industry {gind}: {len(months)} months done")
    return pd.DataFrame(rows, columns=["gind", "date", "theta_hat", "eta_hat", "fit_mad", "n_obs"])


# --------------------------------------------------------------------------
# 4. Misvaluation (eq. 6)
# --------------------------------------------------------------------------
def misvaluation(p, est, cfg=CONFIG):
    m = p[p["est_ok"]].merge(est, on=["gind", "date"], how="inner")
    x = m["ebitda"] / m["ppegt"] ** m["theta_hat"]
    v, ap, go = firm_value(m["ppegt"].values, x.values, m["theta_hat"].values, m["eta_hat"].values,
                           m["mu"].values, m["sigma"].values, m["lam"].values, m["r"].values,
                           cfg["tau"], return_parts=True)
    m["V_model"], m["AP_model"], m["GO_model"] = v, ap, go
    m["misval"] = m["V_obs"] / m["V_model"]
    m["log_misval"] = np.log(m["misval"])
    m["go_ap"] = m["GO_model"] / m["AP_model"]
    bad = ~np.isfinite(m["log_misval"]) | ~np.isfinite(m["go_ap"])
    return m[~bad].copy()


# --------------------------------------------------------------------------
# 5. Counterfactual: assets-in-place-only model (eq. 14)
# --------------------------------------------------------------------------
def estimate_ap_only(p, cfg=CONFIG):
    """Fit theta for V = (1-tau)*EBITDA / (r - mu + lam*theta) with the same
    rolling L1 objective.  Note: with x = EBITDA/K^theta, AP no longer depends on
    K, so within an industry this signal is just EV/EBITDA rescaled."""
    rows = []
    d = p[p["est_ok"]]
    tau = cfg["tau"]
    for gind, df in d.groupby("gind"):
        months = np.sort(df["date"].unique())
        by_m = {m: g for m, g in df.groupby("date")}
        for i in range(cfg["est_window"], len(months)):
            t = months[i]
            obs = pd.concat([by_m[m] for m in months[max(0, i - cfg["est_window"]):i]])
            if len(obs) < cfg["min_obs"]:
                continue
            V, pi = obs["V_obs"].values, obs["ebitda"].values
            mu, lm, r = obs["mu"].values, obs["lam"].values, obs["r"].values
            f = lambda th: np.abs(np.log(V / ((1 - tau) * pi / (r - mu + lm * th)))).sum()
            res = minimize_scalar(f, bounds=(0.01, 0.99), method="bounded")
            rows.append((gind, t, res.x))
    est = pd.DataFrame(rows, columns=["gind", "date", "theta_ap"])
    m = d.merge(est, on=["gind", "date"], how="inner")
    m["V_model_ap"] = (1 - tau) * m["ebitda"] / (m["r"] - m["mu"] + m["lam"] * m["theta_ap"])
    m["log_misval_ap"] = np.log(m["V_obs"] / m["V_model_ap"])
    return m
