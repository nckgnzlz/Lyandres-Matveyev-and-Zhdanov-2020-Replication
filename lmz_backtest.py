"""
lmz_backtest.py
---------------
Portfolio tests from the paper, generic enough to run on dummy or real data.

  decile_portfolios()   monthly misvaluation deciles, VW or EW next-month returns (Table VI)
  alpha_table()         mean returns and factor alphas with Newey-West(6) t-stats
  transition_table()    average future decile of today's deciles (Table V)
  double_sort()         deciles within terciles of industry-median GO/AP (Table X)
  fama_macbeth()        cross-sectional regressions (Table VIII)
  turnover()            one-way monthly turnover of a decile portfolio
"""
import numpy as np
import pandas as pd

FACTOR_MODELS = {
    "Mean": [],
    "CAPM": ["mktrf"],
    "FF3": ["mktrf", "smb", "hml"],
    "FF3+MOM": ["mktrf", "smb", "hml", "umd"],
    "FF5": ["mktrf", "smb", "hml", "rmw", "cma"],
    "FF5+MOM": ["mktrf", "smb", "hml", "rmw", "cma", "umd"],
    "Q": ["mktrf", "r_me", "r_ia", "r_roe"],
    "Q+MOM": ["mktrf", "r_me", "r_ia", "r_roe", "umd"],
}


def nw_ols(y, X, lags=6):
    """OLS with Newey-West (Bartlett) standard errors. X should include a constant."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return b, se, b / se, n


def assign_groups(df, col, n, by=("date",)):
    """Equal-count groups 1..n of `col` within each `by` cell (1 = lowest)."""
    def q(s):
        r = s.rank(method="first")
        return np.ceil(r / len(s) * n).clip(1, n).astype(int)
    return df.groupby(list(by))[col].transform(q)


def decile_portfolios(df, signal="log_misval", n=10, weight="me", ret="ret_fwd", by=("date",), group_name="dec"):
    """Returns wide DataFrame indexed by HOLDING month (formation month + 1)."""
    ew = weight is None or weight == "equal"
    d = df.dropna(subset=[signal, ret] + ([] if ew else [weight])).copy()
    d[group_name] = assign_groups(d, signal, n, by)
    if ew:
        pr = d.groupby(["date", group_name])[ret].mean()
    else:
        d["_rw"] = d[ret] * d[weight]
        g = d.groupby(["date", group_name])
        pr = g["_rw"].sum() / g[weight].sum()          # value-weighted return
    w = pr.unstack(group_name)
    w.index = pd.to_datetime(w.index) + pd.offsets.MonthEnd(1)
    w["1-%d" % n] = w[1] - w[n]
    return w, d


def alpha_table(ports, factors, models=None, lags=6, rf_adjust=True):
    """ports: wide DataFrame of portfolio returns (holding-month index).
    Long-short columns (names containing '-') are not rf-adjusted.
    Returns tidy DataFrame [portfolio, model, alpha_pct, t]."""
    f = factors.copy()
    f["date"] = pd.to_datetime(f["date"]) + pd.offsets.MonthEnd(0)
    f = f.set_index("date")
    models = models or FACTOR_MODELS
    out = []
    for col in ports.columns:
        y = ports[col]
        j = pd.concat([y.rename("y"), f], axis=1, join="inner").dropna(subset=["y"])
        is_ls = isinstance(col, str) and "-" in col
        yy = j["y"] - (0 if (is_ls or not rf_adjust) else j["rf"])
        for name, cols in models.items():
            if any(c not in j.columns or j[c].isna().all() for c in cols):
                continue
            jj = pd.concat([yy.rename("y"), j[cols]], axis=1).dropna()
            X = np.column_stack([np.ones(len(jj))] + [jj[c].values for c in cols])
            b, se, t, nobs = nw_ols(jj["y"].values, X, lags)
            out.append(dict(portfolio=col, model=name, alpha_pct=100 * b[0], t=t[0], n_months=nobs,
                            **{f"b_{c}": b[i + 1] for i, c in enumerate(cols)}))
    return pd.DataFrame(out)


def pivot_alphas(tab):
    a = tab.pivot(index="portfolio", columns="model", values="alpha_pct")
    t = tab.pivot(index="portfolio", columns="model", values="t")
    order = [m for m in FACTOR_MODELS if m in a.columns]
    return a[order], t[order]


def transition_table(d, group_name="dec", horizons=(1, 3, 6, 12, 24, 36)):
    """Average future decile for firms currently in each decile (paper Table V)."""
    x = d[["gvkey", "date", group_name]].copy()
    x["date"] = pd.to_datetime(x["date"])
    res = {}
    for h in horizons:
        y = x.copy()
        y["date"] = y["date"] - pd.offsets.MonthEnd(h)      # value h months later, aligned to today
        m = x.merge(y.rename(columns={group_name: "fut"}), on=["gvkey", "date"])
        res[f"{h}m"] = m.groupby(group_name)["fut"].mean()
    return pd.DataFrame(res)


def double_sort(df, sort1="ind_goap", n1=3, signal="log_misval", n2=10, weight="me", ret="ret_fwd"):
    """Independent terciles on `sort1` (industry-median GO/AP), deciles of the
    misvaluation signal within each tercile.  Returns dict tercile -> wide returns."""
    d = df.dropna(subset=[sort1, signal, ret, weight]).copy()
    d["t1"] = assign_groups(d, sort1, n1)
    out = {}
    for k, g in d.groupby("t1"):
        w, _ = decile_portfolios(g, signal, n2, weight, ret)
        out[k] = w
    return out, d


def fama_macbeth(df, y="ret_fwd", xs=("log_misval",), lags=6, winsor=(0.01, 0.99)):
    rows = []
    for dt, g in df.dropna(subset=[y, *xs]).groupby("date"):
        if len(g) < len(xs) + 10:
            continue
        X = g[list(xs)].copy()
        for c in xs:
            lo, hi = X[c].quantile(list(winsor))
            X[c] = X[c].clip(lo, hi)
        X = np.column_stack([np.ones(len(g)), X.values])
        b = np.linalg.lstsq(X, g[y].values, rcond=None)[0]
        rows.append(b)
    B = np.array(rows)
    res = []
    for i, name in enumerate(["const", *xs]):
        b, se, t, n = nw_ols(B[:, i], np.ones((len(B), 1)), lags)
        res.append(dict(var=name, coef=b[0], t=t[0], n_months=n))
    return pd.DataFrame(res)


def turnover(d, group_name="dec", group=1, weight="me", ret="ret_fwd"):
    """Average one-way monthly turnover of a value-weighted decile portfolio,
    accounting for weight drift from returns."""
    g = d[d[group_name] == group][["gvkey", "date", weight, ret]].copy()
    g["w"] = g[weight] / g.groupby("date")[weight].transform("sum")
    dates = np.sort(g["date"].unique())
    tos = []
    prev = None
    for dt in dates:
        cur = g[g["date"] == dt].set_index("gvkey")
        if prev is not None:
            drift = prev["w"] * (1 + prev[ret].fillna(0))
            drift = drift / drift.sum()
            allk = cur.index.union(drift.index)
            tos.append(0.5 * np.abs(cur["w"].reindex(allk, fill_value=0) - drift.reindex(allk, fill_value=0)).sum())
        prev = cur
    return float(np.mean(tos)) if tos else np.nan
