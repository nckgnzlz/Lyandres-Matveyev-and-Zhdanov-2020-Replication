"""
bloomberg_pull_template.py   (UNTESTED TEMPLATE - must run on a PC with a logged-in Bloomberg Terminal)
--------------------------------------------------------------------------------------------------
Pulls the real-data inputs for the LMZ (2020) strategy from Bloomberg and saves them in the
file format the pipeline reads, so afterwards

    python run_test.py --data data/bloomberg --name real_bloomberg

runs the same code on real firms (with the paper's Internet Appendix settings).

Setup (once, on the terminal PC):
    pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
    pip install xbbg

BEFORE A BIG PULL:
  1. Field names in FIELDS are our best guesses. Confirm each one with FLDS on the terminal
     (type FLDS <GO>, search e.g. "long term growth") and edit the dictionary.
  2. Check units: market cap and balance-sheet items must both be in USD millions.
  3. Bloomberg caps how much data an account can download. Keep TEST_MODE = True first
     (50 firms, 5 years), check the output files, then widen START/END and turn TEST_MODE off.
     The script saves each batch, so a stopped run can be resumed.

Two files you download yourself (free) and convert with the helpers at the bottom:
  * Fama-French factors: Ken French's website, "Fama/French 5 Factors (2x3)" and "Momentum Factor" CSVs
  * Risk-free rate r: FRED series GS10 (10-year Treasury, monthly) CSV
"""
import os
import time

import numpy as np
import pandas as pd

OUT = os.path.join("data", "bloomberg")
START, END = "2010-01-01", "2014-12-31"   # widen once the test pull works
UNIVERSE_INDEX = "RAY Index"             # Russell 3000: historical members include later-delisted firms
TEST_MODE = True                          # first 50 firms only
BATCH = 25                                # tickers per request

FIELDS = dict(                            # CONFIRM WITH FLDS
    index_members="INDX_MWEIGHT_HIST",    # bulk field: index members as of END_DATE_OVERRIDE
    mkt_cap="CUR_MKT_CAP",                # market value of equity (USD millions?)
    tot_ret_index="TOT_RETURN_INDEX_GROSS_DVDS",   # total return index -> monthly returns
    ppe_gross="BS_GROSS_FIX_ASSET",       # gross PP&E (paper: Compustat PPEGT)
    ebitda="EBITDA",
    dep="CF_DEPR_AMORT",                  # depreciation & amortization (paper: DP)
    st_debt="BS_ST_BORROW",
    lt_debt="BS_LT_BORROW",
    sales="SALES_REV_TURN",
    ltg="BEST_EST_LONG_TERM_GROWTH",      # analysts' long-term growth consensus, percent
    gics_industry="GICS_INDUSTRY",        # 6-digit GICS industry code (current, not historical)
    gics_sector="GICS_SECTOR",
)


def _blp():
    from xbbg import blp                  # imported here so the helpers below work without Bloomberg
    return blp


def universe():
    """Every firm that was in the index at any year-end between START and END."""
    blp = _blp()
    tickers = set()
    for d in pd.date_range(START, END, freq="YE"):
        m = blp.bds(UNIVERSE_INDEX, FIELDS["index_members"], END_DATE_OVERRIDE=d.strftime("%Y%m%d"))
        col = m.columns[0]                # first column holds tickers like "AAPL UW"
        tickers |= {f"{t.strip()} Equity" for t in m[col].astype(str)}
        time.sleep(1)
    tickers = sorted(tickers)
    return tickers[:50] if TEST_MODE else tickers


def _bdh_long(tickers, fields, start, end, per):
    """bdh for many tickers -> long table [gvkey, date, field1, field2, ...]."""
    blp = _blp()
    frames = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        df = blp.bdh(chunk, fields, start, end, Per=per)
        if df is None or df.empty:
            continue
        try:
            df = df.stack(level=0, future_stack=True).reset_index()
        except TypeError:                 # pandas < 2.1
            df = df.stack(level=0).reset_index()
        df.columns = ["date", "gvkey"] + [str(c).lower() for c in df.columns[2:]]   # field columns in lower case
        frames.append(df)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cached(name, fn):
    f = os.path.join(OUT, "_raw_" + name + ".csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    df = fn()
    df.to_csv(f, index=False)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    tickers = _cached("universe", lambda: pd.DataFrame({"ticker": universe()}))["ticker"].tolist()
    print(len(tickers), "firms")
    start_fund = (pd.Timestamp(START) - pd.DateOffset(years=4)).strftime("%Y-%m-%d")   # history for lambda, sigma

    # --- monthly market data -> crsp_monthly.csv  [gvkey, date, me, ret]
    F = FIELDS
    f = {k: v.lower() for k, v in FIELDS.items()}   # column names in the downloaded tables
    m = _cached("monthly", lambda: _bdh_long(tickers, [F["mkt_cap"], F["tot_ret_index"]], START, END, "M"))
    m = m.sort_values(["gvkey", "date"])
    m["ret"] = m.groupby("gvkey")[f["tot_ret_index"]].pct_change()
    m.rename(columns={f["mkt_cap"]: "me"})[["gvkey", "date", "me", "ret"]].to_csv(os.path.join(OUT, "crsp_monthly.csv"), index=False)

    # --- static GICS codes (current)
    blp = _blp()
    g = blp.bdp(tickers, [F["gics_industry"], F["gics_sector"]])
    g.columns = [str(c).lower() for c in g.columns]
    g = g.rename_axis("gvkey").reset_index().rename(columns={f["gics_industry"]: "gind", f["gics_sector"]: "gsector"})

    # --- annual fundamentals -> compustat_annual.csv  [gvkey, datadate, gind, gsector, ppegt, ebitda, dp, debt]
    a = _cached("annual", lambda: _bdh_long(tickers, [F["ppe_gross"], F["ebitda"], F["dep"], F["st_debt"], F["lt_debt"]],
                                            start_fund, END, "Y"))
    a["debt"] = a[f["st_debt"]].fillna(0) + a[f["lt_debt"]].fillna(0)
    a = a.rename(columns={"date": "datadate", f["ppe_gross"]: "ppegt", f["ebitda"]: "ebitda", f["dep"]: "dp"})
    a = a.merge(g, on="gvkey", how="left")
    a[["gvkey", "datadate", "gind", "gsector", "ppegt", "ebitda", "dp", "debt"]].to_csv(
        os.path.join(OUT, "compustat_annual.csv"), index=False)

    # --- quarterly sales -> compustat_quarterly.csv  [gvkey, datadate, saleq]
    q = _cached("quarterly", lambda: _bdh_long(tickers, [F["sales"]], start_fund, END, "Q"))
    q.rename(columns={"date": "datadate", f["sales"]: "saleq"})[["gvkey", "datadate", "saleq"]].to_csv(
        os.path.join(OUT, "compustat_quarterly.csv"), index=False)

    # --- analysts' long-term growth -> ibes_ltg.csv  [gvkey, statpers, meanltg]
    l = _cached("ltg", lambda: _bdh_long(tickers, [F["ltg"]], START, END, "M"))
    l.rename(columns={"date": "statpers", f["ltg"]: "meanltg"})[["gvkey", "statpers", "meanltg"]].dropna().to_csv(
        os.path.join(OUT, "ibes_ltg.csv"), index=False)
    print("wrote Bloomberg files to", OUT, "- now add ff_factors.csv and risk_free.csv (helpers below)")


# ----------------------------------------------------------------------------------------------
# Helpers for the two free files (no Bloomberg needed)
# ----------------------------------------------------------------------------------------------
def ken_french_to_csv(five_factor_csv, momentum_csv, out=os.path.join(OUT, "ff_factors.csv")):
    """Convert Ken French's downloaded CSVs into ff_factors.csv [date, mktrf, smb, hml, rmw, cma, umd, rf]."""
    def read_monthly(path):
        rows = []
        for line in open(path, encoding="latin-1"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 1 and len(parts[0]) == 6 and parts[0].isdigit():
                rows.append(parts)
            elif rows and parts[0] == "":
                break                     # monthly block ends; annual block follows
        return rows
    f5 = pd.DataFrame(read_monthly(five_factor_csv), columns=["ym", "mktrf", "smb", "hml", "rmw", "cma", "rf"])
    mo = pd.DataFrame(read_monthly(momentum_csv), columns=["ym", "umd"])
    df = f5.merge(mo, on="ym", how="left")
    df["date"] = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[["date", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]].to_csv(out, index=False)


def fred_to_csv(fred_csv, column="GS10", out=os.path.join(OUT, "risk_free.csv")):
    """Convert a FRED monthly CSV (e.g. GS10, percent) into risk_free.csv [date, r] in decimals."""
    df = pd.read_csv(fred_csv)
    df.columns = ["date", "r"] if len(df.columns) == 2 else df.columns
    df = df.rename(columns={column: "r", "observation_date": "date", "DATE": "date"})
    df["r"] = pd.to_numeric(df["r"], errors="coerce") / 100.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[["date", "r"]].dropna().to_csv(out, index=False)


if __name__ == "__main__":
    main()
