"""
wrds_pull_template.py   (UNTESTED TEMPLATE - needs a WRDS account)
--------------------------------------------------------------------
Pulls the real-data inputs for the LMZ (2020) strategy from WRDS and writes
them with the same file/column names that make_dummy_data.py produces, so

    python run_test.py --data data/real --name real_data

runs the identical pipeline on real firms.

Requirements:  pip install wrds   and a WRDS login whose subscription covers
Compustat (North America), CRSP, CCM link table, and IBES.  Check with your
library before relying on IBES - it's the input most often missing from
university subscriptions.

Things this template does NOT do yet (decide as a group):
  * CRSP delisting returns (crsp.msedelist) - should be merged into ret
  * historical GICS before 1994 (comp.co_hgic starts mid-1990s; the paper's
    sample begins in 1980, so they must backfill or use current GICS)
  * debt beta by distance-to-default quintile (paper Sec. 3.1)
  * risk-free rate series for r (pass to lmz_pipeline.calibrate(r_series=...))
"""
import os
import pandas as pd

START, END = "1977-01-01", "2024-12-31"
OUT = os.path.join("data", "real")


def main():
    import wrds
    db = wrds.Connection()          # prompts for username/password the first time
    os.makedirs(OUT, exist_ok=True)

    # ---------- Compustat annual ----------
    funda = db.raw_sql(f"""
        select f.gvkey, f.fyear, f.datadate, f.at, f.ppegt, f.sale, f.cogs, f.xsga,
               f.dp, f.dlc, f.dltt, f.capx
        from comp.funda f
        where f.indfmt='INDL' and f.datafmt='STD' and f.popsrc='D' and f.consol='C'
          and f.datadate between '{START}' and '{END}'
    """, date_cols=["datadate"])
    # GICS: historical where available, otherwise current
    hg = db.raw_sql("select gvkey, indfrom, indthru, gind, gsector from comp.co_hgic",
                    date_cols=["indfrom", "indthru"])
    cur = db.raw_sql("select gvkey, gind, gsector from comp.company")
    funda = funda.merge(hg, on="gvkey", how="left")
    ok = (funda.datadate >= funda.indfrom) & ((funda.datadate <= funda.indthru) | funda.indthru.isna())
    funda = pd.concat([funda[ok], funda[funda.indfrom.isna()]]).drop(columns=["indfrom", "indthru"])
    funda = funda.drop_duplicates(["gvkey", "datadate"])
    miss = funda.gind.isna()
    funda.loc[miss, ["gind", "gsector"]] = funda.loc[miss, ["gvkey"]].merge(cur, on="gvkey", how="left")[["gind", "gsector"]].values
    funda.to_csv(os.path.join(OUT, "compustat_annual.csv"), index=False)

    # ---------- Compustat quarterly sales ----------
    fundq = db.raw_sql(f"""
        select gvkey, datadate, fyearq, fqtr, saleq
        from comp.fundq
        where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
          and datadate between '{START}' and '{END}' and saleq > 0
    """, date_cols=["datadate"])
    fundq.to_csv(os.path.join(OUT, "compustat_quarterly.csv"), index=False)

    # ---------- CRSP monthly + CCM link ----------
    msf = db.raw_sql(f"""
        select a.permno, a.date, a.prc, a.shrout, a.ret
        from crsp.msf a join crsp.msenames b
          on a.permno=b.permno and a.date between b.namedt and b.nameendt
        where a.date between '{START}' and '{END}'
          and b.shrcd in (10, 11) and b.exchcd in (1, 2, 3)
    """, date_cols=["date"])
    link = db.raw_sql("""
        select gvkey, lpermno as permno, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LU','LC') and linkprim in ('P','C')
    """, date_cols=["linkdt", "linkenddt"])
    m = msf.merge(link, on="permno")
    m = m[(m.date >= m.linkdt) & ((m.date <= m.linkenddt) | m.linkenddt.isna())]
    m = m.drop(columns=["linkdt", "linkenddt"]).drop_duplicates(["permno", "date"])
    m.to_csv(os.path.join(OUT, "crsp_monthly.csv"), index=False)

    # ---------- IBES long-term growth consensus ----------
    ltg = db.raw_sql(f"""
        select ticker, statpers, meanest as meanltg, medest as medltg, numest
        from ibes.statsum_epsus
        where fpi='0' and measure='EPS' and statpers between '{START}' and '{END}'
    """, date_cols=["statpers"])
    iclink = db.raw_sql("select ticker, permno, sdate, edate from wrdsapps.ibcrsphist where score <= 2",
                        date_cols=["sdate", "edate"])
    ltg = ltg.merge(iclink, on="ticker")
    ltg = ltg[(ltg.statpers >= ltg.sdate) & (ltg.statpers <= ltg.edate)]
    ltg = ltg.merge(link[["gvkey", "permno"]].drop_duplicates("permno"), on="permno")
    # Internet Appendix E uses analyst-level forecasts (latest per analyst, median per firm).
    # The summary file's median (medest) is a close stand-in; ibes.det_epsus with fpi='0' matches exactly.
    ltg[["gvkey", "statpers", "meanltg", "medltg", "numest"]].to_csv(os.path.join(OUT, "ibes_ltg.csv"), index=False)

    # ---------- Fama-French factors (+ momentum) ----------
    ff = db.raw_sql(f"select date, mktrf, smb, hml, rmw, cma, umd, rf from ff.fivefactors_monthly "
                    f"where date between '{START}' and '{END}'", date_cols=["date"])
    ff.to_csv(os.path.join(OUT, "ff_factors.csv"), index=False)
    print("wrote", os.listdir(OUT))


if __name__ == "__main__":
    main()
