# LMZ (2020) strategy – feasibility test kit

Replication scaffold for **Lyandres, Matveyev & Zhdanov (2020), "Does the Market
Correctly Value Investment Options?", *Review of Finance* 24(6)**, built for a
first-pass feasibility test with synthetic ("dummy") data. The same code is
meant to run on real data pulled from Bloomberg later.

**Quick check that everything works:** open an Anaconda Prompt in this folder and run

    python quick_test.py

It takes about 1.5 minutes and prints 12 PASS/FAIL checks (model math, fresh dummy data,
the full strategy on a mispriced and a no-mispricing market, and the Bloomberg file format).

## What the strategy does (one paragraph)
Each month, for every non-financial, non-utility firm with positive EBITDA, a
real-options model values the firm as **assets in place (AP) + growth options (GO)**.
Industry-level inputs are calibrated from data (demand drift μ from analysts'
long-term growth minus beta × 6% MRP; demand volatility σ from quarterly sales;
depreciation λ from DP/PP&E). Two technology parameters (θ = returns to scale,
η = cost of capital) are estimated per industry each month by making the model
fit the past 12 months of market values as closely as possible (minimise the sum of
|log(market / model)|). A firm's **misvaluation = (market equity + book debt) /
model value**. Sort all firms into deciles: buy decile 1 (most undervalued), short
decile 10 (most overvalued), value-weighted, rebalance monthly.

Under the hood the model value reduces to

    V = (1-τ)·EBITDA / (r − μ + λθ)  +  C(θ,η,…) · PP&E · (EBITDA / PP&E)^β₁ / (β₁(1−θ)−1)

i.e. a capitalised-EBITDA term plus a convex growth-option term in profitability.

## Files
| File | Purpose |
|---|---|
| `lmz_model.py` | Closed-form model: β₁ (eq. 4), investment boundary (eq. 3), AP/GO/V (eq. 5, 9, 10), above-boundary adjustment (eq. 8), AP-only model (eq. 14) |
| `make_dummy_data.py` | Simulates a synthetic market from the model with known parameters and injected, mean-reverting mispricing of growth options. Writes Compustat/CRSP/IBES-style CSVs + an answer key (`truth_*.csv`) |
| `lmz_pipeline.py` | Builds firm-month panel, calibrates μ/σ/λ, estimates θ/η (rolling 12-month L1 fit), computes misvaluation, AP-only counterfactual |
| `lmz_backtest.py` | Decile sorts, Newey-West alphas, transition table, GO/AP double sort, Fama-MacBeth, turnover |
| `run_test.py` | End-to-end runner (synthetic or real data). Results in `outputs/<run>/` |
| `quick_test.py` | One-command check of the whole kit (~1.5 min, 12 PASS/FAIL checks) |
| `bloomberg_pull_template.py` | **Untested** script to pull the real inputs from a Bloomberg Terminal into the pipeline's file format, plus helpers for the free Fama-French and FRED files |
| `wrds_pull_template.py` | **Untested**; only useful if the group ever gets WRDS access |
| `summarize_runs.py`, `aggregate_results.py` | Collect every run's `summary.json` into `outputs/run_comparison.csv` and the cross-market averages in `outputs/headline_results.csv` |
| `make_charts.py` | Slide charts in `outputs/charts/` |
| `paper_benchmarks.csv` | The paper's own numbers (Tables V, VI, VII, X, XI) for comparison |
| `data_sample/` | 60 firms from the base dummy market, for browsing in Excel. Full datasets regenerate in ~20 s |
| `outputs/<run>/` | Per-run summary, decile returns and alpha tables for the 20 test runs (large firm-month panels left out) |

## Running it
Requires Python 3.9+ with numpy, pandas, scipy, matplotlib (Anaconda has all of them).

    python run_test.py                       # base synthetic world, paper-style calibration (≈5 min)
    python run_test.py --true-inputs         # same world, true mu/sigma/lambda (tests the machinery alone)
    python run_test.py --scenario null       # no mispricing – the strategy should find nothing
    python run_test.py --scenario total      # whole firm mispriced instead of only GO
    python run_test.py --perturb mu=+0.02    # deliberately bias a calibrated input
    python run_test.py --data data/bloomberg # real data (after bloomberg_pull_template.py)

Each run writes `summary.json`, decile returns, alpha tables, the estimated θ/η,
and the full firm-month misvaluation panel to `outputs/<run name>/`.

## Input file format
File names follow the paper's databases (Compustat, CRSP, IBES) even when the data comes
from Bloomberg. `gvkey` is just the firm identifier column: a Bloomberg ticker such as
"AAPL US Equity" works. Money amounts in USD millions.

| File | Columns (dummy data / Compustat style) | Bloomberg-style alternative |
|---|---|---|
| `crsp_monthly.csv` | gvkey, date, prc, shrout (thousands), ret | gvkey, date, me (market cap), ret |
| `compustat_annual.csv` | gvkey, datadate, gind, gsector, ppegt, sale, cogs, xsga, dp, dlc, dltt, at | gvkey, datadate, gind, gsector, ppegt, ebitda, dp, debt |
| `compustat_quarterly.csv` | gvkey, datadate, fqtr, saleq | gvkey, datadate, saleq |
| `ibes_ltg.csv` | gvkey, statpers, meanltg (percent), numest | gvkey, statpers, meanltg |
| `ff_factors.csv` | date, mktrf, rf, and any of smb, hml, umd, rmw, cma (decimals) | same (free from Ken French) |
| `risk_free.csv` (optional) | date, r (annual decimal); otherwise r = 5% | same (free from FRED, e.g. GS10) |

## Getting real data from Bloomberg
1. On the terminal PC: `pip install xbbg` and Bloomberg's `blpapi` package (command in the script's header).
2. Confirm every field name in `bloomberg_pull_template.py` with FLDS on the terminal, and check units.
3. Run it with `TEST_MODE = True` (50 firms, 2010-2014) and look at the files in `data/bloomberg/`.
4. Download Ken French's 5-factor and momentum CSVs and FRED's GS10 CSV; convert them with
   `ken_french_to_csv()` and `fred_to_csv()` from the same script.
5. `python run_test.py --data data/bloomberg --name real_test`, then widen the dates and turn TEST_MODE off.

Watch for: Bloomberg's long-term growth consensus is not IBES; use historical index members
(the script uses the Russell 3000) so delisted firms are included; download caps (pull in batches);
Bloomberg's EBITDA definition differs slightly from sales − COGS − SG&A; GICS codes are current, not historical.

## Settings: dummy vs. real data
`lmz_pipeline.CONFIG` is used for the dummy tests. `lmz_pipeline.REAL_DATA_CONFIG` (used automatically
with `--data`) follows the paper's Internet Appendix (in the SSRN version of the paper):
- Accounting data at least 6 months old; analyst forecasts from months t-3 to t-1 (Fig. IA.1).
- LTG procedure (Appendix E): median LTG per firm, median across firms in the industry,
  Adjusted LTG = 0.007264 + 0.4086 × median LTG, then 5 years at the adjusted rate and 2% forever,
  converted to one perpetual rate at the industry-median cost of capital, then minus beta × 6%.
  The printed eq. IA.27 appears to contain typos; see `terminal_equivalent_growth()`.
  The appendix's worked example (10% → 4.15%) also doesn't match its own formula (10% → 4.81%).
The dummy tests keep a 4-month lag and raw LTG because the simulated analysts are unbiased.

## Choices the paper still leaves open
- Which risk-free rate is `r` (we use a constant 5%; on real data use e.g. the 10-year Treasury yield).
- Exact sales-volatility formula (we use std of quarterly changes in de-seasonalised log sales over 8 quarters, ×2; the paper estimates the seasonal dummies per industry, we do it per firm).
- The printed K* formula uses (r − μ); the exact inverse of eq. (3) uses (r − μ + λθ). We use the exact inverse.
- Bounds on η: we use 1 ≤ η ≤ 5. Not stated, but Table IA.4 has several industries at exactly 1.00 (e.g., Software), consistent with a floor of 1. Without a floor η is poorly identified.
- Beta window: 36 months in the text, 60 in the variable table (we use 36).

## Run names in outputs
`base_seed7` = mispriced market, paper-style calibration · `base_seed7_trueinputs` = same market, true inputs ·
`null_seed7` = no mispricing · `total_seed7` = whole firm mispriced · `..._mu+0.02` etc. = deliberate input errors.
Seeds 7–10 are four independent simulated markets.
