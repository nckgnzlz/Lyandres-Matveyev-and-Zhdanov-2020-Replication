# LMZ (2020) Strategy Replication

Replication code for **Lyandres, Matveyev & Zhdanov (2020), "Does the Market Correctly Value Investment Options?", *Review of Finance* 24(6)**.

The code runs end to end on synthetic data, which it generates itself, and on real data pulled from a Bloomberg Terminal.

## What the strategy does

Each month, every non-financial, non-utility firm with positive EBITDA is valued with a real-options model as **assets in place (AP) + growth options (GO)**. Industry inputs are calibrated from data:
- **Demand drift μ:** analysts' long-term growth minus beta × 6% market risk premium.
- **Demand volatility σ:** from quarterly sales.
- **Depreciation λ:** from D&A / PP&E.

Two technology parameters (returns to scale θ, cost of capital η) are estimated per industry each month by fitting the past 12 months of market values (minimising the sum of |log(market / model)|).

**Misvaluation = (market equity + book debt) / model value.** The strategy buys decile 1 (most undervalued), shorts decile 10 (most overvalued), value-weighted, rebalanced monthly.

## Files

| File | Purpose |
|---|---|
| `lmz_model.py` | Closed-form model: β₁ (eq. 4), investment boundary (eq. 3), AP/GO/V (eqs. 5, 9, 10), above-boundary adjustment (eq. 8), AP-only model (eq. 14) |
| `lmz_pipeline.py` | Builds the firm-month panel, calibrates μ/σ/λ, estimates θ/η, computes misvaluation |
| `lmz_backtest.py` | Decile sorts, Newey-West alphas, transition table, GO/AP double sort, Fama-MacBeth, turnover |
| `make_dummy_data.py` | Simulates a market from the model with known, planted mispricing |
| `run_test.py` | End-to-end runner (synthetic or real data) |
| `quick_test.py` | One-command check of the whole kit (12 PASS/FAIL checks) |
| `bloomberg_smoke_test.py` | Pulls one sample of every Bloomberg input for AAPL to confirm API access and field names |
| `bloomberg_pull_template.py` | Full-universe Bloomberg pull into the pipeline's file format, plus converters for the Fama-French and FRED files |
| `requirements.txt` | Python packages for the core code |

---

## Setup on a new machine

### 1. Install Python

Install **Anaconda** (https://www.anaconda.com/download). It includes Python and every package the core code needs. The code needs Python 3.9 or newer; it was tested on Python 3.14 with numpy 2.4, pandas 3.0 and scipy 1.18.

On Windows, run every command below in the **Anaconda Prompt** (Start menu → Anaconda Prompt), not plain PowerShell or cmd. On many Windows machines, plain `python` opens a Microsoft Store placeholder instead of real Python.

### 2. Get the code

```
git clone https://github.com/nckgnzlz/LMZ_replication.git
cd LMZ_replication
```

### 3. Install the packages

If you use Anaconda, the packages are already installed. Otherwise:

```
pip install -r requirements.txt
```

### 4. Check that everything works

```
python quick_test.py
```

This takes about 2 minutes and should end with `12/12 checks passed`. It checks the model math against the paper, generates two small synthetic markets, runs the full strategy on both, and runs the real-data code path on Bloomberg-style files. It does **not** need Bloomberg.

---

## Running on synthetic data (no Bloomberg needed)

```
python run_test.py                       # mispriced market, paper-style calibration (≈5 min)
python run_test.py --true-inputs         # same market, true mu/sigma/lambda (tests the machinery alone)
python run_test.py --scenario null       # no mispricing: the strategy should find nothing
python run_test.py --scenario total      # whole firm mispriced instead of only growth options
python run_test.py --perturb mu=+0.02    # deliberately bias a calibrated input
python run_test.py --seed 8              # a different simulated market
```

Synthetic data is generated into `data/<scenario>_seed<N>/` on first use. Each run writes `summary.json`, decile returns, alpha tables, the estimated θ/η and the firm-month misvaluation panel to `outputs/<run name>/`.

---

## Running on real data (Bloomberg)

### Requirements

- A PC with a **Bloomberg Terminal** that is installed and **logged in**. The Desktop API only works on the machine running the Terminal.
- The Bloomberg API service `bbcomm.exe`, usually at `C:\blp\DAPI\bbcomm.exe`, must be running. It listens on `localhost:8194`. It normally starts with the Terminal. If the scripts can't connect, double-click it.
- The Bloomberg Python packages, installed in the Anaconda Prompt:

```
pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
pip install xbbg
```

### Step 1: Confirm API access

```
python bloomberg_smoke_test.py
```

This pulls one sample of every input for Apple. It should print:
- about 3,000 Russell 3000 members for 2010-12-31
- 60 monthly rows of market cap and total return index
- monthly long-term growth history
- 9 annual rows of fundamentals
- 36 quarterly rows of sales
- GICS codes

If you see `FIELD EXCEPTIONS` or `SECURITY ERROR`, that field isn't available to your account or its name is wrong. Look it up with `FLDS <GO>` on the Terminal.

### Step 2: Test pull (50 firms)

In `bloomberg_pull_template.py`, leave `TEST_MODE = True` (50 firms, 2010–2014), then run:

```
python bloomberg_pull_template.py
```

The files are written to `data/bloomberg/`. Each block is cached as `_raw_*.csv`, so a stopped run resumes where it left off. Delete the `_raw_*.csv` files to force a fresh pull.

### Step 3: Add the two free files

1. **Fama-French factors:** from Ken French's data library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), download the "Fama/French 5 Factors (2x3)" and "Momentum Factor" CSVs (monthly).
2. **Risk-free rate:** from FRED (https://fred.stlouisfed.org/series/GS10), download the monthly GS10 CSV (10-year Treasury yield).
3. Convert both files:

```
python -c "import bloomberg_pull_template as b; b.ken_french_to_csv('F-F_Research_Data_5_Factors_2x3.csv', 'F-F_Momentum_Factor.csv'); b.fred_to_csv('GS10.csv')"
```

If you skip `risk_free.csv`, the code uses a constant r = 5%.

### Step 4: Run the strategy

```
python run_test.py --data data/bloomberg --name real_test
```

Once the test pull looks right, widen `START`/`END` in `bloomberg_pull_template.py`, set `TEST_MODE = False`, and pull again.

### Expected size of the full pull

| | 2010–2024, about 7,000 unique tickers (estimate) |
|---|---|
| Data points | ~5 million |
| API requests (25 tickers each) | ~1,400 |
| Disk | ~160 MB of CSVs (~300–350 MB with the raw caches) |
| Time | ~1 hour, if there is no throttling |

Bloomberg caps Desktop API usage per account and doesn't publish the limits. Pull in stages if needed; the cache makes that easy.

> **Licensing:** Bloomberg data may not be redistributed. `data/` is in `.gitignore`. Never commit or share pulled files.

---

## Input file format

`run_test.py --data <folder>` reads these files. File names follow the paper's databases (Compustat, CRSP, IBES) even when the data comes from Bloomberg. `gvkey` is just the firm identifier column, so a Bloomberg ticker such as `AAPL US Equity` works. Money amounts are in USD millions.

| File | Compustat-style columns | Bloomberg-style columns |
|---|---|---|
| `crsp_monthly.csv` | gvkey, date, prc, shrout (thousands), ret | gvkey, date, me (market cap), ret |
| `compustat_annual.csv` | gvkey, datadate, gind, gsector, ppegt, sale, cogs, xsga, dp, dlc, dltt, at | gvkey, datadate, gind, gsector, ppegt, ebitda, dp, debt |
| `compustat_quarterly.csv` | gvkey, datadate, fqtr, saleq | gvkey, datadate, saleq |
| `ibes_ltg.csv` | gvkey, statpers, meanltg (percent), numest | gvkey, statpers, meanltg |
| `ff_factors.csv` | date, mktrf, rf, and any of smb, hml, umd, rmw, cma (decimals) | same |
| `risk_free.csv` (optional) | date, r (annual decimal) | same |

Bloomberg fields used:

| Input | Field | Frequency |
|---|---|---|
| Universe (historical Russell 3000 members) | `INDX_MWEIGHT_HIST` on `RAY Index`, override `END_DATE_OVERRIDE` | Year-end |
| Market cap | `CUR_MKT_CAP` | Monthly |
| Total return index | `TOT_RETURN_INDEX_GROSS_DVDS` | Monthly |
| Analyst long-term growth | `BEST_EST_LONG_TERM_GROWTH` | Monthly |
| Gross PP&E | `BS_GROSS_FIX_ASSET` | Annual |
| EBITDA | `EBITDA` | Annual |
| D&A | `CF_DEPR_AMORT` | Annual |
| Short-term / long-term debt | `BS_ST_BORROW`, `BS_LT_BORROW` | Annual |
| Sales | `SALES_REV_TURN` | Quarterly |
| GICS industry / sector | `GICS_INDUSTRY`, `GICS_SECTOR` | Static (current) |

Historical (BDH) pulls return market cap and fundamentals in USD millions, which matches the format above.

---

## Settings: synthetic vs. real data

`lmz_pipeline.CONFIG` is used for synthetic data. `lmz_pipeline.REAL_DATA_CONFIG`, used automatically with `--data`, follows the paper's Internet Appendix (SSRN version):
- Accounting data at least 6 months old; analyst forecasts from months t-3 to t-1 (Fig. IA.1).
- LTG procedure (Appendix E): median LTG per firm, then the median across firms in the industry. Adjusted LTG = 0.007264 + 0.4086 × median LTG, applied for 5 years, then 2% forever. This is converted to one perpetual rate at the industry-median cost of capital, then beta × 6% is subtracted. The printed eq. IA.27 appears to contain typos; see `terminal_equivalent_growth()`. The appendix's worked example (10% → 4.15%) also doesn't match its own formula (10% → 4.81%).

The synthetic tests keep a 4-month lag and raw LTG, because the simulated analysts are unbiased.

## Choices the paper leaves open

- **Risk-free rate r:** a constant 5% by default; on real data, supply `risk_free.csv` (e.g., the 10-year Treasury yield).
- **Sales-volatility formula:** std of quarterly changes in de-seasonalised log sales over 8 quarters, ×2. The paper estimates seasonal dummies per industry; we do it per firm.
- **K\* formula:** the printed formula uses (r − μ); the exact inverse of eq. (3) uses (r − μ + λθ). We use the exact inverse.
- **Bounds on η:** 1 ≤ η ≤ 5. The paper doesn't state them, but Table IA.4 has several industries at exactly 1.00, consistent with a floor of 1.
- **Beta window:** 36 months in the text, 60 in the variable table. We use 36.

## Known data caveats (Bloomberg vs. the paper)

- Bloomberg's long-term growth consensus (BEst) is not IBES.
- Bloomberg's EBITDA definition differs slightly from sales − COGS − SG&A.
- GICS codes are current, not historical, so firms that changed industries may be misclassified in early years.
- Use historical index members (the template does) so delisted firms are included.
