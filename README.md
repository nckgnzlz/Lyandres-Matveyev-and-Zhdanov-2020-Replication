# Does the Market Correctly Value Investment Options? A Replication Using Bloomberg Data

*Replication of Lyandres, Matveyev and Zhdanov (2020), Review of Finance 24(6), 1159–1201.*

## Abstract

We replicate the real-options misvaluation strategy of Lyandres, Matveyev and Zhdanov (2020, hereafter LMZ). Each firm is valued as the sum of its assets in place and its growth options in a closed-form real-options model. Industry-level demand and technology parameters are calibrated from accounting and analyst data and estimated from recent market values. A long-short portfolio that buys the most undervalued decile and sells the most overvalued decile earns 0.90% per month in LMZ's sample (CAPM alpha 1.07%, t = 3.70). Before turning to real data, we validate our implementation on simulated markets with planted mispricing. Under the paper's calibration procedure, the strategy earns 0.88% per month (t = 4.33) when mispricing is present and 0.01% (t = 0.07) when it is absent. We then document a Bloomberg-based data pipeline that substitutes for LMZ's Compustat/CRSP/IBES sources, and verify that every required input is available through the Bloomberg Desktop API.

---

## 1. Introduction

LMZ ask whether equity markets correctly price firms' growth options. Their approach has two steps. First, a structural model gives each firm a fundamental value. Second, stocks are sorted on the ratio of market value to model value. If markets misprice growth options, firms that look cheap by the model should earn higher subsequent returns than firms that look expensive. The effect should also be concentrated among firms whose value comes mostly from growth options.

This repository provides:

1. a vectorised implementation of the LMZ valuation model (`lmz_model.py`);
2. the calibration, estimation and signal-construction pipeline (`lmz_pipeline.py`);
3. portfolio sorts and asset-pricing tests (`lmz_backtest.py`);
4. a simulation framework that generates markets from the model with known mispricing (`make_dummy_data.py`), so the implementation can be tested against ground truth; and
5. a Bloomberg data-acquisition layer (`bloomberg_pull_template.py`, `bloomberg_smoke_test.py`).

## 2. Model

A firm with capital stock $K$ faces a demand shock $x$ that follows a geometric Brownian motion with risk-neutral drift $\mu$ and volatility $\sigma$. Operating profit is $x K^{\theta}$, where $0<\theta<1$ governs returns to scale. Capital depreciates at rate $\lambda$, and installing one unit of capital costs $\eta$. Profits are taxed at rate $\tau = 35\%$, and the risk-free rate is $r$. We back out the demand shock from the data as

$$x = \frac{\text{EBITDA}}{K^{\theta}}.$$

**Option exponent.** $\beta_1$ is the positive root of (LMZ eq. 4)

$$\tfrac{1}{2}\sigma^2\beta(\beta-1) + \big[\mu - \lambda(\theta-1)\big]\beta - (r+\lambda) = 0.$$

**Investment boundary.** The firm invests when $x$ reaches (LMZ eq. 3)

$$X(K) = \frac{\beta_1}{\beta_1-1}\,\frac{(r-\mu+\lambda\theta)\,\eta\,K^{1-\theta}}{(1-\tau)\,\theta}.$$

**Firm value.** Firm value is the value of assets in place plus the value of growth options, $V = AP + GO$ (LMZ eqs. 5, 9, 10). Writing $d = r-\mu+\lambda\theta$,

$$AP = \frac{(1-\tau)\,x\,K^{\theta}}{d}, \qquad
GO = \frac{C\,x^{\beta_1}\,K^{-[\beta_1(1-\theta)-1]}}{\beta_1(1-\theta)-1}, \qquad
C = \left[\frac{(1-\tau)\theta}{\beta_1 d}\right]^{\beta_1}\left[\frac{\beta_1-1}{\eta}\right]^{\beta_1-1}.$$

The value of growth options is finite only if $\theta < 1 - 1/\beta_1$; this bound is enforced in estimation.

**Firms above the boundary.** For a firm observed with $x > X(K)$, LMZ assume it invests immediately up to the target capital $K^\ast$ that puts it exactly on the boundary (LMZ eq. 8). Its value is then $V(K^\ast, x) - \eta\,(K^\ast - K)$.

**Assets-in-place counterfactual.** As a benchmark, LMZ also use a model with growth options switched off, $V^{AP} = AP$ (LMZ eq. 14).

## 3. Data

### 3.1 Sources

LMZ use Compustat, CRSP and IBES. We substitute Bloomberg for all three. Factor returns and the risk-free rate come from public sources.

| Input | LMZ source | Our source | Bloomberg field | Frequency |
|---|---|---|---|---|
| Universe | CRSP common stocks | Historical Russell 3000 constituents | `INDX_MWEIGHT_HIST` on `RAY Index` (override `END_DATE_OVERRIDE`) | Year-end |
| Market equity | CRSP | Bloomberg | `CUR_MKT_CAP` | Monthly |
| Returns | CRSP | Bloomberg total return index | `TOT_RETURN_INDEX_GROSS_DVDS` | Monthly |
| Long-term growth forecast | IBES | Bloomberg BEst consensus | `BEST_EST_LONG_TERM_GROWTH` | Monthly |
| Capital stock $K$ | Compustat PPEGT | Gross PP&E | `BS_GROSS_FIX_ASSET` | Annual |
| Operating profit | Compustat SALE − COGS − XSGA | EBITDA | `EBITDA` | Annual |
| Depreciation | Compustat DP | D&A | `CF_DEPR_AMORT` | Annual |
| Book debt | Compustat DLC + DLTT | Short- + long-term borrowings | `BS_ST_BORROW`, `BS_LT_BORROW` | Annual |
| Sales | Compustat SALEQ | Revenue | `SALES_REV_TURN` | Quarterly |
| Industry | GICS | GICS (current) | `GICS_INDUSTRY`, `GICS_SECTOR` | Static |
| Factors | Ken French | Ken French data library | — | Monthly |
| Risk-free rate $r$ | Not specified | FRED GS10 (10-year Treasury) | — | Monthly |

### 3.2 Data availability

We verified every Bloomberg field by pulling one firm (Apple, `AAPL US Equity`) through the Desktop API (`bloomberg_smoke_test.py`, 18 October 2026):
- The Russell 3000 constituent list returned 2,952 members as of 31 December 2010, including since-delisted securities, so the universe is free of survivorship bias.
- Market data came back for 60 of 60 months (2010–2014), and long-term growth forecasts for 59 of 60.
- Annual fundamentals came back for 9 of 9 fiscal years (2006–2014), and quarterly sales for 36 of 36 quarters.
- Historical values are reported in USD millions.

For the full sample (2010–2024, about 7,000 unique securities, with 4 years of extra history for the fundamentals), we estimate roughly 5 million data points, about 1,400 API requests and about 160 MB of processed data. The binding constraint is Bloomberg's per-account usage limit, which Bloomberg does not publish.

### 3.3 Sample filters

Following LMZ, we exclude financials (GICS sector 40) and utilities (GICS sector 55) and require positive EBITDA. Industries are 6-digit GICS industries. On real data, accounting variables must be at least six months old, and analyst forecasts are taken from months $t-3$ to $t-1$ (LMZ Internet Appendix, Fig. IA.1).

## 4. Methodology

### 4.1 Calibration of industry inputs

For each industry $j$ and month $t$:

- **Demand drift.** We start from analysts' long-term growth forecasts, following the procedure in LMZ Internet Appendix E:
  1. Take the median forecast per firm, then the median across firms in the industry, $g$.
  2. Adjust it with $\tilde g = 0.007264 + 0.4086\, g$.
  3. Assume growth at $\tilde g$ for five years and 2% thereafter, and convert this path into one perpetual-growth equivalent at the industry-median CAPM cost of capital.
  4. Convert it to a risk-neutral drift with $\mu_{j,t} = g'_{j,t} - \beta^{A}_{j,t}\cdot \text{MRP}$, where $\text{MRP} = 6\%$. Asset betas are $\beta^{A} = \beta^{E}\cdot E/(E+D)$, with equity betas estimated over 36 months (minimum 24) and debt betas set to zero.
- **Demand volatility.** $\sigma_{j,t}$ is twice the standard deviation of quarterly changes in de-seasonalised log sales over the past eight quarters.
- **Depreciation.** $\lambda_{j,t}$ is computed from DP/PPEGT over the past three fiscal years.
- **Risk-free rate.** $r_t$ is the 10-year Treasury yield, or a constant 5% if no series is supplied.

### 4.2 Estimation of technology parameters

The parameters $(\theta, \eta)$ are not observable. For each industry and month, they are chosen to make the model match market values over the previous twelve months:

$$(\hat\theta_{j,t}, \hat\eta_{j,t}) = \arg\min_{\theta,\eta}\; \sum_{i\in j}\sum_{s=t-12}^{t-1} \left|\log\frac{ME_{i,s} + D_{i,s}}{V(K_{i,s}, x_{i,s};\theta,\eta)}\right|,$$

subject to $\theta < 1 - 1/\beta_1(\theta)$ and $1 \le \eta \le 5$. The minimisation is done on a 16 × 16 grid followed by local refinement. An industry-month needs at least 30 firm-month observations to be estimated.

### 4.3 Misvaluation and portfolio construction

Misvaluation is

$$M_{i,t} = \frac{ME_{i,t} + D_{i,t}}{V(K_{i,t}, x_{i,t};\hat\theta_{j,t},\hat\eta_{j,t})}.$$

At the end of each month, stocks are sorted into deciles of $M$. The strategy (UMO, "undervalued minus overvalued") is long decile 1 and short decile 10, value-weighted, and rebalanced monthly.

### 4.4 Tests

`lmz_backtest.py` reports:
- decile mean returns and alphas relative to the CAPM, FF3, FF3 + momentum, FF5 and FF5 + momentum models, with Newey-West standard errors;
- equal-weighted sorts;
- double sorts on the growth-options share $GO/AP$;
- the assets-in-place-only counterfactual;
- decile transition probabilities;
- Fama-MacBeth regressions of returns on $\log M$;
- portfolio turnover.

## 5. Validation on Simulated Markets

Because the true mispricing is unobservable in real data, we first test the implementation on markets simulated from the model itself (`make_dummy_data.py`). Each market spans 1986–2014 with about 1,000 firms. Fundamentals follow the model's dynamics, and market prices deviate from model values by a mean-reverting mispricing component that is placed on growth options. We generate four independent markets (seeds 7–10) and three scenarios:
- **true inputs:** the pipeline is given the true $\mu,\sigma,\lambda$;
- **realistic:** the inputs are calibrated as in Section 4.1;
- **control:** no mispricing is planted.

**Table 1. Value-weighted UMO returns (% per month), averaged over four simulated markets.**

| Scenario | Mean | t | CAPM α | High GO/AP tercile | Low GO/AP tercile |
|---|---|---|---|---|---|
| True inputs | 1.27 | 7.65 | 1.29 | 1.65 | 0.65 |
| Realistic calibration | 0.88 | 4.33 | 0.89 | 1.12 | 0.57 |
| Control (no mispricing) | 0.01 | 0.07 | 0.05 | 0.23 | −0.10 |
| *LMZ (2020), real data* | *0.90* | *1.91* | *1.07 (t = 3.70)* | *1.04*† | *0.42*† |

† LMZ Table X, CAPM alphas.

Five findings stand out:

1. The strategy recovers planted mispricing, and its magnitude under realistic calibration is close to the paper's real-data estimate.
2. When there is no mispricing, it produces no spurious returns. This also rules out look-ahead bias in the pipeline.
3. Consistent with LMZ, the effect is stronger among firms whose value comes mostly from growth options.
4. Decile transitions match the paper's. Decile-1 firms are in decile 1.84 on average one month later and 4.44 twelve months later; LMZ report 1.42 and 3.57.
5. The results are robust to deliberate input errors. Biasing $\mu$ by ±2 pp, $\lambda$ by +3 pp, $r$ by +1 pp, or scaling $\sigma$ by 0.8 keeps UMO between 0.82% and 1.03% per month.

Two limitations also stand out. Results vary across simulated markets: one of the four realistic-calibration markets earns only 0.15% per month. And $\theta$ is poorly identified under realistic calibration: the cross-sectional correlation between estimated and true $\theta$ is about zero.

## 6. Deviations from LMZ and Implementation Choices

1. **Data vendor.** Bloomberg's BEst long-term growth consensus differs from IBES. Bloomberg EBITDA differs slightly from SALE − COGS − XSGA. GICS codes are current, not point-in-time.
2. **Risk-free rate.** LMZ do not specify $r$; we use the 10-year Treasury yield.
3. **Sales volatility.** LMZ estimate seasonal dummies per industry; we estimate them per firm.
4. **Target capital.** The printed formula for $K^\ast$ (LMZ Section 3.2.a) uses $(r-\mu)$; the exact inverse of eq. (3) uses $(r-\mu+\lambda\theta)$. We use the exact inverse.
5. **Bounds on $\eta$.** LMZ do not state bounds. We impose $1 \le \eta \le 5$, consistent with several industries at exactly 1.00 in LMZ Table IA.4.
6. **Beta window.** LMZ use 36 months in the text and 60 months in the variable definitions; we use 36.
7. **Long-term growth conversion.** The printed eq. IA.27 appears to contain typos, and the appendix's worked example (10% → 4.15%) is inconsistent with its own formula (10% → 4.81%). Our implementation is in `terminal_equivalent_growth()`.
8. **Debt betas.** LMZ derive debt betas from credit spreads by distress quintile; we currently set them to zero.

## 7. Repository Structure

| File | Contents |
|---|---|
| `lmz_model.py` | Closed-form valuation model (Section 2) |
| `lmz_pipeline.py` | Panel construction, calibration (4.1), estimation (4.2), misvaluation (4.3). `CONFIG` holds the simulation settings; `REAL_DATA_CONFIG` holds the Internet Appendix settings. |
| `lmz_backtest.py` | Portfolio sorts and asset-pricing tests (4.4) |
| `make_dummy_data.py` | Simulated markets with planted mispricing (Section 5) |
| `run_test.py` | End-to-end runner for simulated or real data |
| `quick_test.py` | Twelve automated checks of the full system |
| `bloomberg_smoke_test.py` | Single-firm verification of every Bloomberg input (3.2) |
| `bloomberg_pull_template.py` | Full-universe Bloomberg pull, plus converters for the Ken French and FRED files |

## References

Lyandres, E., Matveyev, E., and Zhdanov, A. (2020). Does the market correctly value investment options? *Review of Finance*, 24(6), 1159–1201.

Fama, E. F., and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.

Newey, W. K., and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.

---

## Appendix A. Reproduction

### A.1 Environment

- **Python:** 3.9 or newer. Tested on Python 3.14 with numpy 2.4, pandas 3.0 and scipy 1.18. Anaconda (https://www.anaconda.com/download) includes everything the core code needs. On Windows, run all commands in the **Anaconda Prompt**, because plain `python` may open a Microsoft Store placeholder.

```
git clone https://github.com/nckgnzlz/LMZ_replication.git
cd LMZ_replication
pip install -r requirements.txt      # not needed with Anaconda
python quick_test.py                 # ~2 min; should end with "12/12 checks passed"
```

### A.2 Simulated markets (Section 5)

```
python run_test.py                       # realistic calibration, seed 7 (~5 min)
python run_test.py --true-inputs         # true mu, sigma, lambda
python run_test.py --scenario null       # control: no mispricing
python run_test.py --scenario total      # mispricing on the whole firm instead of only growth options
python run_test.py --perturb mu=+0.02    # deliberate input error
python run_test.py --seed 8              # another simulated market
```

Simulated data is written to `data/<scenario>_seed<N>/`. Results (`summary.json`, decile returns, alpha tables, $\hat\theta/\hat\eta$ and the firm-month misvaluation panel) go to `outputs/<run name>/`.

### A.3 Real data (Bloomberg)

**Requirements:**
- A Bloomberg Terminal that is installed and logged in on the same machine.
- The local API service `C:\blp\DAPI\bbcomm.exe` running (it listens on `localhost:8194`; start it by hand if the connection fails).
- The Bloomberg Python packages:

```
pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
pip install xbbg
```

**Steps:**

1. **Verify access:** `python bloomberg_smoke_test.py`. Every block should return data; `FIELD EXCEPTIONS` means the field isn't licensed or the name is wrong (check with `FLDS <GO>`).
2. **Test pull:** with `TEST_MODE = True` (50 firms, 2010–2014), run `python bloomberg_pull_template.py`. The output goes to `data/bloomberg/`, and the raw blocks are cached as `_raw_*.csv` so an interrupted pull resumes.
3. **Factors and risk-free rate:** download the monthly "Fama/French 5 Factors (2x3)" and "Momentum Factor" CSVs from the Ken French data library, and the GS10 series from FRED. Convert them with:
   ```
   python -c "import bloomberg_pull_template as b; b.ken_french_to_csv('F-F_Research_Data_5_Factors_2x3.csv', 'F-F_Momentum_Factor.csv'); b.fred_to_csv('GS10.csv')"
   ```
4. **Estimate:** `python run_test.py --data data/bloomberg --name real_test`. The Internet Appendix settings are applied automatically.
5. **Full sample:** widen `START`/`END`, set `TEST_MODE = False`, and repeat steps 2–4.

> **Licensing.** Bloomberg data may not be redistributed. `data/` and `outputs/` are excluded by `.gitignore`.

## Appendix B. Input File Format

`run_test.py --data <folder>` reads the files below. File names follow LMZ's databases regardless of the vendor. `gvkey` is a generic firm identifier, so a Bloomberg ticker works. Monetary amounts are in USD millions.

| File | Compustat-style columns | Bloomberg-style columns |
|---|---|---|
| `crsp_monthly.csv` | gvkey, date, prc, shrout (thousands), ret | gvkey, date, me, ret |
| `compustat_annual.csv` | gvkey, datadate, gind, gsector, ppegt, sale, cogs, xsga, dp, dlc, dltt, at | gvkey, datadate, gind, gsector, ppegt, ebitda, dp, debt |
| `compustat_quarterly.csv` | gvkey, datadate, fqtr, saleq | gvkey, datadate, saleq |
| `ibes_ltg.csv` | gvkey, statpers, meanltg (percent), numest | gvkey, statpers, meanltg |
| `ff_factors.csv` | date, mktrf, rf, and any of smb, hml, umd, rmw, cma (decimals) | same |
| `risk_free.csv` (optional) | date, r (annual decimal) | same |
