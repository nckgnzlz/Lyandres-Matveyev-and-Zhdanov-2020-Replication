"""Aggregate run_comparison.csv across simulated markets (seeds) into the headline table."""
import numpy as np
import pandas as pd

d = pd.read_csv("outputs/run_comparison.csv").set_index("run")


def pooled(runs, col_m, col_t):
    m = d.loc[runs, col_m].astype(float); t = d.loc[runs, col_t].astype(float)
    se = (m / t).abs()
    avg = m.mean(); se_avg = np.sqrt((se ** 2).sum()) / len(m)
    return avg, se_avg, list(m.round(2))


groups = {
    "Mechanics (true inputs)": [f"base_seed{s}_trueinputs" for s in (7, 8, 9, 10)],
    "Realistic (paper-style calibration)": [f"base_seed{s}" for s in (7, 8, 9, 10)],
    "Control (no mispricing)": [f"null_seed{s}" for s in (7, 8, 9, 10)],
}
rows = []
for name, runs in groups.items():
    for label, cm, ct in [("VW long-short", "UMO_VW_mean_pct", "UMO_VW_mean_t"),
                          ("VW long-short CAPM alpha", "UMO_VW_CAPM_pct", "UMO_VW_CAPM_t"),
                          ("EW long-short", "UMO_EW_mean_pct", "UMO_EW_mean_t"),
                          ("Long-only D1 CAPM alpha", "D1_longonly_CAPM_pct", "D1_longonly_CAPM_t"),
                          ("AP-only model VW long-short", "APonly_UMO_mean_pct", "APonly_UMO_mean_t"),
                          ("Low GO/AP tercile VW long-short", "lowGOAP_UMO_mean_pct", "lowGOAP_UMO_mean_t"),
                          ("High GO/AP tercile VW long-short", "highGOAP_UMO_mean_pct", "highGOAP_UMO_mean_t")]:
        a, se, per = pooled(runs, cm, ct)
        rows.append(dict(group=name, metric=label, avg_pct=round(a, 3), se=round(se, 3), t=round(a / se, 2), per_seed=per))
    for c in ["spearman_signal_vs_truth", "theta_corr_truth", "corr_signal_vs_indrel_EV_EBITDA", "turnover_D1", "turnover_D10",
              "D1_decile_after_1m", "D1_decile_after_12m", "D10_decile_after_1m", "D10_decile_after_12m", "oracle_UMO_mean_pct", "FM_t_logmisval"]:
        v = d.loc[runs, c].astype(float)
        rows.append(dict(group=name, metric=c, avg_pct=round(v.mean(), 3), per_seed=list(v.round(2))))
out = pd.DataFrame(rows)
out.to_csv("outputs/headline_results.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 60)
print(out.to_string())
sens = d.loc[[r for r in d.index if r.startswith("base_seed7_trueinputs")],
             ["UMO_VW_mean_pct", "UMO_VW_mean_t", "D1_longonly_CAPM_pct", "spearman_signal_vs_truth", "theta_corr_truth", "theta_hat_median", "eta_hat_median", "turnover_D1"]]
sens.to_csv("outputs/input_error_sensitivity.csv")
print(sens.round(2).to_string())
tot = d.loc[["total_seed7_trueinputs", "total_seed7"], ["UMO_VW_mean_pct", "UMO_VW_mean_t", "UMO_EW_mean_pct", "UMO_EW_mean_t", "lowGOAP_UMO_mean_pct", "lowGOAP_UMO_mean_t", "highGOAP_UMO_mean_pct", "highGOAP_UMO_mean_t", "FM_t_logmisval", "spearman_signal_vs_truth"]]
print(tot.round(2).T.to_string())
