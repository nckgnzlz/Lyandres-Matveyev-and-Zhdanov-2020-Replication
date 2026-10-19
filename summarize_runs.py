"""Collect outputs/*/summary.json into one comparison table (outputs/run_comparison.csv)."""
import glob
import json
import os
import pandas as pd


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return d if d is not None else default


def main(root="outputs"):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        s = json.load(open(f))
        a, t = s["vw_alpha_pct"], s["vw_t"]
        row = {
            "run": s["run"],
            "UMO_VW_mean_pct": g(a, "Mean", "1-10"), "UMO_VW_mean_t": g(t, "Mean", "1-10"),
            "UMO_VW_CAPM_pct": g(a, "CAPM", "1-10"), "UMO_VW_CAPM_t": g(t, "CAPM", "1-10"),
            "UMO_VW_FF3_pct": g(a, "FF3", "1-10"), "UMO_VW_FF3_t": g(t, "FF3", "1-10"),
            "UMO_VW_FF3MOM_pct": g(a, "FF3+MOM", "1-10"), "UMO_VW_FF3MOM_t": g(t, "FF3+MOM", "1-10"),
            "D1_longonly_CAPM_pct": g(a, "CAPM", "1"), "D1_longonly_CAPM_t": g(t, "CAPM", "1"),
            "D10_CAPM_pct": g(a, "CAPM", "10"),
            "UMO_EW_mean_pct": g(s, "ew_spread", "alpha_pct", "Mean"), "UMO_EW_mean_t": g(s, "ew_spread", "t", "Mean"),
            "lowGOAP_UMO_mean_pct": g(s, "double_sort", "GOAP_tercile_1", "Mean", default=[None])[0],
            "lowGOAP_UMO_mean_t": g(s, "double_sort", "GOAP_tercile_1", "Mean", default=[None, None])[1],
            "highGOAP_UMO_mean_pct": g(s, "double_sort", "GOAP_tercile_3", "Mean", default=[None])[0],
            "highGOAP_UMO_mean_t": g(s, "double_sort", "GOAP_tercile_3", "Mean", default=[None, None])[1],
            "APonly_UMO_mean_pct": g(s, "ap_only_spread", "Mean", default=[None])[0],
            "APonly_UMO_mean_t": g(s, "ap_only_spread", "Mean", default=[None, None])[1],
            "oracle_UMO_mean_pct": g(s, "oracle_spread", "Mean", default=[None])[0],
            "spearman_signal_vs_truth": g(s, "signal_quality", "spearman_pooled_avg"),
            "spearman_within_industry": g(s, "signal_quality", "spearman_within_industry_avg"),
            "D1_share_truly_underv": g(s, "signal_quality", "share_D1_truly_undervalued_bottom3"),
            "D10_share_truly_overv": g(s, "signal_quality", "share_D10_truly_overvalued_top3"),
            "theta_corr_truth": g(s, "signal_quality", "theta_corr_with_truth"),
            "eta_corr_truth": g(s, "signal_quality", "eta_corr_with_truth"),
            "goap_rankcorr_truth": g(s, "signal_quality", "goap_rank_corr_model_vs_true"),
            "theta_hat_median": g(s, "estimates_summary", "theta_hat_median"),
            "eta_hat_median": g(s, "estimates_summary", "eta_hat_median"),
            "goap_median": g(s, "estimates_summary", "goap_median"),
            "corr_signal_vs_indrel_EV_EBITDA": s.get("spearman_signal_vs_indrel_ev_ebitda"),
            "corr_signal_vs_APonly": s.get("corr_full_vs_ap_only_signal"),
            "turnover_D1": s.get("turnover_D1"), "turnover_D10": s.get("turnover_D10"),
            "D1_decile_after_1m": g(s, "transition", "1m", "1"), "D1_decile_after_12m": g(s, "transition", "12m", "1"),
            "D10_decile_after_1m": g(s, "transition", "1m", "10"), "D10_decile_after_12m": g(s, "transition", "12m", "10"),
            "FM_coef_logmisval": s["fama_macbeth"][1]["coef"] if s.get("fama_macbeth") else None,
            "FM_t_logmisval": s["fama_macbeth"][1]["t"] if s.get("fama_macbeth") else None,
            "months": s.get("first_month", "") + " to " + s.get("last_month", ""),
            "firms_per_decile": s.get("avg_firms_per_decile"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "run_comparison.csv"), index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(main().set_index("run").T.to_string())
