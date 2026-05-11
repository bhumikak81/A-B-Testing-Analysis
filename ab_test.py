"""
ab_test.py
----------
Reusable helper functions for A/B test analysis.
Covers data cleaning, EDA, hypothesis testing, and reporting.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Styling ──────────────────────────────────────────────────────────────────

COLORS = {"control": "#6B7280", "treatment": "#3B82F6"}
PALETTE = ["#6B7280", "#3B82F6"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "font.size": 11,
})


# ─── 1. Data Loading & Cleaning ───────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and parse timestamps."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw A/B test dataset:
      - Remove duplicate user_id entries
      - Drop users assigned to multiple groups
      - Drop mismatched group/landing_page combos
    Returns a clean DataFrame and prints a summary.
    """
    original = len(df)

    # 1. Remove duplicate rows
    df = df.drop_duplicates(subset="user_id", keep="first")
    after_dedup = len(df)

    # 2. Keep only users in exactly one group
    group_counts = df.groupby("user_id")["group"].nunique()
    valid_users = group_counts[group_counts == 1].index
    df = df[df["user_id"].isin(valid_users)]
    after_single_group = len(df)

    # 3. Drop mismatched page assignments
    valid_mask = (
        ((df["group"] == "control") & (df["landing_page"] == "old_page")) |
        ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
    )
    df = df[valid_mask].reset_index(drop=True)
    final = len(df)

    print("── Data Cleaning Summary ──────────────────────")
    print(f"  Original rows       : {original:>10,}")
    print(f"  After dedup         : {after_dedup:>10,}  (-{original - after_dedup:,})")
    print(f"  After single-group  : {after_single_group:>10,}  (-{after_dedup - after_single_group:,})")
    print(f"  After page fix      : {final:>10,}  (-{after_single_group - final:,})")
    print("───────────────────────────────────────────────")
    return df


def check_sample_ratio_mismatch(df: pd.DataFrame, tolerance: float = 0.01) -> bool:
    """
    Check if the split between control and treatment is roughly 50/50.
    Raises a warning if the ratio deviates beyond tolerance.
    """
    counts = df["group"].value_counts()
    ratio = counts.min() / counts.max()
    srm_ok = ratio >= (1 - tolerance)
    print(f"\nSample Ratio Check")
    print(f"  Control   : {counts.get('control', 0):,}")
    print(f"  Treatment : {counts.get('treatment', 0):,}")
    print(f"  Ratio     : {ratio:.4f}  ({'OK' if srm_ok else 'WARNING: SRM detected'})")
    return srm_ok


# ─── 2. EDA ───────────────────────────────────────────────────────────────────

def conversion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of conversion stats per group."""
    summary = df.groupby("group")["converted"].agg(
        users="count",
        conversions="sum",
        conversion_rate="mean"
    ).reset_index()
    summary["conversion_rate_pct"] = (summary["conversion_rate"] * 100).round(2)
    return summary


def plot_conversion_rates(df: pd.DataFrame, save_path: str = None):
    """Bar chart comparing conversion rates between groups."""
    summary = conversion_summary(df)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        summary["group"],
        summary["conversion_rate_pct"],
        color=[COLORS[g] for g in summary["group"]],
        width=0.45,
        zorder=3
    )

    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{row.conversion_rate_pct:.2f}%\n({row.conversions:,} / {row.users:,})",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title("Conversion Rate by Group", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Group")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_ylim(0, summary["conversion_rate_pct"].max() * 1.25)

    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in COLORS.items()]
    ax.legend(handles=patches, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_daily_conversions(df: pd.DataFrame, save_path: str = None):
    """Line chart of daily conversion rates over time per group."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby(["date", "group"])["converted"].mean().reset_index()
    daily["conversion_rate_pct"] = daily["converted"] * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for group, color in COLORS.items():
        grp = daily[daily["group"] == group]
        ax.plot(grp["date"], grp["conversion_rate_pct"], label=group.capitalize(),
                color=color, linewidth=2, marker="o", markersize=3)

    ax.set_title("Daily Conversion Rates Over Time", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Conversion Rate (%)")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ─── 3. Hypothesis Testing ────────────────────────────────────────────────────

def run_z_test(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    One-tailed two-proportion z-test.
    H0: p_treatment <= p_control
    H1: p_treatment >  p_control

    Returns a results dict with all key statistics.
    """
    control = df[df["group"] == "control"]["converted"]
    treatment = df[df["group"] == "treatment"]["converted"]

    n_c, n_t = len(control), len(treatment)
    p_c, p_t = control.mean(), treatment.mean()
    conversions_c = control.sum()
    conversions_t = treatment.sum()

    # Pooled proportion
    p_pool = (conversions_c + conversions_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
    z_stat = (p_t - p_c) / se
    p_value = 1 - stats.norm.cdf(z_stat)   # one-tailed

    # 95% confidence interval on the difference
    se_diff = np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t)
    margin = 1.96 * se_diff
    ci_lower = (p_t - p_c) - margin
    ci_upper = (p_t - p_c) + margin

    significant = p_value < alpha
    relative_uplift = (p_t - p_c) / p_c * 100

    results = {
        "n_control": n_c, "n_treatment": n_t,
        "conversions_control": conversions_c,
        "conversions_treatment": conversions_t,
        "p_control": p_c, "p_treatment": p_t,
        "absolute_diff": p_t - p_c,
        "relative_uplift_pct": relative_uplift,
        "pooled_proportion": p_pool,
        "z_statistic": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "alpha": alpha,
        "significant": significant,
    }
    return results


def print_results(results: dict):
    """Pretty-print hypothesis test results."""
    verdict = "REJECT H₀ — statistically significant" if results["significant"] \
              else "FAIL TO REJECT H₀ — not statistically significant"

    print("\n══════════════════════════════════════════════")
    print("  A/B TEST RESULTS")
    print("══════════════════════════════════════════════")
    print(f"  Control conversion rate   : {results['p_control']*100:.3f}%  (n={results['n_control']:,})")
    print(f"  Treatment conversion rate : {results['p_treatment']*100:.3f}%  (n={results['n_treatment']:,})")
    print(f"  Absolute difference       : {results['absolute_diff']*100:+.3f} pp")
    print(f"  Relative uplift           : {results['relative_uplift_pct']:+.2f}%")
    print(f"  95% CI on difference      : [{results['ci_lower']*100:.3f}%, {results['ci_upper']*100:.3f}%]")
    print(f"  Z-statistic               : {results['z_statistic']:.4f}")
    print(f"  P-value (one-tailed)      : {results['p_value']:.4f}")
    print(f"  Significance level (α)    : {results['alpha']}")
    print(f"  Decision                  : {verdict}")
    print("══════════════════════════════════════════════\n")


def plot_z_distribution(results: dict, save_path: str = None):
    """Visualise the z-distribution with observed statistic and critical region."""
    z = results["z_statistic"]
    alpha = results["alpha"]
    z_crit = stats.norm.ppf(1 - alpha)

    x = np.linspace(-4, 4, 400)
    y = stats.norm.pdf(x)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, y, color="#1F2937", linewidth=2)

    # Rejection region
    x_crit = x[x >= z_crit]
    ax.fill_between(x_crit, stats.norm.pdf(x_crit), alpha=0.25, color="#EF4444", label=f"Rejection region (α={alpha})")

    # Observed z
    ax.axvline(z, color="#3B82F6", linestyle="--", linewidth=2, label=f"Observed z = {z:.3f}")
    ax.axvline(z_crit, color="#EF4444", linestyle=":", linewidth=1.5, label=f"Critical z = {z_crit:.3f}")

    ax.annotate(f"p = {results['p_value']:.3f}", xy=(z, stats.norm.pdf(z)),
                xytext=(z + 0.3, stats.norm.pdf(z) + 0.04),
                arrowprops=dict(arrowstyle="->", color="#3B82F6"), fontsize=10, color="#3B82F6")

    ax.set_title("Z-Distribution: Hypothesis Test", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ─── 4. Power Analysis ────────────────────────────────────────────────────────

def minimum_sample_size(p_baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Calculate the minimum sample size per group needed to detect
    a given minimum detectable effect (MDE) with the desired power.
    """
    p2 = p_baseline + mde
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    p_avg = (p_baseline + p2) / 2
    n = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + z_beta * np.sqrt(p_baseline * (1 - p_baseline) + p2 * (1 - p2))) ** 2 / (mde ** 2)
    return int(np.ceil(n))


def plot_power_curve(p_baseline: float, mde_range: list = None, save_path: str = None):
    """Plot required sample size vs MDE for multiple power levels."""
    if mde_range is None:
        mde_range = np.linspace(0.001, 0.02, 100)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for power, color in zip([0.70, 0.80, 0.90], ["#FBBF24", "#3B82F6", "#10B981"]):
        sizes = [minimum_sample_size(p_baseline, mde, power=power) for mde in mde_range]
        ax.plot(mde_range * 100, sizes, label=f"Power = {int(power*100)}%", color=color, linewidth=2)

    ax.axvline(0.5, color="#6B7280", linestyle="--", linewidth=1.2, label="Current MDE (0.5 pp)")
    ax.set_title("Sample Size vs Minimum Detectable Effect", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Minimum Detectable Effect (percentage points)")
    ax.set_ylabel("Required Sample Size (per group)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
