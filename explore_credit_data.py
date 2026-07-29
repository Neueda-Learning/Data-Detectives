"""
Step 4 · Explore & analyse (risk-prediction direction)
=======================================================
These are WORKING charts, not presentation deliverables — plain, fast, meant
to help the team see patterns before deciding the story (Step 5). Final
polished visuals belong in Power BI (Step 7).

Run: python explore_credit_data.py
Input:  credit_data_clean.csv   (output of clean_credit_data.py)
Output: charts/*.png            (one or more PNGs per strand)
        Printed summary stats for each strand
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_CSV = "credit_data_clean.csv"
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

df = pd.read_csv(IN_CSV)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols\n")

OVERALL_RATE = df["DEFAULT"].mean()
print(f"Overall default rate: {OVERALL_RATE:.2%}\n")


def default_rate_bar(groupby_col, title, fname, order=None, labels=None, figsize=(7, 4)):
    """Bar chart of default rate by category, with sample size annotated."""
    g = df.groupby(groupby_col, observed=True)["DEFAULT"].agg(["mean", "count"])
    if order is not None:
        g = g.reindex(order)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(g))
    bars = ax.bar(x, g["mean"], color="#4C72B0")
    ax.axhline(OVERALL_RATE, color="grey", linestyle="--", linewidth=1, label=f"overall = {OVERALL_RATE:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels if labels else g.index.astype(str), rotation=0)
    ax.set_ylabel("Default rate")
    ax.set_title(title)
    for xi, (rate, n) in zip(x, zip(g["mean"], g["count"])):
        ax.text(xi, rate + 0.005, f"n={n}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / fname, dpi=120)
    plt.close(fig)
    print(f"[{fname}]\n{g}\n")


# ---------------------------------------------------------------------------
# Strand A — Demographics: does default vary by age / sex / education / marriage?
# ---------------------------------------------------------------------------
print("=" * 70, "\nSTRAND A — Demographics\n", "=" * 70)

default_rate_bar("SEX", "Default rate by sex (1=male, 2=female)", "a1_sex.png")
default_rate_bar(
    "EDUCATION", "Default rate by education level",
    "a2_education.png", order=[1, 2, 3, 4],
    labels=["Grad school", "University", "High school", "Others"],
)
default_rate_bar(
    "MARRIAGE", "Default rate by marital status",
    "a3_marriage.png", order=[1, 2, 3],
    labels=["Married", "Single", "Others"],
)

# Age: bin into ranges rather than plotting all 59 raw ages
df["AGE_BAND"] = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 60, 80],
                         labels=["21-30", "31-40", "41-50", "51-60", "61+"])
default_rate_bar("AGE_BAND", "Default rate by age band", "a4_age_band.png")

age_spread = df.groupby("AGE_BAND", observed=True)["DEFAULT"].mean()
print(f"Age band default-rate spread: {age_spread.max() - age_spread.min():.2%} "
      f"(max {age_spread.idxmax()} vs min {age_spread.idxmin()})\n")

# ---------------------------------------------------------------------------
# Strand B — Credit exposure: limit & utilisation vs default
# ---------------------------------------------------------------------------
print("=" * 70, "\nSTRAND B — Credit limit & utilisation\n", "=" * 70)

df["LIMIT_BAND"] = pd.qcut(df["LIMIT_BAL"], q=5,
                            labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
default_rate_bar("LIMIT_BAND", "Default rate by credit-limit quintile", "b1_limit_band.png")

df["UTIL_BAND"] = pd.qcut(df["AVG_UTILIZATION"].clip(lower=-1, upper=2), q=5, duplicates="drop")
default_rate_bar("UTIL_BAND", "Default rate by utilisation quintile", "b2_utilization_band.png",
                  figsize=(8, 4))

# Scatter-ish view: mean utilisation for defaulters vs non-defaulters
fig, ax = plt.subplots(figsize=(5, 4))
df.boxplot(column="AVG_UTILIZATION", by="DEFAULT", ax=ax, showfliers=False)
ax.set_title("Utilisation ratio: default vs no default")
ax.set_xlabel("DEFAULT (0=no, 1=yes)")
ax.set_ylabel("Avg bill / credit limit")
plt.suptitle("")
fig.tight_layout()
fig.savefig(CHART_DIR / "b3_utilization_boxplot.png", dpi=120)
plt.close(fig)
print(df.groupby("DEFAULT")["AVG_UTILIZATION"].describe()[["mean", "50%"]], "\n")

# ---------------------------------------------------------------------------
# Strand C — Repayment behaviour: PAY_1..6 vs default
# ---------------------------------------------------------------------------
print("=" * 70, "\nSTRAND C — Repayment status\n", "=" * 70)

default_rate_bar("EVER_DELAYED", "Default rate: ever delayed vs never",
                  "c1_ever_delayed.png", order=[0, 1], labels=["Never delayed", "Ever delayed"])

default_rate_bar("MONTHS_DELAYED_COUNT", "Default rate by number of months delayed (0-6)",
                  "c2_months_delayed.png", figsize=(8, 4))

# Which single month's PAY status correlates most with default?
pay_cols = [f"PAY_{i}" for i in range(1, 7)]
corr_with_default = df[pay_cols + ["DEFAULT"]].corr()["DEFAULT"].drop("DEFAULT").sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(corr_with_default.index, corr_with_default.values, color="#C44E52")
ax.set_title("Correlation of each month's PAY status with DEFAULT")
ax.set_ylabel("Correlation")
fig.tight_layout()
fig.savefig(CHART_DIR / "c3_pay_month_correlation.png", dpi=120)
plt.close(fig)
print("Correlation of PAY_1..PAY_6 with DEFAULT:\n", corr_with_default, "\n")
print(f"-> Strongest single-month signal: {corr_with_default.idxmax()} "
      f"(r={corr_with_default.max():.3f})\n")

# ---------------------------------------------------------------------------
# Strand D — Trends over time: do bill/payment patterns shift before default?
# ---------------------------------------------------------------------------
print("=" * 70, "\nSTRAND D — Payment trend over the 6 months\n", "=" * 70)

bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
# Columns are ordered most-recent(1) -> oldest(6); reverse for a left-to-right timeline
months = list(range(6, 0, -1))

avg_bill_by_group = df.groupby("DEFAULT")[bill_cols].mean()[[f"BILL_AMT{i}" for i in months]]
avg_pay_by_group = df.groupby("DEFAULT")[pay_amt_cols].mean()[[f"PAY_AMT{i}" for i in months]]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for default_val, style in zip([0, 1], ["-o", "-s"]):
    axes[0].plot(months, avg_bill_by_group.loc[default_val], style, label=f"DEFAULT={default_val}")
    axes[1].plot(months, avg_pay_by_group.loc[default_val], style, label=f"DEFAULT={default_val}")
axes[0].set_title("Avg bill amount over time")
axes[1].set_title("Avg payment amount over time")
for ax in axes:
    ax.set_xlabel("Months before target month (6=oldest, 1=most recent)")
    ax.invert_xaxis()
    ax.legend()
fig.tight_layout()
fig.savefig(CHART_DIR / "d1_trend_bill_payment.png", dpi=120)
plt.close(fig)
print("Avg bill amount by month (rows=DEFAULT):\n", avg_bill_by_group, "\n")
print("Avg payment amount by month (rows=DEFAULT):\n", avg_pay_by_group, "\n")

default_rate_bar(
    pd.qcut(df["PAYMENT_TREND"], q=5, duplicates="drop").rename("trend_band"),
    "Default rate by payment-trend quintile (PAY_AMT1 - PAY_AMT6)",
    "d2_payment_trend_band.png", figsize=(9, 4),
)

# ---------------------------------------------------------------------------
# Strand E — Combined risk picture: correlation of all engineered features with DEFAULT
# ---------------------------------------------------------------------------
print("=" * 70, "\nSTRAND E — Combined feature ranking\n", "=" * 70)

candidate_features = [
    "LIMIT_BAL", "AGE", "AVG_UTILIZATION", "EVER_DELAYED",
    "MAX_DELAY_MONTHS", "MONTHS_DELAYED_COUNT", "PAYMENT_TREND",
] + pay_cols

corr_all = df[candidate_features + ["DEFAULT"]].corr()["DEFAULT"].drop("DEFAULT")
corr_all = corr_all.reindex(corr_all.abs().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#C44E52" if v > 0 else "#4C72B0" for v in corr_all.values]
ax.barh(corr_all.index[::-1], corr_all.values[::-1], color=colors[::-1])
ax.set_title("Feature correlation with DEFAULT (ranked by strength)")
ax.set_xlabel("Correlation")
fig.tight_layout()
fig.savefig(CHART_DIR / "e1_feature_ranking.png", dpi=120)
plt.close(fig)
print("Features ranked by |correlation| with DEFAULT:\n", corr_all, "\n")

# Simple 2-factor risk segmentation: high utilisation AND ever delayed
df["RISK_SEGMENT"] = np.select(
    [
        (df["EVER_DELAYED"] == 1) & (df["AVG_UTILIZATION"] > df["AVG_UTILIZATION"].median()),
        (df["EVER_DELAYED"] == 1) | (df["AVG_UTILIZATION"] > df["AVG_UTILIZATION"].median()),
    ],
    ["High risk (delayed + high util)", "Medium risk (one factor)"],
    default="Low risk (neither)",
)
default_rate_bar("RISK_SEGMENT", "Default rate by simple 2-factor risk segment",
                  "e2_risk_segment.png", order=["Low risk (neither)", "Medium risk (one factor)",
                                                  "High risk (delayed + high util)"],
                  figsize=(8, 4))

print(f"\nAll charts saved to {CHART_DIR.resolve()}/")
