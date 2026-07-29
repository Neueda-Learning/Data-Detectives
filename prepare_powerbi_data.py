"""
Step 6 · Prepare data for Power BI (risk-prediction story)
=============================================================
All cleaning/labelling/reshaping happens here in Python. Power BI receives
data that is already in good order — it should only need to drag fields onto
visuals, not clean or recode anything.

Produces THREE files:
  1. powerbi_main.csv     - one row per client (wide), labelled, business-
                             friendly column names. Use for strands A, B, C, E
                             (bar charts, segmentation, correlation-style views).
  2. powerbi_monthly.csv  - one row per client PER MONTH (long/tidy format).
                             Use for strand D (trend line charts over the 6
                             months) — a long table is far easier to plot by
                             month in Power BI than 6 separate wide columns.
  3. summary_reference.csv - the key headline numbers computed in Python
                             (e.g. default rate by risk segment), so the team
                             can sanity-check that Power BI visuals reproduce
                             the same figures during rehearsal.

Run: python prepare_powerbi_data.py
Input:  credit_data_clean.csv   (output of clean_credit_data.py)
"""

import numpy as np
import pandas as pd

IN_CSV = "credit_data_clean.csv"
df = pd.read_csv(IN_CSV)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")

# ---------------------------------------------------------------------------
# 1. Recreate the analysis bands used in Step 4, so Power BI gets them
#    ready-made rather than the team recoding them again in DAX.
# ---------------------------------------------------------------------------
df["AGE_BAND"] = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 60, 80],
                         labels=["21-30", "31-40", "41-50", "51-60", "61+"])
df["LIMIT_BAND"] = pd.qcut(df["LIMIT_BAL"], q=5,
                            labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
df["RISK_SEGMENT"] = np.select(
    [
        (df["EVER_DELAYED"] == 1) & (df["AVG_UTILIZATION"] > df["AVG_UTILIZATION"].median()),
        (df["EVER_DELAYED"] == 1) | (df["AVG_UTILIZATION"] > df["AVG_UTILIZATION"].median()),
    ],
    ["High risk (delayed + high util)", "Medium risk (one factor)"],
    default="Low risk (neither)",
)

# ---------------------------------------------------------------------------
# 2. Decode categorical codes into readable labels
# ---------------------------------------------------------------------------
sex_map = {1: "Male", 2: "Female"}
education_map = {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}
marriage_map = {1: "Married", 2: "Single", 3: "Others"}
default_map = {0: "No default", 1: "Default"}

df["Sex"] = df["SEX"].map(sex_map)
df["Education"] = df["EDUCATION"].map(education_map)
df["Marriage"] = df["MARRIAGE"].map(marriage_map)
df["Default"] = df["DEFAULT"].map(default_map)

assert df[["Sex", "Education", "Marriage", "Default"]].isnull().sum().sum() == 0, \
    "Unmapped category found — check the code->label dictionaries above"

# ---------------------------------------------------------------------------
# 3. Build the MAIN table (client grain, wide, business-friendly names)
# ---------------------------------------------------------------------------
main = df[[
    "ID", "Sex", "Education", "Marriage", "AGE", "AGE_BAND",
    "LIMIT_BAL", "LIMIT_BAND", "AVG_UTILIZATION",
    "EVER_DELAYED", "MAX_DELAY_MONTHS", "MONTHS_DELAYED_COUNT",
    "PAYMENT_TREND", "RISK_SEGMENT", "Default",
]].rename(columns={
    "ID": "ClientID",
    "AGE": "Age",
    "AGE_BAND": "Age Band",
    "LIMIT_BAL": "Credit Limit",
    "LIMIT_BAND": "Credit Limit Band",
    "AVG_UTILIZATION": "Utilization Rate",
    "EVER_DELAYED": "Ever Delayed",
    "MAX_DELAY_MONTHS": "Max Delay (months)",
    "MONTHS_DELAYED_COUNT": "Months Delayed (count)",
    "PAYMENT_TREND": "Payment Trend",
    "RISK_SEGMENT": "Risk Segment",
})
# Ever Delayed as a readable label too, since it will likely be used as a
# legend/slicer in Power BI, not just a numeric filter.
main["Ever Delayed"] = main["Ever Delayed"].map({0: "No", 1: "Yes"})

main.to_csv("powerbi_main.csv", index=False)
print(f"\npowerbi_main.csv -> {main.shape[0]} rows x {main.shape[1]} cols")
print(main.dtypes)

# ---------------------------------------------------------------------------
# 4. Build the MONTHLY table (client x month grain, long/tidy format)
# ---------------------------------------------------------------------------
# BILL_AMT1/PAY_AMT1/PAY_1 = most recent month, ...6 = 6 months ago.
# Convert "N months ago" (1..6) into a Month Offset so Power BI can sort and
# plot it as a proper timeline (0 = most recent, -5 = 6 months ago).
records = []
for i in range(1, 7):
    month_offset = -(i - 1)  # 1 -> 0 (most recent), 6 -> -5 (6 months back)
    records.append(pd.DataFrame({
        "ClientID": df["ID"],
        "Month Offset": month_offset,
        "Bill Amount": df[f"BILL_AMT{i}"],
        "Payment Amount": df[f"PAY_AMT{i}"],
        "Pay Status Code": df[f"PAY_{i}"],
        "Default": df["Default"],
    }))
monthly = pd.concat(records, ignore_index=True).sort_values(["ClientID", "Month Offset"])
monthly.to_csv("powerbi_monthly.csv", index=False)
print(f"\npowerbi_monthly.csv -> {monthly.shape[0]} rows x {monthly.shape[1]} cols "
      f"({df.shape[0]} clients x 6 months)")

# ---------------------------------------------------------------------------
# 5. Build the SUMMARY REFERENCE table (headline numbers for rehearsal)
# ---------------------------------------------------------------------------
summary_rows = []


def add_summary(strand, metric, group_col, source=main):
    g = source.groupby(group_col, observed=True)["Default"].apply(
        lambda s: (s == "Default").mean()
    )
    for k, v in g.items():
        summary_rows.append({"Strand": strand, "Metric": metric, "Group": k, "Default Rate": round(v, 4)})


add_summary("A", "Default rate by sex", "Sex")
add_summary("A", "Default rate by education", "Education")
add_summary("A", "Default rate by age band", "Age Band")
add_summary("B", "Default rate by credit limit band", "Credit Limit Band")
add_summary("C", "Default rate by ever delayed", "Ever Delayed")
add_summary("E", "Default rate by risk segment", "Risk Segment")

summary_rows.append({
    "Strand": "Overall", "Metric": "Overall default rate", "Group": "All clients",
    "Default Rate": round((main["Default"] == "Default").mean(), 4),
})

summary = pd.DataFrame(summary_rows)
summary.to_csv("summary_reference.csv", index=False)
print(f"\nsummary_reference.csv -> {summary.shape[0]} rows")
print(summary.to_string(index=False))

print("\nDone. All three files are ready to import into Power BI.")
print("Relationship to set up in Power BI: powerbi_main[ClientID] 1 -> * powerbi_monthly[ClientID]")
