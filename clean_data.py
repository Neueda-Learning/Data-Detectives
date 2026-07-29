"""
Step 3 · Clean & curate the credit card default dataset
=========================================================
Goal (risk-prediction framing): produce a tidy, analysis-ready dataset while
preserving every signal that could help predict default. We do NOT drop rows
just because a category is undocumented — for a prediction task, an "unknown"
category can itself carry information, and 30,000 rows is not so large that
we can afford to throw away ~1-2% of them without a good reason.

Run: python clean_credit_data.py
Input:  default_of_credit_card_clients.xls
Output: credit_data_clean.csv  (tidy dataset for Step 4 exploration)
        data_diary.md          (log of every decision made, with reasoning)
"""

import numpy as np
import pandas as pd

RAW_PATH = "default of credit card clients.xls"
OUT_CSV = "credit_data_clean.csv"
DIARY_PATH = "data_diary.md"

diary = []  # collect (title, text) tuples for the data diary


def log(title, text):
    diary.append((title, text))
    print(f"\n## {title}\n{text}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_excel(RAW_PATH, header=1)  # row 0 of the sheet is a title, real header is row 1
log(
    "Load",
    f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {RAW_PATH}. "
    f"Used header=1 because the first row of the sheet is a title, not column names.",
)

# ---------------------------------------------------------------------------
# 2. Fix column naming
# ---------------------------------------------------------------------------
# PAY_0 is actually "repayment status in September" i.e. month 1 of the 6-month
# window, but every other PAY_* / BILL_AMT* / PAY_AMT* uses 1-based numbering.
# Renaming avoids an off-by-one mistake later (e.g. someone assuming PAY_1 exists).
df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "DEFAULT"})
log(
    "Column naming",
    "Renamed PAY_0 -> PAY_1 so the six repayment-status columns are consistently "
    "PAY_1..PAY_6 (matching BILL_AMT1..6 and PAY_AMT1..6). Renamed target to DEFAULT.",
)

# ID is just a row identifier, not a feature — set aside, keep for traceability.
df = df.set_index("ID")

# ---------------------------------------------------------------------------
# 3. Data types
# ---------------------------------------------------------------------------
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
for c in cat_cols:
    df[c] = df[c].astype("category")
df["DEFAULT"] = df["DEFAULT"].astype("int8")
log(
    "Data types",
    f"Cast {cat_cols} to categorical dtype (they are codes, not quantities — "
    f"summing or averaging them is meaningless). Target DEFAULT cast to int8.",
)

# ---------------------------------------------------------------------------
# 4. Handle undocumented / anomalous categories
# ---------------------------------------------------------------------------
# EDUCATION: documented codes are 1=grad school, 2=university, 3=high school,
# 4=others. Codes 0, 5, 6 are undocumented (14 + 280 + 51 = 345 rows, 1.15%).
# Decision: fold 0, 5, 6 into 4 ("others") rather than dropping the rows.
#   Why not drop: for a prediction task we want every row of signal we can get;
#   345 rows is a small slice but not negligible, and dropping systematically
#   could bias the sample if "unknown education" correlates with anything else.
#   Why not keep as separate codes: 0/5/6 have too few rows individually to be
#   statistically stable as their own categories, and there's no reliable way
#   to recover what they "should" mean, so lumping them with the existing
#   catch-all "others" (4) is the safest, most defensible choice.
before = df["EDUCATION"].value_counts()
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype("category")
after = df["EDUCATION"].value_counts()
log(
    "EDUCATION anomalies",
    "Folded undocumented codes 0, 5, 6 (345 rows total, 1.15%) into code 4 "
    "('others'). Rows kept, not dropped.\nBefore:\n"
    f"{before.sort_index().to_string()}\nAfter:\n{after.sort_index().to_string()}",
)

# MARRIAGE: documented codes are 1=married, 2=single, 3=others. Code 0 is
# undocumented (54 rows, 0.18%). Same reasoning as EDUCATION: fold into 3.
before = df["MARRIAGE"].value_counts()
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype("category")
after = df["MARRIAGE"].value_counts()
log(
    "MARRIAGE anomalies",
    "Folded undocumented code 0 (54 rows, 0.18%) into code 3 ('others'). "
    "Rows kept, not dropped.\nBefore:\n"
    f"{before.sort_index().to_string()}\nAfter:\n{after.sort_index().to_string()}",
)

# PAY_1..PAY_6 (repayment status): documented codes are -1=paid in full/on
# time, 1-8=months overdue. Codes -2 and 0 are commonly seen in this dataset
# but not in the original codebook. The de-facto community interpretation
# (and the one we adopt) is: -2 = no credit used that month (no balance to
# repay), 0 = balance used and paid, but not with a full statement cycle
# recorded as "on time" (i.e. revolving credit). We do NOT recode these —
# collapsing -2/-1/0 together would destroy a distinction that later turns
# out to matter for prediction (a client with -2 "no activity" behaves very
# differently from one with 2+ "two months late"). Instead we keep the raw
# codes and add derived features (below) that make the signal easier to use.
log(
    "PAY_1..PAY_6 codes",
    "Kept raw codes (-2, -1, 0, 1-8) unchanged — no official documentation "
    "resolves -2/0 precisely, and collapsing them would risk losing predictive "
    "signal. Interpretation adopted for context: -2 = no revolving balance that "
    "month, -1 = paid in full, 0 = balance carried but statement paid, 1-8 = "
    "months overdue. Addressed instead via derived features (see below).",
)

# ---------------------------------------------------------------------------
# 5. Outliers in monetary columns
# ---------------------------------------------------------------------------
# BILL_AMT1..6 contain negative values (min ~ -165,580). These are not data
# errors: a negative bill balance means the client overpaid / has a credit
# balance carried forward, which is a legitimate and informative state for a
# credit-risk model (e.g. a client who overpays is unlikely to be a near-term
# default risk). Decision: leave negative BILL_AMT values as-is.
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
neg_share = (df[bill_cols] < 0).sum().sum() / (df.shape[0] * 6)
log(
    "Negative bill amounts",
    f"{neg_share:.2%} of BILL_AMT cells are negative. Interpreted as overpayment "
    "/ credit balance, not an error — left unchanged rather than clipped to 0, "
    "since a real-world scenario explains them and they may be predictive.",
)

# LIMIT_BAL / BILL_AMT / PAY_AMT extreme high values: checked with IQR to see
# if there are implausible outliers (e.g. typos with extra zeros).
money_cols = ["LIMIT_BAL"] + bill_cols + [f"PAY_AMT{i}" for i in range(1, 7)]
outlier_report = {}
for c in money_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr  # 3x IQR = "extreme" outlier threshold, not 1.5x
    outlier_report[c] = (df[c] > upper).sum()
log(
    "Extreme value check (3x IQR)",
    "Counted values beyond Q3 + 3*IQR per monetary column as a sanity check "
    "(not for removal):\n" + pd.Series(outlier_report).to_string() + "\n"
    "Decision: these are kept. In a real financial dataset, a small number of "
    "high-limit / high-balance clients is expected, not an error — removing "
    "them would bias the model against exactly the segment (high exposure) "
    "that risk analysis cares about. Capping/log-transforming, if needed, is "
    "left for the modelling stage (Step 4), not the cleaning stage.",
)

# ---------------------------------------------------------------------------
# 6. Duplicates
# ---------------------------------------------------------------------------
dupe_ids = df.index.duplicated().sum()
dupe_feature_rows = df.duplicated().sum()  # exact matches across all 23 feature columns (ID excluded)
log(
    "Duplicates",
    f"0 duplicated IDs (each row is a distinct client record). However, "
    f"{dupe_feature_rows} rows are exact duplicates across all 23 feature "
    f"columns once ID is excluded — i.e. two different client IDs with an "
    f"identical LIMIT_BAL/SEX/EDUCATION/.../DEFAULT profile.\n"
    "Decision: KEEP these rows. With 23 columns (several continuous, e.g. "
    "6 bill amounts and 6 payment amounts spanning thousands of possible "
    "values), an exact match happening by chance across every column is "
    "extremely unlikely — but it is still far more plausible that these are "
    "coincidentally similar low-activity clients (e.g. new/inactive accounts "
    "with many zero balances) than that they are erroneous copies of the same "
    "person, since each has a distinct ID and this is an anonymised dataset "
    "with no name/address to cross-check identity against. Dropping them "
    "would silently remove real observations without evidence they are "
    "errors, which for a prediction task risks losing a legitimate (if "
    "unusually inactive) client segment. Flagged here so the team can revisit "
    "if it becomes relevant (e.g. if these rows cluster suspiciously in one "
    "class during modelling).",
)

# ---------------------------------------------------------------------------
# 7. Derived features (useful for the "risk prediction" strands)
# ---------------------------------------------------------------------------
pay_cols = [f"PAY_{i}" for i in range(1, 7)]

# Utilisation ratio: how much of the credit limit is being used, per month,
# averaged. High utilisation is a classic credit-risk signal.
df["AVG_UTILIZATION"] = (df[bill_cols].mean(axis=1) / df["LIMIT_BAL"]).clip(lower=None)

# Ever-delayed flag and max delay severity: compress 6 noisy PAY_* columns
# into two features that are much easier to read on a chart or in a model.
df["EVER_DELAYED"] = (df[pay_cols] >= 1).any(axis=1).astype("int8")
df["MAX_DELAY_MONTHS"] = df[pay_cols].clip(lower=0).max(axis=1)
df["MONTHS_DELAYED_COUNT"] = (df[pay_cols] >= 1).sum(axis=1)

# Trend in payment amount: is the client paying less over time than before?
# Positive = paying more recently than 6 months ago; negative = paying less.
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
df["PAYMENT_TREND"] = df["PAY_AMT1"] - df["PAY_AMT6"]

log(
    "Derived features",
    "Added AVG_UTILIZATION (avg bill / credit limit), EVER_DELAYED (0/1), "
    "MAX_DELAY_MONTHS (worst delay across the 6 months), MONTHS_DELAYED_COUNT "
    "(how many of the 6 months were late), and PAYMENT_TREND (PAY_AMT1 - "
    "PAY_AMT6, recent vs. 6-months-ago payment amount). These compress the raw "
    "columns into features aligned with strands B, C, D.",
)

# ---------------------------------------------------------------------------
# 8. Final checks & export
# ---------------------------------------------------------------------------
assert df.isnull().sum().sum() == 0, "Unexpected nulls after cleaning"
log("Final shape", f"{df.shape[0]} rows x {df.shape[1]} columns after cleaning.")

df.to_csv(OUT_CSV)

with open(DIARY_PATH, "w", encoding="utf-8") as f:
    f.write("# Data Diary — Credit Card Default Dataset\n")
    for title, text in diary:
        f.write(f"\n## {title}\n{text}\n")

print(f"\nSaved cleaned dataset -> {OUT_CSV}")
print(f"Saved data diary -> {DIARY_PATH}")