# Data Diary — Credit Card Default Dataset

## Load
Loaded 30000 rows x 25 columns from default of credit card clients.xls. Used header=1 because the first row of the sheet is a title, not column names.

## Column naming
Renamed PAY_0 -> PAY_1 so the six repayment-status columns are consistently PAY_1..PAY_6 (matching BILL_AMT1..6 and PAY_AMT1..6). Renamed target to DEFAULT.

## Data types
Cast ['SEX', 'EDUCATION', 'MARRIAGE'] to categorical dtype (they are codes, not quantities — summing or averaging them is meaningless). Target DEFAULT cast to int8.

## EDUCATION anomalies
Folded undocumented codes 0, 5, 6 (345 rows total, 1.15%) into code 4 ('others'). Rows kept, not dropped.
Before:
EDUCATION
0       14
1    10585
2    14030
3     4917
4      123
5      280
6       51
After:
EDUCATION
0        0
1    10585
2    14030
3     4917
4      468
5        0
6        0

## MARRIAGE anomalies
Folded undocumented code 0 (54 rows, 0.18%) into code 3 ('others'). Rows kept, not dropped.
Before:
MARRIAGE
0       54
1    13659
2    15964
3      323
After:
MARRIAGE
0        0
1    13659
2    15964
3      377

## PAY_1..PAY_6 codes
Kept raw codes (-2, -1, 0, 1-8) unchanged — no official documentation resolves -2/0 precisely, and collapsing them would risk losing predictive signal. Interpretation adopted for context: -2 = no revolving balance that month, -1 = paid in full, 0 = balance carried but statement paid, 1-8 = months overdue. Addressed instead via derived features (see below).

## Negative bill amounts
2.18% of BILL_AMT cells are negative. Interpreted as overpayment / credit balance, not an error — left unchanged rather than clipped to 0, since a real-world scenario explains them and they may be predictive.

## Extreme value check (3x IQR)
Counted values beyond Q3 + 3*IQR per monetary column as a sanity check (not for removal):
LIMIT_BAL       1
BILL_AMT1     786
BILL_AMT2     791
BILL_AMT3     858
BILL_AMT4     899
BILL_AMT5     949
BILL_AMT6     927
PAY_AMT1     1629
PAY_AMT2     1638
PAY_AMT3     1560
PAY_AMT4     1583
PAY_AMT5     1544
PAY_AMT6     1639
Decision: these are kept. In a real financial dataset, a small number of high-limit / high-balance clients is expected, not an error — removing them would bias the model against exactly the segment (high exposure) that risk analysis cares about. Capping/log-transforming, if needed, is left for the modelling stage (Step 4), not the cleaning stage.

## Duplicates
0 duplicated IDs (each row is a distinct client record). However, 35 rows are exact duplicates across all 23 feature columns once ID is excluded — i.e. two different client IDs with an identical LIMIT_BAL/SEX/EDUCATION/.../DEFAULT profile.
Decision: KEEP these rows. With 23 columns (several continuous, e.g. 6 bill amounts and 6 payment amounts spanning thousands of possible values), an exact match happening by chance across every column is extremely unlikely — but it is still far more plausible that these are coincidentally similar low-activity clients (e.g. new/inactive accounts with many zero balances) than that they are erroneous copies of the same person, since each has a distinct ID and this is an anonymised dataset with no name/address to cross-check identity against. Dropping them would silently remove real observations without evidence they are errors, which for a prediction task risks losing a legitimate (if unusually inactive) client segment. Flagged here so the team can revisit if it becomes relevant (e.g. if these rows cluster suspiciously in one class during modelling).

## Derived features
Added AVG_UTILIZATION (avg bill / credit limit), EVER_DELAYED (0/1), MAX_DELAY_MONTHS (worst delay across the 6 months), MONTHS_DELAYED_COUNT (how many of the 6 months were late), and PAYMENT_TREND (PAY_AMT1 - PAY_AMT6, recent vs. 6-months-ago payment amount). These compress the raw columns into features aligned with strands B, C, D.

## Final shape
30000 rows x 29 columns after cleaning.
