"""
信用卡违约数据集 - 数据清洗与可视化
UCI Default of Credit Card Clients Dataset
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式，避免显示窗口
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局样式配置
# ============================================================
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

# 设置中文字体（Windows 环境）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = r'c:\Users\Administrator\Desktop\pandatraning'

# ============================================================
# Part 1：数据加载与初步探查
# ============================================================
print("=" * 70)
print("Part 1: 数据加载与初步探查")
print("=" * 70)

# 尝试读取 xls / xlsx / csv 格式的数据文件
data_file = None
for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith(('.xls', '.xlsx', '.csv')):
        data_file = os.path.join(OUTPUT_DIR, fname)
        break

if data_file is None:
    raise FileNotFoundError(
        "未找到数据文件！请将 .xls / .xlsx / .csv 文件放置到：\n"
        f"  {OUTPUT_DIR}"
    )

print(f"[INFO] 读取文件：{data_file}")

if data_file.endswith('.csv'):
    df = pd.read_csv(data_file)
else:
    # UCI 原始 xls 文件第一行是标题说明，第二行才是列名
    try:
        df = pd.read_excel(data_file, header=1)
    except Exception:
        df = pd.read_excel(data_file, header=0)

# 去掉可能存在的 ID 列（列名含 'id' 不区分大小写）
id_cols = [c for c in df.columns if str(c).strip().upper() == 'ID']
if id_cols:
    df.drop(columns=id_cols, inplace=True)

# 将列名统一处理（去空格）
df.columns = [str(c).strip() for c in df.columns]

# 兼容 UCI 英文语义列名，统一映射为 X1~X23
rename_map = {
    'LIMIT_BAL': 'X1',
    'SEX': 'X2',
    'EDUCATION': 'X3',
    'MARRIAGE': 'X4',
    'AGE': 'X5',
    'PAY_0': 'X6',
    'PAY_2': 'X7',
    'PAY_3': 'X8',
    'PAY_4': 'X9',
    'PAY_5': 'X10',
    'PAY_6': 'X11',
    'BILL_AMT1': 'X12',
    'BILL_AMT2': 'X13',
    'BILL_AMT3': 'X14',
    'BILL_AMT4': 'X15',
    'BILL_AMT5': 'X16',
    'BILL_AMT6': 'X17',
    'PAY_AMT1': 'X18',
    'PAY_AMT2': 'X19',
    'PAY_AMT3': 'X20',
    'PAY_AMT4': 'X21',
    'PAY_AMT5': 'X22',
    'PAY_AMT6': 'X23',
}
df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map}, inplace=True)

# 统一目标列名
target_candidates = [c for c in df.columns if 'default' in c.lower()]
if target_candidates and target_candidates[0] != 'default_payment':
    df.rename(columns={target_candidates[0]: 'default_payment'}, inplace=True)

# 必需字段校验，提前报错避免后续 KeyError
required_cols = [f'X{i}' for i in range(1, 24)]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(
        f"缺少必需字段: {missing_required}；当前字段: {df.columns.tolist()}"
    )

print(f"\n[Shape] 行数: {df.shape[0]}，列数: {df.shape[1]}")
print("\n[前 5 行]")
print(df.head())
print("\n[后 5 行]")
print(df.tail())
print("\n[基本信息]")
print(df.info())
print("\n[缺失值统计]")
print(df.isnull().sum())
print("\n[每个变量唯一值分布（value_counts）]")

for col in df.columns:
    vc = df[col].value_counts().sort_index()
    print(f"  {col}: {vc.to_dict()}")
print(df.columns.tolist())
# ============================================================
# Part 2：清洗任务一 —— X3 教育程度幽灵类别归并
# ============================================================
print("\n" + "=" * 70)
print("--- Part 2: X3 Education Cleaning ---")
print("=" * 70)

# 清洗原因：X3 存在 0/5/6 等文档未定义的编码，属于幽灵类别，
#           需统一归并为 4（Others），避免模型将其视为独立类别。
# 清洗方法：使用 Pandas .replace() 将不合法值映射到 4。

x3_before = df['X3'].value_counts().sort_index()
print(f"Before: {x3_before.to_dict()}")

# 找出所有不属于 [1,2,3,4] 的取值
invalid_x3 = [v for v in df['X3'].unique() if v not in [1, 2, 3, 4]]
replace_map_x3 = {v: 4 for v in invalid_x3}
df['X3'] = df['X3'].replace(replace_map_x3)

x3_after = df['X3'].value_counts().sort_index()
print(f"After:  {x3_after.to_dict()}")

# ---- 图 1-A：清洗前 ----
fig, ax = plt.subplots(figsize=(10, 6))
all_vals_x3 = sorted(x3_before.index.tolist())
colors_1a = ['red' if v not in [1, 2, 3, 4] else 'steelblue' for v in all_vals_x3]
ax.bar([str(v) for v in all_vals_x3], x3_before[all_vals_x3].values, color=colors_1a)
ax.set_xlabel('Education Level Code')
ax.set_ylabel('Number of Customers')
ax.set_title('X3 Education Level Distribution (Before Cleaning)')
# 标注异常值
invalid_present = [v for v in all_vals_x3 if v not in [1, 2, 3, 4]]
if invalid_present:
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='steelblue', label='Valid (1-4)'),
        Patch(color='red', label=f'Invalid ({invalid_present}) → merged to 4')
    ]
    ax.legend(handles=legend_handles)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_1A_education_before.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_1A_education_before.png")

# ---- 图 1-B：清洗后 ----
label_map_x3 = {1: 'Graduate\nSchool', 2: 'University', 3: 'High School', 4: 'Others'}
fig, ax = plt.subplots(figsize=(10, 6))
vals_after = sorted(x3_after.index.tolist())
bar_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars = ax.bar(
    [label_map_x3.get(v, str(v)) for v in vals_after],
    x3_after[vals_after].values,
    color=bar_colors[:len(vals_after)]
)
# 数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 50,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('Education Level')
ax.set_ylabel('Number of Customers')
ax.set_title('X3 Education Level Distribution (After Cleaning)')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_1B_education_after.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_1B_education_after.png")

# ============================================================
# Part 3：清洗任务二 —— X4 婚姻状态幽灵类别归并
# ============================================================
print("\n" + "=" * 70)
print("--- Part 3: X4 Marital Cleaning ---")
print("=" * 70)

# 清洗原因：X4 存在 0 等文档未定义的编码，统一归并为 3（Others）。
# 清洗方法：使用 Pandas .replace() 将不合法值映射到 3。

x4_before = df['X4'].value_counts().sort_index()
print(f"Before: {x4_before.to_dict()}")

invalid_x4 = [v for v in df['X4'].unique() if v not in [1, 2, 3]]
replace_map_x4 = {v: 3 for v in invalid_x4}
df['X4'] = df['X4'].replace(replace_map_x4)

x4_after = df['X4'].value_counts().sort_index()
print(f"After:  {x4_after.to_dict()}")

# ---- 图 2-A：清洗前饼图 ----
fig, ax = plt.subplots(figsize=(8, 8))
all_vals_x4 = sorted(x4_before.index.tolist())
pie_colors_before = []
explode_before = []
for v in all_vals_x4:
    if v not in [1, 2, 3]:
        pie_colors_before.append('red')
        explode_before.append(0.1)
    else:
        pie_colors_before.append(None)
        explode_before.append(0)

# matplotlib 不接受 None 颜色，需要完整颜色列表
default_colors = plt.cm.Set2.colors
full_colors_before = []
ci = 0
for v in all_vals_x4:
    if v not in [1, 2, 3]:
        full_colors_before.append('red')
    else:
        full_colors_before.append(default_colors[ci % len(default_colors)])
        ci += 1

wedge_labels = [f'{v} ({"Married" if v==1 else "Single" if v==2 else "Others" if v==3 else "Invalid"})' for v in all_vals_x4]
ax.pie(x4_before[all_vals_x4].values, labels=wedge_labels,
       autopct='%1.1f%%', colors=full_colors_before,
       explode=explode_before, startangle=140)
ax.set_title('X4 Marital Status Distribution (Before Cleaning)')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_2A_marital_before.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_2A_marital_before.png")

# ---- 图 2-B：清洗后饼图 ----
fig, ax = plt.subplots(figsize=(8, 8))
vals_x4_after = sorted(x4_after.index.tolist())
label_map_x4 = {1: 'Married', 2: 'Single', 3: 'Others'}
wedge_labels_after = [label_map_x4.get(v, str(v)) for v in vals_x4_after]
colors_after_x4 = ['#4878CF', '#6ACC65', '#D65F5F'][:len(vals_x4_after)]
ax.pie(x4_after[vals_x4_after].values, labels=wedge_labels_after,
       autopct='%1.1f%%', colors=colors_after_x4, startangle=140)
ax.set_title('X4 Marital Status Distribution (After Cleaning)')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_2B_marital_after.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_2B_marital_after.png")

# ============================================================
# Part 4：清洗任务三 —— 还款状态语义统一（X6-X11）
# ============================================================
print("\n" + "=" * 70)
print("--- Part 4: Repayment Status Recoding ---")
print("=" * 70)

# 清洗原因：-2 和 -1 语义相同（均为按时还款），但数值不同会误导模型；
#           统一将 <=0 的值重编码为 0（无逾期）。
# 清洗方法：使用 Pandas .clip(lower=0) 向量化截断。

repay_cols = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11']
month_labels = ['September', 'August', 'July', 'June', 'May', 'April']

# 保存清洗前数据（用于绘图）
df_repay_before = df[repay_cols].copy()

for col in repay_cols:
    print(f"  {col} unique values before: {sorted(df[col].unique().tolist())}")

# 执行清洗（向量化，禁止 for 循环处理行数据）
df[repay_cols] = df[repay_cols].clip(lower=0)

for col in repay_cols:
    print(f"  {col} unique values after:  {sorted(df[col].unique().tolist())}")

# ---- 图 3-A：清洗前分组柱状图 ----
fig, ax = plt.subplots(figsize=(14, 6))
all_codes_before = sorted(df_repay_before.values.flatten())
unique_codes_before = sorted(set(all_codes_before))

x_idx = np.arange(len(unique_codes_before))
bar_w = 0.13
palette = sns.color_palette('tab10', n_colors=len(repay_cols))

for i, (col, month) in enumerate(zip(repay_cols, month_labels)):
    vc = df_repay_before[col].value_counts()
    heights = [vc.get(c, 0) for c in unique_codes_before]
    ax.bar(x_idx + i * bar_w, heights, width=bar_w, label=month, color=palette[i])

ax.set_xticks(x_idx + bar_w * (len(repay_cols) - 1) / 2)
ax.set_xticklabels([str(c) for c in unique_codes_before])
ax.set_xlabel('Repayment Status Code')
ax.set_ylabel('Number of Customers')
ax.set_title('Repayment Status Distribution by Month (Before Cleaning)')
ax.legend(title='Month')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_3A_repayment_before.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_3A_repayment_before.png")

# ---- 图 3-B：清洗后分组柱状图 ----
fig, ax = plt.subplots(figsize=(14, 6))
unique_codes_after = sorted(df[repay_cols].values.flatten())
unique_codes_after = sorted(set(unique_codes_after))

x_idx2 = np.arange(len(unique_codes_after))

for i, (col, month) in enumerate(zip(repay_cols, month_labels)):
    vc = df[col].value_counts()
    heights = [vc.get(c, 0) for c in unique_codes_after]
    ax.bar(x_idx2 + i * bar_w, heights, width=bar_w, label=month, color=palette[i])

ax.set_xticks(x_idx2 + bar_w * (len(repay_cols) - 1) / 2)
tick_labels_after = ['0\n(On Time)' if c == 0 else str(c) for c in unique_codes_after]
ax.set_xticklabels(tick_labels_after)
ax.set_xlabel('Repayment Status Code')
ax.set_ylabel('Number of Customers')
ax.set_title('Repayment Status Distribution by Month (After Cleaning)')
ax.legend(title='Month')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_3B_repayment_after.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_3B_repayment_after.png")

# ============================================================
# Part 5：特征工程 —— 构造新变量
# ============================================================
print("\n" + "=" * 70)
print("--- Part 5: Feature Engineering ---")
print("=" * 70)

bill_cols = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17']
pay_cols  = ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']

# 1. Avg_Bill：月均账单
df['Avg_Bill'] = df[bill_cols].mean(axis=1)

# 2. Avg_Pay：月均还款
df['Avg_Pay'] = df[pay_cols].mean(axis=1)

# 3. Repayment_Ratio：还款率（Avg_Bill=0 时填充 0，避免除零）
try:
    df['Repayment_Ratio'] = np.where(
        df['Avg_Bill'] == 0,
        0.0,
        df['Avg_Pay'] / df['Avg_Bill']
    )
except ZeroDivisionError:
    df['Repayment_Ratio'] = 0.0

# 4. Ever_Overdue：是否曾经逾期（向量化，禁止 for 循环）
df['Ever_Overdue'] = (df[repay_cols].max(axis=1) >= 1).astype(int)

# 5. Utilization_Rate：额度使用率（X1=0 时填充 0）
try:
    df['Utilization_Rate'] = np.where(
        df['X1'] == 0,
        0.0,
        df['Avg_Bill'] / df['X1']
    )
except ZeroDivisionError:
    df['Utilization_Rate'] = 0.0

print("New columns added: Avg_Bill, Avg_Pay, Repayment_Ratio, Ever_Overdue, Utilization_Rate")
print(df[['Avg_Bill', 'Avg_Pay', 'Repayment_Ratio', 'Ever_Overdue', 'Utilization_Rate']].describe())

# ---- 图 4-A：Avg_Bill 直方图 ----
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Avg_Bill'], bins=50, ax=ax, color='steelblue', kde=True)
ax.set_xlabel('Average Monthly Bill (NT$)')
ax.set_ylabel('Number of Customers')
ax.set_title('Average Monthly Bill Distribution')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_4A_avg_bill.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_4A_avg_bill.png")

# ---- 图 4-B：Avg_Pay 直方图 ----
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Avg_Pay'], bins=50, ax=ax, color='green', kde=True)
ax.set_xlabel('Average Monthly Payment (NT$)')
ax.set_ylabel('Number of Customers')
ax.set_title('Average Monthly Payment Distribution')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_4B_avg_pay.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_4B_avg_pay.png")

# ---- 图 4-C：Repayment_Ratio 直方图 ----
fig, ax = plt.subplots(figsize=(10, 6))
ratio_plot = df['Repayment_Ratio'].clip(upper=3)  # 截断极值仅用于可视化
sns.histplot(ratio_plot, bins=50, ax=ax, color='coral', kde=True)
ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='Full Payment Line')
ax.set_xlabel('Repayment Ratio (Avg_Pay / Avg_Bill)')
ax.set_ylabel('Number of Customers')
ax.set_title('Repayment Ratio Distribution (Avg_Pay / Avg_Bill)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_4C_repayment_ratio.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_4C_repayment_ratio.png")

# ---- 图 4-D：Ever_Overdue 饼图 ----
fig, ax = plt.subplots(figsize=(8, 8))
vc_overdue = df['Ever_Overdue'].value_counts().sort_index()
labels_overdue = ['Never Overdue (0)' if k == 0 else 'Ever Overdue (1)' for k in vc_overdue.index]
ax.pie(vc_overdue.values, labels=labels_overdue,
       autopct='%1.1f%%', colors=['#55A868', '#C44E52'], startangle=140)
ax.set_title('Proportion of Customers with Overdue History')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_4D_ever_overdue.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_4D_ever_overdue.png")

# ---- 图 4-E：Utilization_Rate 直方图 ----
fig, ax = plt.subplots(figsize=(10, 6))
util_plot = df['Utilization_Rate'].clip(upper=2)  # 截断极值仅用于可视化
sns.histplot(util_plot, bins=50, ax=ax, color='mediumpurple', kde=True)
ax.axvline(x=0.8, color='red', linestyle='--', linewidth=1.5, label='High Risk Threshold (80%)')
ax.set_xlabel('Credit Utilization Rate (Avg_Bill / X1)')
ax.set_ylabel('Number of Customers')
ax.set_title('Credit Utilization Rate Distribution (Avg_Bill / X1)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_4E_utilization_rate.png'), dpi=150)
plt.close(fig)
print("[Saved] fig_4E_utilization_rate.png")

# ============================================================
# Part 6：清洗后数据质量校验（Checklist）
# ============================================================
print("\n" + "=" * 70)
print("--- Part 6: Data Quality Checklist ---")
print("=" * 70)

checklist_lines = []

# 1. 缺失值
missing = df.isnull().sum()
checklist_lines.append("=== 1. Missing Values ===")
checklist_lines.append(str(missing))
all_no_missing = (missing.sum() == 0)
status_missing = "✓" if all_no_missing else "✗"
print(f"Missing values: {'0 in all columns' if all_no_missing else missing[missing > 0].to_dict()} {status_missing}")
checklist_lines.append(f"Result: {'PASS' if all_no_missing else 'FAIL'} — {'0 missing' if all_no_missing else 'has missing values'}")

# 2. 数据类型
dtypes_str = str(df.dtypes)
checklist_lines.append("\n=== 2. Data Types ===")
checklist_lines.append(dtypes_str)
print(f"\n[Data Types]\n{df.dtypes}")

# 3. 取值范围校验
checklist_lines.append("\n=== 3. Value Range Check ===")

# X2
x2_vals = set(df['X2'].unique())
x2_ok = x2_vals <= {1, 2}
msg_x2 = f"X2 value range: {x2_vals} {'✓' if x2_ok else '✗'}"
print(msg_x2)
checklist_lines.append(msg_x2)

# X3
x3_vals = set(df['X3'].unique())
x3_ok = x3_vals <= {1, 2, 3, 4}
msg_x3 = f"X3 value range: {x3_vals} {'✓' if x3_ok else '✗'}"
print(msg_x3)
checklist_lines.append(msg_x3)

# X4
x4_vals = set(df['X4'].unique())
x4_ok = x4_vals <= {1, 2, 3}
msg_x4 = f"X4 value range: {x4_vals} {'✓' if x4_ok else '✗'}"
print(msg_x4)
checklist_lines.append(msg_x4)

# X6-X11
repay_ok = True
for col in repay_cols:
    col_vals = set(df[col].unique())
    ok = col_vals <= set(range(10))
    msg = f"{col} value range: {col_vals} {'✓' if ok else '✗'}"
    print(msg)
    checklist_lines.append(msg)
    repay_ok = repay_ok and ok
print(f"X6-X11 value range: {'✓' if repay_ok else '✗'}")

# 4. 逻辑抽查
checklist_lines.append("\n=== 4. Logical Check ===")
sample_check = df[['X12', 'X18']].sample(10, random_state=42)
checklist_lines.append("Sample X12 (Bill) vs X18 (Payment) — 10 rows:")
checklist_lines.append(str(sample_check))
print("\n[Logical Check] 抽查 10 行 X18（还款）vs X12（账单）：")
print(sample_check)

extra_check_pass = (df['Avg_Bill'].min() >= 0) and (df['Avg_Pay'].min() >= 0) and (df['Utilization_Rate'].min() >= 0)
msg_extra = f"Avg_Bill >= 0, Avg_Pay >= 0, Utilization_Rate >= 0: {'✓' if extra_check_pass else '✗'}"
print(msg_extra)
checklist_lines.append(msg_extra)

print(f"\nLogical check: {'Passed ✓' if extra_check_pass else 'Failed ✗'}")

# 图 5-A/B/C：print 输出
print("\n--- 图 5-A: 清洗后 X3 教育程度 value_counts ---")
print(df['X3'].value_counts().sort_index())

print("\n--- 图 5-B: 清洗后 X4 婚姻状态 value_counts ---")
print(df['X4'].value_counts().sort_index())

print("\n--- 图 5-C: 清洗后 X6-X11 各月还款状态 value_counts ---")
for col in repay_cols:
    print(f"  {col}:\n{df[col].value_counts().sort_index()}\n")

# ============================================================
# Part 7：文件导出
# ============================================================
print("\n" + "=" * 70)
print("--- Part 7: 文件导出 ---")
print("=" * 70)

# 导出清洗后数据
csv_path = os.path.join(OUTPUT_DIR, 'cleaned_credit_card_data.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"[Saved] {csv_path}")

# 导出质量校验报告
checklist_path = os.path.join(OUTPUT_DIR, 'data_quality_checklist.txt')
with open(checklist_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("Data Quality Checklist\n")
    f.write("=" * 70 + "\n\n")
    f.write("\n".join(checklist_lines))
print(f"[Saved] {checklist_path}")

# ============================================================
# 运行总结
# ============================================================
print("\n" + "=" * 70)
print(
    f"Data cleaning completed.\n"
    f"Total records: {len(df)}\n"
    f"New features created: Avg_Bill, Avg_Pay, Repayment_Ratio, Ever_Overdue, Utilization_Rate\n"
    f"Files exported: cleaned_credit_card_data.csv, data_quality_checklist.txt"
)
print("=" * 70)
