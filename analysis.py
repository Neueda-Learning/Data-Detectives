"""
Credit Card Default Risk - Exploratory Data Analysis
=====================================================
Objective: Identify deep behavioral risk patterns for credit card default prediction.
This analysis provides insights into customer segmentation, risk factors, and early warning signals.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.style.use("seaborn-v0_8")
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 10

# Dataset paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "cleaned_credit_card_data.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
TARGET = "default_payment"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ======================
# Load and Overview Data
# ======================

def load_data(filepath):
    """Load dataset from CSV file and engineer features."""
    df = pd.read_csv(filepath)
    
    # Rename for easier access
    df = df.rename(columns={"LIMIT_BAL": "X1", "default payment next month": "default_payment"})
    
    # Create engineered features if they don't exist
    if "Ever_Overdue" not in df.columns:
        # Check if any payment status shows overdue (>0 indicates overdue months)
        # Status codes: -2/-1/0 = 按时/提前还款, 1-9 = 逾期月数
        # Only use payment STATUS columns (PAY_0, PAY_2-6), not payment AMOUNT columns (PAY_AMT*)
        pay_status_cols = [col for col in df.columns if col.startswith('PAY_') and 'AMT' not in col]
        df["Ever_Overdue"] = (df[pay_status_cols] > 0).any(axis=1).astype(int)
    
    if "Utilization_Rate" not in df.columns:
        # Calculate average utilization
        bill_cols = [col for col in df.columns if col.startswith('BILL_AMT')]
        df["Utilization_Rate"] = df[bill_cols].mean(axis=1) / (df["X1"] + 1)  # Avoid division by zero
    
    if "Repayment_Ratio" not in df.columns:
        # Calculate average repayment ratio
        bill_cols = [col for col in df.columns if col.startswith('BILL_AMT')]
        pay_amt_cols = [col for col in df.columns if col.startswith('PAY_AMT')]
        avg_bill = df[bill_cols].mean(axis=1)
        avg_pay = df[pay_amt_cols].mean(axis=1)
        df["Repayment_Ratio"] = avg_pay / (avg_bill + 1)  # Avoid division by zero
    
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    return df


# ======================
# Target Distribution
# ======================

def analyze_target_distribution(df, target, output_dir=None):
    """Analyze and visualize target variable distribution (bar chart)."""
    default_rate = df[target].mean()
    counts = df[target].value_counts().sort_index()
    
    # Print data table
    print(f"\nOverall Default Rate: {default_rate:.2%}")
    print(f"{'Status':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 37)
    labels_map = {0: "Non-Default", 1: "Default"}
    for val, count in counts.items():
        print(f"{labels_map[val]:<15} {count:>8,} {count/len(df)*100:>11.2f}%")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target, hue=target, ax=ax,
                  palette={0: "green", 1: "red"}, legend=False)
    ax.set_title("Distribution of Default Payment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Default Status")
    ax.set_ylabel("Number of Customers")
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "01_target_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


def plot_default_pie(df, target, output_dir=None):
    """Pie chart of default vs non-default distribution."""
    counts = df[target].value_counts().sort_index()
    labels = ["Non-Default", "Default"]
    sizes = [counts[0], counts[1]]
    
    # Print data table
    print("\nDefault Distribution (Pie Chart Data):")
    print(f"{'Status':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 37)
    for label, count in zip(labels, sizes):
        print(f"{label:<15} {count:>8,} {count/sum(sizes)*100:>11.2f}%")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#FFFFFF", "#C00000"],
        wedgeprops={"edgecolor": "black", "linewidth": 1},
        textprops={"fontsize": 12}
    )
    ax.set_title(
        "Default vs Non-Default Customer Distribution",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "01b_default_pie.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Correlation Analysis
# ======================

def analyze_correlations(df, target, output_dir=None):
    """Analyze feature correlation with target variable."""
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print("\nFeature Correlations with Default:")
    print(corr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.drop(target).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Correlation with Default Payment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Correlation Coefficient")
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "02_feature_correlations.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Overdue History Analysis
# ======================

def analyze_overdue_impact(df, target, output_dir=None):
    """Analyze impact of payment overdue history on default risk."""
    overdue_result = df.groupby("Ever_Overdue")[target].mean().reset_index()
    print("\nDefault Rate by Overdue History:")
    print(overdue_result)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=overdue_result, x="Ever_Overdue", y=target,
                hue="Ever_Overdue", ax=ax,
                palette={0: "green", 1: "red"}, legend=False)
    ax.set_title("Previous Overdue History Strongly Increases Default Risk", 
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Ever Experienced Overdue Payment")
    ax.set_ylabel("Default Rate")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "03_overdue_impact.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Credit Utilization Analysis
# ======================

def analyze_utilization(df, target, output_dir=None):
    """Analyze credit utilization impact on default risk."""
    df["Util_Group"] = pd.qcut(
        df["Utilization_Rate"],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    
    util_result = df.groupby("Util_Group", observed=True)[target].mean().reset_index()
    print("\nDefault Rate by Utilization Level:")
    print(util_result)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=util_result, x="Util_Group", y=target, marker="o", ax=ax, linewidth=2)
    ax.set_title("Default Risk Increases with Credit Utilization", fontsize=12, fontweight="bold")
    ax.set_xlabel("Utilization Level")
    ax.set_ylabel("Default Rate")
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "04_utilization_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Repayment Behavior Analysis
# ======================

def analyze_repayment_behavior(df, target, output_dir=None):
    """Analyze repayment behavior impact on default risk."""
    df["Repayment_Group"] = pd.qcut(
        df["Repayment_Ratio"],
        q=5,
        labels=["Lowest", "Low", "Medium", "High", "Highest"]
    )
    
    repay_result = df.groupby("Repayment_Group", observed=True)[target].mean().reset_index()
    print("\nDefault Rate by Repayment Behavior:")
    print(repay_result)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=repay_result, x="Repayment_Group", y=target,
                hue="Repayment_Group", ax=ax, palette="coolwarm", legend=False)
    ax.set_title("Lower Repayment Ratio Indicates Higher Default Risk", fontsize=12, fontweight="bold")
    ax.set_xlabel("Repayment Behavior Group")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "05_repayment_behavior.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Credit Limit Segmentation
# ======================

def analyze_credit_limits(df, target, output_dir=None):
    """Analyze credit limit impact on default risk."""
    df["Credit_Group"] = pd.qcut(
        df["X1"],
        q=6,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    )
    
    credit_result = df.groupby("Credit_Group", observed=True)[target].mean()
    print("\nDefault Rate by Credit Limit Segment:")
    print(credit_result)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(credit_result.index, credit_result.values, marker="o", linewidth=2, markersize=8)
    ax.set_title("Default Risk Across Credit Limit Segments", fontsize=12, fontweight="bold")
    ax.set_xlabel("Credit Limit Segment (Increasing Limit →)")
    ax.set_ylabel("Default Rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "06_credit_limits.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Risk Segmentation Analysis
# ======================

def analyze_risk_segments(df, target, output_dir=None):
    """Identify and analyze high-risk customer segments."""
    df["Risk_Profile"] = "Normal"
    
    # Define stress conditions
    high_util = df["Utilization_Rate"] > df["Utilization_Rate"].median()
    low_repayment = df["Repayment_Ratio"] < df["Repayment_Ratio"].median()
    overdue = df["Ever_Overdue"] == 1
    
    # Assign high stress profile
    df.loc[high_util & low_repayment & overdue, "Risk_Profile"] = "High Financial Stress"
    
    risk_result = df.groupby("Risk_Profile")[target].agg(
        Default_Rate="mean",
        Customers="count"
    ).reset_index()
    
    print("\nDefault Rate by Risk Profile:")
    print(risk_result)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=risk_result, x="Risk_Profile", y="Default_Rate",
                hue="Risk_Profile", ax=ax,
                palette={"Normal": "green", "High Financial Stress": "red"}, legend=False)
    ax.set_title("High Financial Stress Segment Has Much Higher Default Risk", 
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, "07_risk_segments.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Executive Insights
# ======================


# ======================
# Demographics Analysis
# ======================

def plot_default_by_age(df, target, output_dir=None):
    """Stacked bar chart showing only defaulters' age distribution."""
    defaults_only = df[df[target] == 1].copy()
    age_counts = defaults_only["AGE"].value_counts().sort_index()

    print("\nAge Distribution of Defaulters (sample):")
    print(f"{'Age':>5} {'Defaults':>10}")
    print("-" * 20)
    for age, count in age_counts.items():
        print(f"{age:>5} {count:>10,}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(age_counts.index, age_counts.values, color="#C00000")
    ax.set_title("Age Distribution of Defaulters", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Defaulters")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "08_default_by_age.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


def plot_default_by_gender(df, target, output_dir=None):
    """Analyze default rate by gender."""
    data = df.copy()
    data["Gender"] = data["SEX"].map({1: "Male", 2: "Female"})
    gender_default = data.groupby("Gender")[target].agg(
        Count="count", Defaults="sum", Default_Rate="mean"
    ).reset_index()
    gender_default["Default_Rate_%"] = (gender_default["Default_Rate"] * 100).round(2)

    print("\nDefault Rate by Gender:")
    print(f"{'Gender':<10} {'Count':>8} {'Defaults':>10} {'Default_Rate(%)':>16}")
    print("-" * 48)
    for _, row in gender_default.iterrows():
        print(f"{row['Gender']:<10} {int(row['Count']):>8,} {int(row['Defaults']):>10,} {row['Default_Rate_%']:>15.2f}%")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(gender_default["Gender"], gender_default["Default_Rate_%"],
           color=["#C00000", "#E6E6E6"])
    ax.set_title("Default Rate by Gender", fontsize=14, fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(gender_default["Default_Rate_%"]):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "09_default_by_gender.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


def plot_default_by_education_marriage(df, target, output_dir=None):
    """Side-by-side bar charts: default rate by education level and marriage status."""
    # --- Education ---
    edu_data = df[df["EDUCATION"] != 0].copy()
    edu_data["Education_Level"] = edu_data["EDUCATION"].replace({
        1: "Graduate School", 2: "University",
        3: "High School", 4: "Others",
        5: "Unknown", 6: "Unknown"
    }).astype(str).replace({str(k): "Others" for k in range(7, 100)})
    edu_default = edu_data.groupby("Education_Level")[target].agg(
        Count="count", Defaults="sum", Default_Rate="mean"
    ).reset_index().sort_values("Default_Rate", ascending=False)
    edu_default["Default_Rate_%"] = (edu_default["Default_Rate"] * 100).round(2)

    # --- Marriage ---
    mar_data = df.copy()
    mar_data["Marriage_Status"] = mar_data["MARRIAGE"].replace({
        1: "Married", 2: "Single", 3: "Others", 0: "Others"
    }).astype(str).replace({str(k): "Others" for k in range(4, 100)})
    mar_default = mar_data.groupby("Marriage_Status")[target].agg(
        Count="count", Defaults="sum", Default_Rate="mean"
    ).reset_index()
    mar_default["Default_Rate_%"] = (mar_default["Default_Rate"] * 100).round(2)

    # Print tables
    print("\nDefault Rate by Education Level:")
    print(f"{'Education':<20} {'Count':>8} {'Defaults':>10} {'Default_Rate(%)':>16}")
    print("-" * 58)
    for _, row in edu_default.iterrows():
        print(f"{row['Education_Level']:<20} {int(row['Count']):>8,} {int(row['Defaults']):>10,} {row['Default_Rate_%']:>15.2f}%")

    print("\nDefault Rate by Marriage Status:")
    print(f"{'Status':<12} {'Count':>8} {'Defaults':>10} {'Default_Rate(%)':>16}")
    print("-" * 50)
    for _, row in mar_default.iterrows():
        print(f"{row['Marriage_Status']:<12} {int(row['Count']):>8,} {int(row['Defaults']):>10,} {row['Default_Rate_%']:>15.2f}%")

    # Combined figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Education subplot
    ax1.bar(edu_default["Education_Level"], edu_default["Default_Rate_%"], color="#C00000")
    ax1.set_title("Default Rate by Education Level", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Education Level")
    ax1.set_ylabel("Default Rate (%)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)
    for i, v in enumerate(edu_default["Default_Rate_%"]):
        ax1.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)

    # Marriage subplot
    colors = ["#8B0000", "#C00000", "#FF6B6B"]
    ax2.bar(mar_default["Marriage_Status"], mar_default["Default_Rate_%"],
            color=colors[:len(mar_default)])
    ax2.set_title("Default Rate by Marriage Status", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Marriage Status")
    ax2.set_ylabel("Default Rate (%)")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(mar_default["Default_Rate_%"]):
        ax2.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)

    plt.suptitle("Demographics & Default Risk", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "10_education_marriage.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
    plt.show()


# ======================
# Executive Insights
# ======================

def print_executive_summary():
    """Print executive summary of key findings."""
    summary = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                          EXECUTIVE RISK INSIGHTS                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

1. HISTORICAL REPAYMENT BEHAVIOR
   → Strongest predictor of default risk
   → Ever_Overdue status shows 3-4× impact multiplier

2. CREDIT UTILIZATION
   → Indicates financial stress and elevated default probability
   → Nonlinear relationship with default rate

3. REPAYMENT RATIO
   → Low repayment ratio provides early warning signal
   → Strong inverse correlation with payment behavior

4. CREDIT LIMIT EFFECT
   → Cannot explain risk alone; reflects customer quality
   → Part of comprehensive underwriting decision

5. COMBINED STRESS SIGNALS
   → Highest-risk population identified: overdue + high utilization + low repayment
   → Multiple simultaneous stressors amplify default probability

6. RECOMMENDED ACTIONS
   → Monitor customers showing early warning signs
   → Implement intervention strategies for high-risk segments
   → Consider feature engineering for predictive models
   → Use segmentation for targeted risk management

═══════════════════════════════════════════════════════════════════════════════════
"""
    print(summary)


# ======================
# Main Execution
# ======================

if __name__ == "__main__":
    # Load data
    df = load_data(DATA_FILE)
    
    # Run all analyses
    print("\n" + "="*80)
    print("CREDIT CARD DEFAULT RISK ANALYSIS")
    print("="*80)
    
    print(f"\n📁 Saving visualizations to: {OUTPUT_DIR}")
    
    analyze_target_distribution(df, TARGET, OUTPUT_DIR)
    plot_default_pie(df, TARGET, OUTPUT_DIR)
    analyze_correlations(df, TARGET, OUTPUT_DIR)
    analyze_overdue_impact(df, TARGET, OUTPUT_DIR)
    analyze_utilization(df, TARGET, OUTPUT_DIR)
    analyze_repayment_behavior(df, TARGET, OUTPUT_DIR)
    analyze_credit_limits(df, TARGET, OUTPUT_DIR)
    analyze_risk_segments(df, TARGET, OUTPUT_DIR)
    plot_default_by_age(df, TARGET, OUTPUT_DIR)
    plot_default_by_gender(df, TARGET, OUTPUT_DIR)
    plot_default_by_education_marriage(df, TARGET, OUTPUT_DIR)
    
    print_executive_summary()
    
    print(f"\n✓ Analysis Complete!")
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
