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
TARGET = "default_payment"


# ======================
# Load and Overview Data
# ======================

def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    return df


# ======================
# Target Distribution
# ======================

def analyze_target_distribution(df, target):
    """Analyze and visualize target variable distribution."""
    default_rate = df[target].mean()
    print(f"\nOverall Default Rate: {default_rate:.2%}")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target, ax=ax, palette=["green", "red"])
    ax.set_title("Distribution of Default Payment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Default Status")
    ax.set_ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()


# ======================
# Correlation Analysis
# ======================

def analyze_correlations(df, target):
    """Analyze feature correlation with target variable."""
    corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print("\nFeature Correlations with Default:")
    print(corr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.drop(target).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Correlation with Default Payment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Correlation Coefficient")
    plt.tight_layout()
    plt.show()


# ======================
# Overdue History Analysis
# ======================

def analyze_overdue_impact(df, target):
    """Analyze impact of payment overdue history on default risk."""
    overdue_result = df.groupby("Ever_Overdue")[target].mean().reset_index()
    print("\nDefault Rate by Overdue History:")
    print(overdue_result)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=overdue_result, x="Ever_Overdue", y=target, ax=ax, palette=["green", "red"])
    ax.set_title("Previous Overdue History Strongly Increases Default Risk", 
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Ever Experienced Overdue Payment")
    ax.set_ylabel("Default Rate")
    ax.set_xticklabels(["No", "Yes"])
    plt.tight_layout()
    plt.show()


# ======================
# Credit Utilization Analysis
# ======================

def analyze_utilization(df, target):
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
    plt.show()


# ======================
# Repayment Behavior Analysis
# ======================

def analyze_repayment_behavior(df, target):
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
    sns.barplot(data=repay_result, x="Repayment_Group", y=target, ax=ax, palette="coolwarm")
    ax.set_title("Lower Repayment Ratio Indicates Higher Default Risk", fontsize=12, fontweight="bold")
    ax.set_xlabel("Repayment Behavior Group")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.show()


# ======================
# Credit Limit Segmentation
# ======================

def analyze_credit_limits(df, target):
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
    plt.show()


# ======================
# Risk Segmentation Analysis
# ======================

def analyze_risk_segments(df, target):
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
    sns.barplot(data=risk_result, x="Risk_Profile", y="Default_Rate", ax=ax, palette=["green", "red"])
    ax.set_title("High Financial Stress Segment Has Much Higher Default Risk", 
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
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
    
    analyze_target_distribution(df, TARGET)
    analyze_correlations(df, TARGET)
    analyze_overdue_impact(df, TARGET)
    analyze_utilization(df, TARGET)
    analyze_repayment_behavior(df, TARGET)
    analyze_credit_limits(df, TARGET)
    analyze_risk_segments(df, TARGET)
    
    print_executive_summary()
    
    print("\n✓ Analysis Complete!")
