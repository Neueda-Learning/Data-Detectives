"""
╔════════════════════════════════════════════════════════════════╗
║         CREDIT CARD DEFAULT DATASET - COMPREHENSIVE EDA        ║
║              Taiwan Credit Card Default Analysis               ║
║         Risk Analytics Workflow (10-Step Process)              ║
╚════════════════════════════════════════════════════════════════╝

Philosophy:
-----------
• NO pre-built hypotheses
• Let DATA reveal patterns naturally
• Observations ONLY (no causal claims)
• Business evidence-driven insights
• Modular, reusable code

Steps:
------
1. Dataset Understanding
2. Distribution Analysis (Numerical Variables)
3. Categorical Variables Exploration
4. Variable Quality Assessment
5. Target Variable Relationships
6. Feature Interactions
7. Correlation Analysis
8. Customer Segmentation
9. Anomaly & Pattern Discovery
10. Business Insights Generation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.sans-serif'] = ['STHeiti', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

NUMERICAL_FEATURES = ['X1', 'X5', 'Avg_Bill', 'Avg_Pay', 
                      'Repayment_Ratio', 'Utilization_Rate']
CATEGORICAL_FEATURES = ['X2', 'X3', 'X4', 'Ever_Overdue']
TARGET = 'default_payment'

# ═══════════════════════════════════════════════════════════════
# STEP 1: DATASET UNDERSTANDING
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load the credit card dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "cleaned_credit_card_data.csv")
    df = pd.read_csv(csv_path)
    return df


def step1_dataset_understanding(df):
    """
    STEP 1: Understand the raw dataset
    
    This is the foundation of all analysis.
    We establish what we're working with:
    - Dataset dimensions
    - Data types
    - Missing values
    - Duplicates
    - Basic statistics
    
    Why: Prevents downstream errors and reveals data quality issues.
    """
    print("\n" + "="*70)
    print("STEP 1: DATASET UNDERSTANDING")
    print("="*70)
    
    # 1.1 Basic Information
    print("\n📊 BASIC INFORMATION")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # 1.2 Data Types
    print("\n📋 DATA TYPES")
    print(f"   Numerical: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"   Categorical: {df.select_dtypes(include=['object']).shape[1]}")
    
    # 1.3 Missing Values
    print("\n⚠️  MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values detected")
    else:
        print(missing[missing > 0])
    
    # 1.4 Duplicates
    print("\n🔄 DUPLICATES")
    duplicates = df.duplicated().sum()
    print(f"   Complete duplicates: {duplicates}")
    
    # 1.5 Target Variable
    print("\n🎯 TARGET VARIABLE: default_payment")
    target_dist = df[TARGET].value_counts()
    target_pct = df[TARGET].value_counts(normalize=True) * 100
    for cls in sorted(df[TARGET].unique()):
        print(f"   Class {cls}: {target_dist[cls]:,} ({target_pct[cls]:.1f}%)")
    
    # 1.6 Descriptive Statistics
    print("\n📈 DESCRIPTIVE STATISTICS")
    print(df.describe().to_string())
    
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: DISTRIBUTION ANALYSIS - NUMERICAL VARIABLES
# ═══════════════════════════════════════════════════════════════

def analyze_numerical_distribution(df, feature):
    """
    Analyze a single numerical feature:
    - Skewness
    - Kurtosis
    - Outliers
    - Distribution shape
    """
    stats = {
        'mean': df[feature].mean(),
        'median': df[feature].median(),
        'std': df[feature].std(),
        'min': df[feature].min(),
        'max': df[feature].max(),
        'skewness': df[feature].skew(),
        'kurtosis': df[feature].kurtosis(),
        'q25': df[feature].quantile(0.25),
        'q75': df[feature].quantile(0.75),
    }
    
    # Detect outliers using IQR method
    IQR = stats['q75'] - stats['q25']
    lower_bound = stats['q25'] - 1.5 * IQR
    upper_bound = stats['q75'] + 1.5 * IQR
    outlier_count = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
    outlier_pct = (outlier_count / len(df)) * 100
    stats['outlier_count'] = outlier_count
    stats['outlier_pct'] = outlier_pct
    
    return stats


def plot_numerical_distribution(df, features, max_cols=3):
    """
    Create distribution plots for numerical features.
    Includes: histogram + KDE + box plot
    """
    n_features = len(features)
    n_rows = (n_features + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Histogram + KDE
        ax.hist(df[feature], bins=30, alpha=0.7, color='#3498db', edgecolor='black', density=True)
        df[feature].plot(kind='kde', ax=ax, color='#e74c3c', linewidth=2.5)
        
        # Statistics
        stats = analyze_numerical_distribution(df, feature)
        title = f"{feature}\n"
        title += f"Mean={stats['mean']:.0f} | Median={stats['median']:.0f} | "
        title += f"Skew={stats['skewness']:.2f} | Kurt={stats['kurtosis']:.2f}"
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig, stats


def step2_distribution_analysis(df):
    """
    STEP 2: Explore distributions of numerical variables
    
    Purpose:
    - Identify distribution shapes (normal, skewed, bimodal, etc.)
    - Detect outliers
    - Find variables with unusual patterns
    - Determine which variables need transformation
    
    Why: Distributions reveal data quality issues and transformation needs.
    """
    print("\n" + "="*70)
    print("STEP 2: DISTRIBUTION ANALYSIS - NUMERICAL VARIABLES")
    print("="*70)
    
    print("\n📊 Analyzing distributions...")
    
    # Select numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != TARGET]
    
    # Show statistics table
    print("\n📈 STATISTICAL SUMMARY")
    print("─" * 100)
    print(f"{'Feature':<20} {'Mean':>12} {'Median':>12} {'Std':>12} {'Skew':>10} {'Kurt':>10} {'Outliers':>12}")
    print("─" * 100)
    
    for feature in numerical_cols:
        stats = analyze_numerical_distribution(df, feature)
        print(f"{feature:<20} {stats['mean']:>12.0f} {stats['median']:>12.0f} "
              f"{stats['std']:>12.0f} {stats['skewness']:>10.2f} {stats['kurtosis']:>10.2f} "
              f"{stats['outlier_count']:>6.0f} ({stats['outlier_pct']:>4.1f}%)")
    
    # Create distribution plots
    fig, _ = plot_numerical_distribution(df, numerical_cols)
    plt.savefig('02_numerical_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 02_numerical_distributions.png")
    plt.show()
    
    # Auto-detect interesting variables
    print("\n🔍 KEY OBSERVATIONS")
    print("─" * 70)
    
    for feature in numerical_cols:
        stats = analyze_numerical_distribution(df, feature)
        observations = []
        
        if abs(stats['skewness']) > 1:
            observations.append(f"Highly skewed ({stats['skewness']:.2f})")
        if stats['outlier_pct'] > 5:
            observations.append(f"High outlier rate ({stats['outlier_pct']:.1f}%)")
        if stats['kurtosis'] > 3:
            observations.append("Heavy tails (high kurtosis)")
        
        if observations:
            print(f"   {feature}: {', '.join(observations)}")
    
    return numerical_cols


if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Execute steps
    df = step1_dataset_understanding(df)
    numerical_cols = step2_distribution_analysis(df)
    
    print("\n" + "="*70)
    print("✓ STEP 1-2 COMPLETE")
    print("="*70)
    print("\nNext: Run analysis_step3_to_10.py for remaining steps")
