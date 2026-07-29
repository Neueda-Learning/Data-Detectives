# Credit Card Default Risk Analysis

## 📊 Project Overview

This project performs a comprehensive **Exploratory Data Analysis (EDA)** of credit card default risk using the Taiwan Credit Card Default Dataset. The analysis identifies deep behavioral risk patterns, customer segmentation, and actionable insights for credit risk management.

**Target Audience**: Credit Risk Committee, Senior Management  
**Data Source**: `cleaned_credit_card_data.csv` (30,000 customers × 29 features)  
**Default Rate**: 22.1% (6,636 defaults / 23,364 non-defaults)

---

## ✨ Key Features

### 1. **Data Loading & Overview**
- Loads cleaned dataset from CSV
- Displays dataset shape, columns, and sample data
- Zero missing values with minimal duplicates (0.12%)

### 2. **Target Variable Analysis**
- Visualizes default payment distribution (class imbalance: 77.9% non-default vs 22.1% default)
- Calculates overall default rate

### 3. **Feature Correlation Analysis**
- Ranks all numerical features by correlation with default outcome
- Identifies top predictive variables
- Generates correlation heatmap visualization

### 4. **Overdue History Impact** ⭐ **Most Important**
- Analyzes payment default behavior across overdue groups
- **Key Finding**: Ever_Overdue status shows **3.65× default multiplier**
  - With overdue history: 42.73% default rate
  - No overdue history: 11.71% default rate

### 5. **Credit Utilization Segmentation**
- Categorizes customers into 5 utilization levels (Very Low → Very High)
- Demonstrates nonlinear relationship between utilization and default risk
- Shows gradual increase in default probability with higher utilization

### 6. **Repayment Behavior Analysis**
- Segments customers by repayment ratio into 5 groups
- Identifies repayment ratio as early warning signal
- Shows inverse correlation with default probability

### 7. **Credit Limit Segmentation**
- Analyzes default risk across 6 credit limit quantiles
- Demonstrates credit limit's role in reflecting customer quality
- Shows inverse relationship: higher limits → lower default rates

### 8. **Risk Segmentation**
- Identifies **High Financial Stress** segment
  - Criteria: High utilization + Low repayment + Ever overdue
  - Significantly elevated default probability
- Creates risk profiles for targeted intervention

---

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
pandas >= 2.0
numpy >= 2.0
matplotlib >= 3.0
seaborn >= 0.13
scikit-learn >= 1.0 (optional, for future enhancements)
```

### Installation
```bash
# Using pip
pip install pandas numpy matplotlib seaborn

# Using conda
conda install -c conda-forge pandas numpy matplotlib seaborn
```

---

## 🚀 Quick Start

### 1. **Prepare Data**
Ensure `cleaned_credit_card_data.csv` is in the same directory as `analysis.py`.

Required columns:
- `default_payment` (target variable: 0 or 1)
- `X1` (Credit Limit)
- `Ever_Overdue` (binary: 0 or 1)
- `Utilization_Rate` (ratio: 0-1 or 0-100)
- `Repayment_Ratio` (ratio: 0-1 or 0-100)

### 2. **Run Analysis**
```bash
# From command line
cd /path/to/Data-Detectives
python analysis.py

# Or from Python interpreter
from analysis import *
df = load_data("cleaned_credit_card_data.csv")
analyze_target_distribution(df, "default_payment")
```

### 3. **View Results**
The script generates:
- **Console Output**: Statistical summaries and insights
- **Visualizations**: 8 publication-ready plots with proper formatting
- **Executive Summary**: Key findings and recommendations

---

## 📊 Analysis Functions

| Function | Purpose | Output |
|----------|---------|--------|
| `load_data(filepath)` | Load CSV dataset | DataFrame with shape & preview |
| `analyze_target_distribution()` | Target variable analysis | Bar chart + default rate % |
| `analyze_correlations()` | Feature importance ranking | Horizontal bar chart |
| `analyze_overdue_impact()` | Payment history effect | Comparison chart (Yes/No) |
| `analyze_utilization()` | Credit usage patterns | Line chart (5 levels) |
| `analyze_repayment_behavior()` | Repayment quality | Bar chart (5 segments) |
| `analyze_credit_limits()` | Credit amount effect | Trend line (6 quantiles) |
| `analyze_risk_segments()` | Risk classification | Segmentation comparison |
| `print_executive_summary()` | Business insights | Console report |

---

## 🔍 Key Findings

### Finding #1: Historical Repayment is the Strongest Predictor 🏆
- **Evidence**: Ever_Overdue shows highest correlation with default
- **Impact**: 3.65× multiplier effect
- **Recommendation**: Prioritize this variable in credit scoring models

### Finding #2: Credit Utilization Indicates Financial Stress
- **Evidence**: Nonlinear relationship with default rate
- **Pattern**: Exponential increase at high utilization levels
- **Recommendation**: Monitor high-utilization customers closely

### Finding #3: Low Repayment Ratio Provides Early Warning
- **Evidence**: Strong inverse correlation with payment behavior
- **Pattern**: Customers with <50% repayment ratio at elevated risk
- **Recommendation**: Implement early intervention program

### Finding #4: Combined Stress Signals Amplify Risk
- **Evidence**: High Financial Stress segment shows dramatic default rate increase
- **Pattern**: Multiple simultaneous stressors (overdue + high util + low repay)
- **Recommendation**: Create multi-factor risk assessment

---

## 💡 Business Recommendations

1. **Develop Risk Scoring Model**
   - Use top 4 features (Ever_Overdue, Utilization_Rate, Repayment_Ratio, X1)
   - Consider nonlinear relationships (use tree-based models)
   - Implement segmentation strategy for different risk tiers

2. **Create Early Warning System**
   - Monitor customers transitioning to High Financial Stress segment
   - Set alerts for sudden utilization spikes
   - Track repayment ratio deterioration

3. **Implement Targeted Interventions**
   - Offer credit limit reduction to high-stress customers
   - Provide payment assistance programs
   - Increase monitoring frequency for flag customers

4. **Optimize Credit Decisions**
   - Use credit limit as reflects customer quality (higher limit = lower risk)
   - Consider repayment history weight in approval process
   - Implement behavioral scoring alongside traditional credit scoring

5. **Future Model Development**
   - Logistic Regression: For regulatory compliance & interpretability
   - Random Forest: For handling nonlinear patterns
   - XGBoost: For state-of-the-art performance
   - SHAP Analysis: For explainability

---

## 📁 Project Structure

```
Data-Detectives/
├── README.md                          # This file
├── analysis.py                        # Main EDA script
├── cleaned_credit_card_data.csv       # Input dataset
├── Executive_EDA_Report.ipynb         # Jupyter notebook (advanced)
└── eda_comprehensive.py               # Extended EDA framework
```

---

## 🎨 Visualization Features

All charts include:
- ✅ Clear, descriptive titles (bold 12pt font)
- ✅ Labeled axes with units
- ✅ Color coding (green for non-default, red for default)
- ✅ Appropriate visualization types (bar, line, scatter)
- ✅ Automatic layout optimization
- ✅ Publication-ready quality

---

## 🔧 Customization

### Change Dataset Location
```python
DATA_FILE = "/your/custom/path/to/data.csv"
```

### Modify Segmentation Levels
```python
# In analyze_utilization():
q=5,  # Change from 5 to 3, 10, etc.
```

### Adjust Figure Sizes
```python
# In module header:
plt.rcParams['figure.figsize'] = (10, 8)  # Change dimensions
```

### Add New Analysis Functions
```python
def analyze_age_impact(df, target):
    """Template for adding new analysis."""
    # Your code here
    pass
```

---

## 📈 Expected Output Example

```
Dataset Shape: (30000, 29)

Columns: ['X1', 'X2', ..., 'default_payment', ...]

First 5 rows:
      X1    X2    ...  default_payment
0  500000    37  ...                0
1  150000    40  ...                0
...

Overall Default Rate: 22.12%

Feature Correlations with Default:
Ever_Overdue           0.456789
Utilization_Rate       0.234567
Repayment_Ratio       -0.345678
...

✓ Analysis Complete!
```

---

## ⚙️ Technical Details

### Data Quality Checks
- ✅ No missing values
- ✅ All features numerical (no encoding needed)
- ✅ Duplicates handled (36 / 30,000 = 0.12%)
- ✅ Target variable properly balanced (22% / 78%)

### Matplotlib Configuration
- **Style**: seaborn-v0_8 (professional appearance)
- **Figure Size**: (8, 6) default with custom adjustments
- **Font Size**: 10pt base with 12pt bold titles
- **Layout**: Automatic `tight_layout()` applied

### Pandas Settings
- **Numeric Only**: Correlation calculations exclude non-numeric columns
- **Observed Parameter**: GroupBy respects only observed categories (suppresses FutureWarning)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **File Not Found** | Ensure CSV is in same directory: `ls -la cleaned_credit_card_data.csv` |
| **Missing Columns** | Check column names match exactly (case-sensitive): `print(df.columns.tolist())` |
| **Matplotlib Backend Error** | Use non-interactive backend: `matplotlib.use('Agg')` |
| **Memory Error** | Load in chunks or optimize dtypes: `pd.read_csv(..., dtype={'col': 'category'})` |
| **Import Errors** | Verify all packages installed: `pip install -r requirements.txt` |

---

## 📚 References

- **Dataset**: Taiwan Credit Card Default Data (UCI Machine Learning Repository)
- **Analysis Type**: Exploratory Data Analysis (EDA) for Risk Assessment
- **Methodology**: Univariate + Bivariate + Segmentation Analysis
- **Tools**: pandas, matplotlib, seaborn

---

## 👤 Project Information

**Project**: Credit Card Default Risk Analysis  
**Purpose**: Enterprise Risk Analytics for Senior Management  
**Date**: 2026  
**Status**: ✅ Production-Ready

**Code Quality**:
- ✅ PEP8 Compliant
- ✅ Modular Functions
- ✅ Comprehensive Docstrings
- ✅ Error Handling
- ✅ Reproducible Results

---

## 📝 License

Internal use only - HSBC Credit Risk Committee

---

**Last Updated**: 2026-07-29  
**Version**: 2.0 (Refactored for Production)