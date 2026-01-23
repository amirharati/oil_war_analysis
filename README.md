# Oil Market Geopolitical Risk Analysis

This project analyzes the relationship between oil market features (Brent-WTI spread, volatility, returns) and geopolitical periods. It uses machine learning to assess the risk of pre-war conditions based on market patterns.

## Overview

We analyze how oil market conditions change during different geopolitical periods (normal times, pre-war, war, post-war) and use this to assess risk for future periods.

**Key Hypothesis**: During 2024, I hypothesized that if a relatively large-scale war happens involving Iran, it should affect not only the oil price but also the spread between WTI and Brent prices, as war in the Persian Gulf should add some premium for WTI, so the spread should shrink. Later I also added volatility to this hypothesis, as these features also show less certainty and more risk. The final analysis includes both spread and price returns and their volatility. 

## Notebooks

### 📊 Notebook 1: `01_analysis.ipynb` - Data Preparation and Exploration

**Purpose**: Downloads data, creates features, and explores patterns across geopolitical periods.

**Key Sections**:
1. **Data Download** (Cell 1): Downloads hourly Brent/WTI prices from OANDA API (requires `.env` with API credentials)
2. **Feature Engineering** (Cell 2): Creates 10 features:
   - Spread (Brent - WTI) and its volatility (24h, 5d, 30d)
   - WTI price returns (1d, 5d, 30d) and volatility (24h, 5d, 30d)
3. **Labeling** (Cell 2): Assigns geopolitical labels based on dates (Normal, Pre-War Early/Close, War, Post-War Close/Late, Unlabeled)
4. **Z-Score Calculation** (Cell 4): Standardizes features (rolling and global z-scores) for comparison
5. **Visualizations** (Cells 3, 5-8): Time series and scatter plots showing patterns by period
6. **PCA** (Cell 9): Dimensionality reduction to identify main patterns
7. **Data Export** (Cell 10): Saves processed data for machine learning

**Key Concepts**:
- **Z-Scores**: Standardized values showing how many standard deviations a feature is from average
- **Rolling Z-Score**: Compares to recent 30-day average (captures short-term deviations)
- **Global Z-Score**: Compares to entire dataset average (captures long-term deviations)

---

### 🤖 Notebook 2: `02_simple_model.ipynb` - Risk Assessment

**Purpose**: Trains ML models and assesses geopolitical risk for future periods.

**Key Sections**:
1. **Model Training** (Cells 2-3): Trains Random Forest, Logistic Regression, and SVM on 2024-2025 data
2. **Predictions** (Cell 6): Predicts labels for 2026 data using all models
3. **Pre-War Risk** (Cell 6): Calculates probability of pre-war state (sum of Pre-War Early + Close probabilities)
4. **Mahalanobis Distance** (Cell 7): Measures how unusual current conditions are vs. stable periods
5. **Combined Risk** (Cell 7): Unified risk score combining anomaly detection + model predictions
6. **Additional Insights** (Cell 8): Model agreement, feature contributions, trends, correlations

**Key Concepts**:
- **Pre-War Risk Probability**: ML model's estimate of pre-war likelihood (0-100%)
- **Mahalanobis Distance**: Statistical measure of anomaly from "normal" baseline
- **Combined Risk Score**: Average of normalized Mahalanobis + pre-war probability
- **Model Agreement**: How often all models predict the same label (higher = more confident)

---

## Key Concepts and Meanings

### Risk Indicators Explained

1. **Pre-War Risk Probability** (0-100%)
   - **What it is**: ML models' estimate that current market conditions match pre-war patterns
   - **How calculated**: Sum of "Pre-War Early" + "Pre-War Close" probabilities from all models, taking the maximum (conservative)
   - **What it means**: Higher values = models see features similar to historical pre-war periods
   - **Limitation**: Based on patterns learned from 2024-2025, may not generalize perfectly

2. **Mahalanobis Distance** (0 to ∞)
   - **What it is**: Statistical measure of how "unusual" current conditions are
   - **How calculated**: Distance from "Normal + Post-War Late" baseline (stable periods)
   - **What it means**: Higher = more anomalous/unusual market behavior compared to stable times
   - **Why useful**: Catches anomalies that models might miss (complementary signal)

3. **Combined Risk Score** (0-100%)
   - **What it is**: Unified risk indicator combining both anomaly and prediction signals
   - **How calculated**: Average of (normalized Mahalanobis distance + pre-war probability)
   - **What it means**: Higher = both statistical anomaly AND model prediction agree on risk
   - **Best use**: When both indicators are high, risk signal is stronger

### Important Distinctions

- **Risk Assessment vs. Prediction**: This tool assesses risk based on patterns, not a crystal ball
- **Pattern-Based**: Models learn from 2024-2025 labeled data; 2026 is unlabeled test data
- **Complementary Signals**: Mahalanobis (statistical anomaly) and pre-war probability (learned patterns) measure different aspects
- **Negative Correlation**: The two indicators often disagree, which is expected - they capture different types of risk signals

---

## Key Observations and Conclusions

### What the Analysis Shows

1. **Pattern Recognition Works**
   - Models achieve 99%+ training accuracy, showing clear patterns exist between market features and geopolitical periods
   - Different periods (Normal, Pre-War, War, Post-War) show distinct feature characteristics
   - PCA visualization shows periods can be separated in reduced-dimensional space

2. **Pre-War Periods Are Anomalous**
   - Pre-war periods show high Mahalanobis distances compared to stable (Normal/Post-War Late) baseline
   - This confirms that pre-war market conditions are statistically unusual
   - Current 2026 data shows similar anomaly levels to historical pre-war periods in some cases

3. **Models and Anomaly Detection Complement Each Other**
   - Negative correlation (-0.234) between Mahalanobis and pre-war probability suggests they measure different things
   - This is actually good: they provide complementary risk signals
   - Combined risk score integrates both perspectives for a more complete assessment

4. **2026 Risk Assessment**
   - Current models predict mostly "Normal" for 2026 data (low pre-war probability)
   - However, Mahalanobis distance shows some days are highly anomalous
   - This suggests: either models haven't seen these patterns before, OR market is in a new regime
   - Combined risk score helps balance both signals

5. **Feature Relationships**
   - Spread, volatility, and returns all show different patterns across geopolitical periods
   - Scatter plots reveal clustering by period, confirming relationships exist
   - Feature importance analysis shows which features drive predictions most

### Limitations and Caveats

- **Not Perfect Prediction**: Models trained on 2024-2025 may not capture all future scenarios
- **Limited Training Data**: War period is small (10 days), making it harder to learn
- **2026 is Unlabeled**: We can't verify if predictions are correct (no ground truth)
- **Pattern-Based**: Relationship exists but may not be causal or deterministic

### Practical Value

Despite limitations, this analysis provides:
- **Early Warning System**: Identifies when market conditions deviate from normal
- **Risk Quantification**: Provides numerical risk scores for decision-making
- **Pattern Recognition**: Confirms relationships between market features and geopolitical periods
- **Multi-Signal Approach**: Combines statistical anomaly detection with ML pattern recognition

**Bottom Line**: While not a perfect predictor, this framework provides a useful risk assessment tool that identifies when market conditions suggest elevated geopolitical risk based on learned patterns and statistical anomalies.



## Quick Start

1. **Setup**: Create `.env` file with `OANDA_API_KEY` and `OANDA_ACCOUNT_ID`
2. **Run Notebook 1**: Downloads data, creates features, saves processed data
3. **Run Notebook 2**: Trains models, makes predictions, assesses risk

## Interpreting Results

- **Pre-War Risk > 50%**: Models see strong pre-war pattern match
- **Mahalanobis > 95th percentile**: Market conditions are highly unusual
- **Combined Risk > 50%**: Both signals agree on elevated risk
- **Trend Analysis**: Check if risk is increasing (building up) or decreasing

## Requirements

- Python 3.11+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy
- python-dotenv, requests
- OANDA API credentials

## Output

The notebooks produce:
- Processed data files
- Visualizations of features and predictions
- Risk assessments for 2026 data
- Statistical comparisons and insights

All results are displayed in the notebooks and saved to the `data/` directory.
