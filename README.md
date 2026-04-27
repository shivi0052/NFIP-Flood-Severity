# 🌊 NFIP Flood Claim Severity Prediction
### ISOM 835 – Predictive Analytics and Machine Learning | Suffolk University MSBA | Spring 2026

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-FF6600?logo=xgboost)](https://xgboost.readthedocs.io)
[![FEMA NFIP](https://img.shields.io/badge/Dataset-FEMA_NFIP-0052A5)](https://www.fema.gov/flood-insurance/work-with-nfip/claim)
[![R²](https://img.shields.io/badge/R²-0.669-22c55e)](#model-results)
[![MAE](https://img.shields.io/badge/MAE-%2416%2C474-22c55e)](#model-results)

---


## 🌐 Project Website

A single-file interactive website summarizing the full project is available at:

> **https://shivi0052.github.io/NFIP-Flood-Severity/**

---

## Project Summary

This project builds an **end-to-end machine learning pipeline** to predict the severity of individual flood insurance claims filed with the National Flood Insurance Program (NFIP), managed by FEMA. Given a flood claim record — including property characteristics, flood zone, water depth, and building type — the model predicts how much NFIP will pay on the building coverage.

**Best model:** XGBoost (manually tuned) · **R² = 0.669** · **MAE = $16,474** on 380K held-out claims.

The complete analysis covers 1,989,613 historical flood insurance claims, spans data cleaning → feature engineering → EDA → three predictive models → SHAP interpretation → three actionable business recommendations.

---

## Project Objectives

1. **Predict** the dollar amount NFIP pays on building coverage for an incoming flood claim, using only information available at claim submission.
2. **Identify** the structural, geographic, and hazard features that most strongly drive claim severity , validated by both domain knowledge and SHAP analysis.
3. **Demonstrate** why non-linear tree-based models are necessary for this problem (spoiler: conditional interactions that linear models cannot capture explain the 2× performance gap between XGBoost and Linear Regression).
4. **Translate** model findings into three concrete, deployment-ready recommendations for FEMA's claims operations.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [FEMA NFIP Redacted Claims V2](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) |
| **Records** | 1,989,613 historical flood insurance claims |
| **Original columns** | 73 |
| **Target variable** | `amountPaidOnBuildingClaim` — actual NFIP building payout ($0.01 – $250,000) |
| **Time span** | Multi-decade, all 50 U.S. states + territories |
| **Format** | Parquet (`FimaNfipClaimsV2.parquet`) |

> **Note:** The dataset file is not included in this repository due to its size (~1GB). Download directly from the [OpenFEMA Data Page](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) and place `FimaNfipClaimsV2.parquet` in the root directory before running the notebook.

---

## Repository Structure

```
nfip-flood-severity/
│
├── report/
│   └── NFIP_Final_Report.md         ← 10-page final report
│
├── visualizations/
│   ├── Building Vulnerability.png
│   ├── Correlation Analysis.png
│   ├── Feature Importance.png
│   ├── Flood Zone Risk.png
│   ├── Geographic Concentration of NFIP Flood Losses.png
│   ├── Hazard Intensity.png
│   ├── High Severity vs Low Severity Claim Profiles.png
│   ├── Model Performance Comparison.png
│   ├── SHAP Analysis.png
│   ├── Seasonality of NFIP Flood Claims.png
│   ├── Understanding Our Target.png
│   └── XG Boost Model Diagnostic.png
│
├── NFIP_Complete_Analysis_v3.ipynb  ← Main analysis notebook (Colab)
├── README.md                        ← You are here
├── .gitignore
└── index.html                       ← Project website (GitHub Pages)
```

---

## 🔗 Google Colab Notebook

> **➡️ [Open in Google Colab](https://colab.research.google.com/drive/1myySw-avWrTSmwQrTm7ytpGIHBkwBy3n?usp=sharing)**

### How to Run

1. Open the Colab notebook via the link above.
2. Download `FimaNfipClaimsV2.parquet` from [OpenFEMA](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2).
3. Upload the parquet file to your Colab session storage (or mount Google Drive).
4. Run all cells top-to-bottom (`Runtime → Run all`).
5. Expected total runtime: ~25–35 minutes (Random Forest and XGBoost training are the slowest steps).

---

## Notebook Structure

| Part | Content | Key Output |
|---|---|---|
| 🔵 **Part 1 – The Situation** | Data loading, quality audit, target variable cleaning | 1.99M valid records, 30 columns dropped |
| 🟡 **Part 2 – The Discovery** | 22 engineered features + 8 EDA questions | Depth-damage curve, freeboard threshold, geographic concentration |
| 🟣 **Part 3 – The Model** | Linear Regression → Random Forest → XGBoost + Optuna | XGBoost R² = 0.669, MAE = $16,474 |
| 🟢 **Part 4 – The Recommendation** | SHAP values, diagnostics, business recommendations | 3 deployment-ready use cases |

---

## Model Results

| Model | R² | RMSE | MAE | Notes |
|---|---|---|---|---|
| Linear Regression | 0.307 | $39,888 | $25,384 | Baseline — fails on non-linear interactions |
| Random Forest | 0.551 | $32,116 | $19,777 | +80% R² improvement vs. LR |
| **XGBoost v1 ✅** | **0.669** | **$27,577** | **$16,474** | **Final model — manually tuned** |
| XGBoost v2 (Optuna) | 0.666 | $27,694 | $16,564 | Automated search — slightly underperforms manual |

**Prediction accuracy on held-out test set:**
- 35% of predictions within **$5,000** of actual
- 67% of predictions within **$15,000** of actual
- 80% of predictions within **$25,000** of actual
- Median error: **+$1,822** (near-zero systematic bias)
- Train R² = 0.7254 · Test R² = 0.6688 · Gap = **0.057** (no overfitting)

---

## Top 5 Drivers of Flood Claim Severity (SHAP-validated)

| Rank | Feature | Importance | What It Tells Us |
|---|---|---|---|
| 1 | `cat_year_coastal` | 0.2676 | Catastrophe year × coastal state — Katrina (2005), Harvey (2017), Sandy (2012) |
| 2 | `elevatedBuildingIndicator` | 0.0632 | #1 structural predictor — elevated buildings pay ~10% less on average |
| 3 | `waterDepth` | 0.0478 | Non-linear depth-damage relationship — every inch matters differently |
| 4 | `buildingPropertyValue` | 0.0339 | Higher value = larger absolute dollar loss |
| 5 | `longitude` | 0.0275 | Gulf Coast vs. inland geographic exposure |

---

## Key EDA Findings

- **Target distribution:** Severely right-skewed (median ~$15K, mean ~$33K). Log-transform justified.
- **Temporal:** Claims spike in Aug–Oct (hurricane season). 2005/2012/2017 are visible catastrophe-year outliers.
- **Water depth:** Classic non-linear depth-damage curve — motivates `water_depth_log` feature.
- **Flood zones:** Zone X generates the most claim *volume* despite being low-risk ("the volume paradox").
- **Elevation:** Crossing above the Base Flood Elevation cuts median claims by **63%** — threshold effect, not linear.
- **Geography:** LA + FL + TX = **63.3% of all NFIP building claim payments** (just 3 of 50 states).
- **Correlation:** `freeboard_positive` shows −0.078 linear correlation but is a top SHAP contributor — its effect is *conditional*, which linear models cannot detect.

---

## 📊 Key Visualizations

**Target Variable Distribution**
![Target Distribution](visualizations/Understanding%20Our%20Target.png)

**Hazard Intensity & Water Depth**
![Hazard Intensity](visualizations/Hazard%20Intensity.png)

**Geographic Concentration of Flood Losses**
![Geography](visualizations/Geographic%20Concentration%20of%20NFIP%20Flood%20Losses.png)

**Model Performance Comparison**
![Model Comparison](visualizations/Model%20Performance%20Comparison.png)



## Tools & Libraries

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | `pandas`, `numpy`, `pyarrow` |
| Visualization | `matplotlib`, `seaborn` |
| Modeling | `scikit-learn`, `xgboost` |
| Interpretation | `shap` |
| Tuning | `optuna` |
| Platform | Google Colab, GitHub |

---

## Business Questions Answered

1. **What property and hazard characteristics most strongly predict the size of an NFIP flood claim?** → `cat_year_coastal`, elevation status, and water depth dominate. Geography encodes catastrophe exposure the model cannot otherwise see.

2. **Why does XGBoost dramatically outperform Linear Regression on this dataset?** → Flood damage is governed by conditional interactions (elevation only matters in certain zones at certain depths). Linear models average across these conditions; XGBoost discovers them through tree splits.

3. **Can the model be used operationally to improve FEMA's claims process?** → Yes, in three ways: real-time claims triage, pre-season portfolio risk screening, and property-specific elevation valuation.



## Ethical Considerations

This model uses geography (`state`, `longitude`, `is_coastal_state`) as predictive features. Because flood risk is genuinely driven by coastal proximity and regional catastrophe exposure, geographic features are physically justified — not proxies for protected demographic characteristics. However, any deployment must ensure the model does not inadvertently compound existing inequities in disaster recovery. Full ethics discussion in the [Final Report](report/NFIP_Final_Report.md).

---

*ISOM 835 – Predictive Analytics and Machine Learning · Suffolk University MSBA · Spring 2026*
*Instructor: Hasan Arslan*
