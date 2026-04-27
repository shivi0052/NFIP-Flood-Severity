# NFIP Flood Claim Severity Prediction
## Final Project Report — ISOM 835: Predictive Analytics and Machine Learning

---

**Course:** ISOM 835 – Predictive Analytics and Machine Learning  
**Name:** Shivani  
**Instructor:** Hasan Arslan  
**Institution:** Suffolk University, Sawyer Business School  
**Program:** Master of Science in Business Analytics (MSBA)  
**Term:** Spring 2026  

---

## Table of Contents

1. [Introduction & Dataset Description](#1-introduction--dataset-description)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Data Cleaning & Preprocessing](#3-data-cleaning--preprocessing)
4. [Business Questions](#4-business-questions)
5. [Predictive Modeling](#5-predictive-modeling)
6. [Insights & Recommendations](#6-insights--recommendations)
7. [Ethics & Interpretability](#7-ethics--interpretability)
8. [Appendix](#8-appendix)

---

## 1. Introduction & Dataset Description

### 1.1 Background and Motivation

The National Flood Insurance Program (NFIP), administered by the Federal Emergency Management Agency (FEMA), is the primary source of flood insurance coverage in the United States. With over five million active policies and more than **$70 billion in historical claim payments**, the NFIP represents one of the largest government-backed insurance programs in the world. Floods are the most frequent and costly natural disasters in the U.S., and the NFIP exists to provide affordable coverage where private insurers typically will not.

Accurately predicting individual flood claim severity — how much NFIP will pay on a building claim — is a high-value operational problem with direct applications in three areas: **claims triage** (prioritizing adjusters to the highest-value losses after a disaster), **reinsurance reserving** (estimating total expected payouts for catastrophe events), and **portfolio risk management** (identifying the highest-risk properties in the insured pool before they flood).

This project builds an end-to-end regression pipeline to address this prediction problem, using 1.99 million historical NFIP claims as training and evaluation data.

### 1.2 Dataset Description

The dataset used is the **FEMA NFIP Redacted Claims V2**, available via the OpenFEMA Data Initiative. It contains historical flood insurance building claims filed under the NFIP, covering multiple decades and all U.S. states and territories.

| Property | Value |
|---|---|
| Source | OpenFEMA — FIMA NFIP Redacted Claims V2 |
| Total Records | 1,989,613 |
| Original Columns | 73 |
| File Format | Parquet |
| Geographic Coverage | All 50 U.S. states + territories |
| Time Span | Multi-decade (1970s through 2022) |

**Target variable:** `amountPaidOnBuildingClaim` — the actual dollar amount NFIP paid for structural building damage, ranging from $0.01 to $250,000 (the NFIP maximum building coverage limit). This is a continuous regression target.

**Why this is a hard prediction problem:** Flood claim severity is governed by the simultaneous interaction of three dimensions — hazard intensity (how severe the flood was), building vulnerability (how exposed the structure was), and financial structure (the insurance policy terms). A shallow flood on a non-elevated Pre-FIRM building in a high-risk zone causes catastrophic losses. The exact same water depth on a Post-FIRM elevated building in a low-risk zone causes almost none. This conditionality means that linear models, which assume each feature contributes independently and additively, fail dramatically on this dataset.

---

## 2. Exploratory Data Analysis

Eight targeted EDA questions were formulated, each designed to directly inform a modeling decision. Charts for all eight are included in the `visualizations/` folder.

### EDA Q1: What Does the Target Variable Actually Look Like?

The raw distribution of `amountPaidOnBuildingClaim` is severely right-skewed, with a median of approximately $15,000 and a mean of $33,000. The distribution has a long right tail with extreme claims up to $250,000. The log-transformed distribution (`log1p`) is approximately bell-shaped, which is important for Linear Regression's error normality assumption (though XGBoost is robust to skewness). The frequency distribution confirms that small-to-moderate claims ($500–$15K) are the dominant class — a model that only predicts large losses accurately would fail on the majority of real-world claims.

**Modeling implication:** Retain the raw target for tree-based models (scale-invariant). Log-transform is applied as an exploration tool and for target-encoding features.

### EDA Q2: When Do Floods Cause the Most Severe Damage?

Claims are dramatically concentrated in August through October, matching the peak of the Atlantic hurricane season (June–November). The 2005 (Hurricane Katrina), 2012 (Hurricane Sandy), and 2017 (Hurricanes Harvey, Irma, Maria) years are visible as distinct spikes in both claim volume and average severity. The `is_hurricane_season` binary feature shows one of the strongest linear correlations with the target (Pearson r = 0.239).

**Modeling implication:** `is_hurricane_season` and `loss_month` are valid and important features. The catastrophe year pattern motivates `year_target_enc` (mean log-claim per year) as a target-encoded feature.

### EDA Q3: Does Water Depth Actually Predict Claim Size?

The relationship between water depth and claim amount is non-linear — the depth-damage curve is concave, meaning the first few inches of water cause a disproportionately large jump in losses, while marginal damage diminishes at extreme depths. This pattern is well-established in flood engineering literature. The log-transformed depth (`water_depth_log`) improves the linear correlation with the target from 0.156 to 0.248, directly justifying the engineered feature.

**Modeling implication:** Both `waterDepth` (for tree models that can discover non-linearity) and `water_depth_log` (for linear model signal) are retained.

### EDA Q4: Do FEMA's Flood Zones Actually Predict Losses?

High-risk zones (AE, VE — coastal high-hazard areas) do produce higher average severity. However, Zone X (the lowest-risk designation) generates the most total claim volume due to sheer policy count. This "volume paradox" means a model trained only on zone designation would misunderstand the portfolio: the average Zone X claim is small, but Zone X accounts for more total paid dollars than many high-risk zones simply because so many properties sit there. Physical hazard features (water depth, elevation) are essential complements to zone.

### EDA Q5: Do Elevation & Construction Era Reduce Losses?

Building-level interventions do reduce losses, but less cleanly than simple averages suggest. Non-elevated buildings average $33,535 vs. $30,410 for elevated buildings — a real but modest 10% difference. More importantly, **crossing above the Base Flood Elevation (BFE) cuts median claims by 63%** (from $29,000 to $11,000), revealing a threshold effect rather than a linear gradient. Post-FIRM buildings actually show higher average claims ($41,033) than Pre-FIRM ($30,270) due to higher insured values — newer buildings are worth more, so the same physical damage costs more to repair. This is a Total Insured Value (TIV) effect, not a vulnerability effect.

### EDA Q6: Where Are Flood Losses Geographically Concentrated?

Losses are dramatically concentrated geographically. Louisiana ($15.9B), Florida ($13.1B), and Texas ($12.5B) account for **63.3% of all NFIP building claim payments** despite representing only 3 of 50 states. Louisiana leads in claim volume (~360,000 claims at $44,000 average), while Florida leads in per-claim severity ($44,784 average). Mississippi has moderate volume but the third-highest average severity ($43,343), driven almost entirely by Hurricane Katrina (2005). New York and New Jersey appear in top-15 severity rankings due entirely to Hurricane Sandy (2012) — a single extreme event that inflates their per-claim average across the entire history.

### EDA Q7: How Are All Features Related to Each Other?

Individual feature-to-target correlations are modest (maximum ~0.25), which is normal for insurance loss data — the damage relationship is driven by interactions, not additive effects. The full correlation matrix reveals strong collinearity between elevation features (`height_above_BFE`, `freeboard_positive`, `elevationDifference` — correlation ~0.81), which would destabilize a linear model but is harmless for XGBoost. Critically, `freeboard_positive` shows only −0.078 linear correlation with the target, yet SHAP analysis in Part 4 reveals it as a top XGBoost contributor. This discrepancy is the core insight of the project: the protective effect of elevation is conditional on flood zone and water depth, and linear correlation cannot detect conditional effects.

### EDA Q8: What Does a High-Severity vs. Low-Severity Claim Look Like?

A comparison of the top 10% (high-severity) vs. bottom 30% (low-severity) claims reveals a physically sensible profile. High-severity claims average 8.55 inches of water depth vs. 2.21 inches for low-severity (3.9× more). High-severity claims are 3.5× more likely to be in high-risk zones (39.2% vs. 11.2%), 94.5% coastal vs. 71.9%, and have average total coverage of $317,000 vs. $113,000 (2.8×). Buildings average 39.9 years old vs. 27.4 years. These profiles are coherent with flood science — the model is learning real damage relationships, not statistical noise.

---

## 3. Data Cleaning & Preprocessing

### 3.1 Target Variable Cleaning

The target variable `amountPaidOnBuildingClaim` was stored as an object (text) type in the raw data. After conversion to numeric, three categories of invalid records were removed:

- **Negative values:** Impossible for an insurance payout (0 records removed)
- **Zero values:** No loss occurred; cannot model severity for non-events (removed ~800K+ records — these represent claims where NFIP paid nothing)
- **Values above $250,000:** Exceeds the NFIP building coverage ceiling; likely data entry errors

After filtering: **~1.9 million valid building claims retained.**

### 3.2 Column Removal

30 columns were dropped across four categories:

| Category | Columns Dropped | Rationale |
|---|---|---|
| Data leakage | `netBuildingPaymentAmount`, `netContentsPaymentAmount`, etc. | Derived from or parallel to the target — would inflate performance |
| Contents-related | `amountPaidOnContentsClaim`, `contentsDamageAmount`, etc. | Project scope is building claims only |
| Administrative identifiers | `id`, `asOfDate`, `ficoNumber`, etc. | No predictive value |
| Extremely sparse (>95% missing) | `nonPaymentReasonBuilding`, `floodCharacteristicsIndicator`, etc. | Cannot be reliably imputed |

### 3.3 Missing Value Strategy

A deliberate, domain-informed imputation strategy was applied:

- **Elevation columns:** Binary `has_<col>` missingness flags were created first (absence of an elevation certificate is itself a risk signal), then median fill was applied.
- **FEMA sentinel value:** `elevationDifference` uses 9999 to mean "not reported" — replaced with NaN before computing the median to avoid a polluted fill value.
- **Physical bounds:** `waterDepth` was clipped to a maximum of 72 inches (6 feet) to handle FEMA's mixed-unit recording; `elevationDifference` was clipped to ±30 feet.
- **Categorical columns:** Filled with `"Unknown"` — honest representation of genuinely missing data.
- **Model boundary guarantee:** Before any model receives data, a three-step imputation (force to numeric → drop all-NaN columns → fill remaining NaN with column medians) is applied, verified with an assertion.

**Residual missing values after Section 1.5:** 11,110 (< 0.015% of all data cells) — resolved entirely at the model boundary.

### 3.4 Feature Engineering

22 new features were engineered across five domain categories:

| Category | Features | Rationale |
|---|---|---|
| Building Age | `building_age_at_loss` | Older buildings predate NFIP flood codes; structurally more vulnerable |
| Elevation & Freeboard | `height_above_BFE`, `height_above_ground`, `freeboard_positive`, `freeboard_deficit` | Floor height above BFE is the primary physical determinant of flooding |
| Hazard Timing | `is_hurricane_season`, `water_depth_log` | Atlantic hurricane season (June–Nov); log depth matches depth-damage curve shape |
| Flood Zone | `flood_zone_risk_score`, `is_high_risk_zone` | FEMA zones encoded as ordinal risk hierarchy |
| Building Type & Geography | 8 `is_*` flags, `is_coastal_state`, `risk_zone_elevation_interaction`, `deductible_amount`, `cat_year_coastal`, `state_target_enc`, `year_target_enc` | Structure type, coastal exposure, catastrophe year × coastal interaction |

**Target encoding:** `state_encoded` (arbitrary integer) was replaced by `state_target_enc` (mean log-claim for that state) and `year_target_enc` (mean log-claim per loss year), giving the model direct geographic and temporal damage signals rather than meaningless integer labels.

### 3.5 Train/Test Split

An **80/20 random split** with `random_state=42` was used. With ~1.9M records, the test set contains ~380,000 claims — large enough that performance metrics are highly stable and not sensitive to split choice. Train and test mean claim amounts are nearly identical, confirming an unbiased split.

---

## 4. Business Questions

Three business questions were formulated to align the analysis with FEMA's operational needs:

### BQ1: What property and hazard characteristics most strongly predict NFIP flood claim severity?

**Relevance:** FEMA adjusters and portfolio managers need to know which factors to prioritize when assessing an incoming claim or evaluating portfolio risk. The answer from this model — that catastrophe year × coastal state interaction dominates, followed by building elevation and water depth — directly informs which data fields should be collected reliably at policy inception.

**Answer:** The top predictors are `cat_year_coastal` (importance 0.2676), `elevatedBuildingIndicator` (0.0632), `waterDepth` (0.0478), `buildingPropertyValue` (0.0339), and `longitude` (0.0275). Hazard intensity and geographic catastrophe exposure dominate; individual building characteristics are secondary but meaningful.

### BQ2: Why does XGBoost dramatically outperform Linear Regression on this dataset?

**Relevance:** Stakeholders investing in model deployment need to understand why a more complex model is necessary, and under what conditions simpler models might be adequate.

**Answer:** Flood damage is governed by conditional interactions that linear models cannot capture. Elevation only reduces losses when the building is also in a flood zone that floods regularly. Hurricane season only amplifies severity when combined with coastal location and high-risk zone. These conditionalities average out to near-zero in linear correlations but are captured precisely by XGBoost's tree splits, explaining the 30.7% → 66.9% R² improvement.

### BQ3: Can the model be operationalized to improve FEMA's claims process and portfolio management?

**Relevance:** A model's value is ultimately in its operational application, not its R² score. FEMA's claims workflow and mitigation programs represent specific, high-value deployment opportunities.

**Answer:** Yes — three distinct use cases are feasible with the current model (see Section 6). The most immediate value is **claims triage**: scoring incoming claims in real-time to dispatch senior adjusters to the highest expected losses first, rather than processing in arrival order.

---

## 5. Predictive Modeling

### 5.1 Model 1: Linear Regression (Baseline)

Linear Regression was trained as the minimum viable baseline. It assumes a linear relationship between each feature and the target, additively — no interactions, no non-linearity.

**Expected weakness:** Cannot capture the depth-damage curve, the conditional effect of elevation, or the multiplicative interaction between hazard intensity and building vulnerability. Any model claiming to be better must demonstrably outperform this floor.

| Metric | Value |
|---|---|
| R² | 0.307 |
| RMSE | $39,888 |
| MAE | $25,384 |
| Variation explained | 30.7% |

**Interpretation:** The model explains 30.7% of claim variation. On average, it is $25,384 away from the actual claim amount — more than the median claim value. This confirms that flood claim severity cannot be adequately modeled with a linear approach.

### 5.2 Model 2: Random Forest

A Random Forest of 100 independently trained decision trees was built. Each tree partitions the feature space differently; the final prediction is the average across all 100 trees.

**Hyperparameter choices:**
- `n_estimators=100` — 100 trees for stable predictions
- `max_depth=10` — limits complexity to prevent overfitting on 1.5M training records
- `min_samples_leaf=50` — prevents fitting individual noisy claims; each leaf must contain at least 50 claims

| Metric | Value | vs. Linear Regression |
|---|---|---|
| R² | 0.551 | +79% improvement |
| RMSE | $32,116 | −$7,772 better |
| MAE | $19,777 | −$5,607 better |

**Interpretation:** The 80% improvement in R² over Linear Regression confirms that non-linear relationships and feature interactions are critical. Random Forest discovers the depth-damage curve and zone×elevation interactions without any explicit specification.

### 5.3 Model 3: XGBoost (Final Model)

XGBoost uses gradient boosting — trees are built sequentially, with each new tree specifically targeting the residuals (errors) of all previous trees. This focused error correction on the hardest-to-predict claims (the right tail of large severity events) gives XGBoost its advantage over Random Forest on insurance data.

**Hyperparameter choices (v1 — manually tuned):**
- `n_estimators=450` — 450 sequential trees
- `max_depth=10` — allows complex interactions
- `min_child_weight=50` — minimum claim count per leaf
- `learning_rate=0.05` — slow learning rate for stable convergence
- `subsample=0.8`, `colsample_bytree=0.8` — stochastic sampling for regularization

**Optuna comparison (v2):** A 15-trial Optuna hyperparameter search found `n_estimators=355, max_depth=10, lr=0.0988`, achieving R² = 0.6660 — slightly underperforming the manually tuned v1 (0.6688). This confirms that domain-guided tuning can match or exceed automated search on structured tabular data. **XGBoost v1 is retained as the final model.**

| Metric | Linear Regression | Random Forest | **XGBoost v1 ✅** | XGBoost v2 (Optuna) |
|---|---|---|---|---|
| R² | 0.307 | 0.551 | **0.6688** | 0.6660 |
| RMSE | $39,888 | $32,116 | **$27,577** | $27,694 |
| MAE | $25,384 | $19,777 | **$16,474** | $16,564 |
| Variation Explained | 30.7% | 55.1% | **66.9%** | 66.6% |

### 5.4 Overfitting Check

| | Value |
|---|---|
| Train R² | 0.7254 |
| Test R² | 0.6688 |
| Gap | **0.057** |

The small train/test gap confirms the model generalizes well to unseen claims. It is learning real flood damage patterns — not memorizing training data.

### 5.5 Prediction Accuracy Distribution (XGBoost v1)

| Tolerance | % of Predictions |
|---|---|
| Within $5,000 of actual | 35% |
| Within $15,000 of actual | 67% |
| Within $25,000 of actual | 80% |
| Off by more than $50,000 | 7% |
| Median error | +$1,822 (near-zero bias) |

---

## 6. Insights & Recommendations

### Core Insight: Conditional Interactions Drive Flood Losses

The fundamental finding of this project is that flood claim severity is not determined by any single factor, but by the conditional interaction of hazard intensity, building vulnerability, and financial structure. Elevation only matters significantly when combined with high-risk zone designation. Hurricane season only amplifies severity when the property is coastal. These conditional effects explain why XGBoost (which discovers interactions through tree splits) outperforms Linear Regression by a factor of 2.2× in R² — and why insurers who rely on simple linear pricing rules systematically underprice high-interaction-risk properties.

### Recommendation 1: Real-Time Claims Triage

**Problem:** After a major flood event, thousands of claims arrive simultaneously and are typically processed in arrival order. A $180,000 loss at queue position 800 receives the same priority as a $2,000 loss at position 1.

**Solution:** Score every incoming claim instantly at submission, using data already collected (water depth, flood zone, building value, location, elevation status). Rank by predicted severity. Dispatch senior adjusters to the top-15% predicted segment first.

**Important caveat:** The model systematically under-predicts catastrophic claims in the $150,000+ range (a known limitation of gradient boosting on heavy-tailed targets). The priority queue should use the top 15% (not top 10%) to buffer for tail underestimation.

**Expected impact:** Faster settlement for the highest losses. Reduced litigation. Better resource allocation during the critical 24–72 hours after a major disaster event.

### Recommendation 2: Pre-Season Portfolio Risk Screening

**Problem:** NFIP knows which properties are insured but does not systematically identify which ones are most likely to generate large claims in the coming season.

**Solution:** Run the model on all in-force policies before each hurricane season. Set `waterDepth` to the median historical value for each flood zone (not zero — assume the property will eventually flood). The model output is a predicted severity score representing expected loss if the property floods. Flag the combination of `postFIRMConstructionIndicator = 0` + `elevatedBuildingIndicator = 0` + `is_coastal_state = 1` — the highest-risk segment confirmed by both EDA Q5 and the model's feature importance rankings.

**Important caveat:** This score assumes the property floods. It is not a flood probability estimate. Pair with historical flood frequency data (return period analysis) for a complete expected annual loss picture.

**Expected impact:** Proactive mitigation outreach — elevation certificate campaigns targeted at the properties where mitigation reduces the most expected loss per dollar invested.

### Recommendation 3: Property-Specific Elevation Valuation

**Problem:** NFIP elevation-based premium discounts are set using flat actuarial rate tables — the same reduction regardless of the specific property's flood zone, building value, or location context.

**Solution:** Run two predictions on each in-force policy: once with `elevatedBuildingIndicator = 0`, once with `elevatedBuildingIndicator = 1`. The difference is a property-specific dollar estimate of how much elevation reduces expected loss for that specific building in that specific context.

`predict(elevated=0) − predict(elevated=1) = estimated annual savings from elevation`

`elevatedBuildingIndicator` is the #2 feature by gain importance (0.0632), and its directional effect is clear and consistent across all flood zones in the SHAP analysis.

**Expected impact:** More accurate, defensible premium pricing. Data-driven dollar amounts to justify mitigation grant programs. Property owners receive a specific financial incentive rather than a generic rate table discount.

### Model Limitations

The current model explains 66.9% of claim severity variation — strong for insurance regression, but 33.1% remains unexplained. Key missing features that would likely improve performance beyond 67%:

- **Construction material:** Wood vs. masonry vs. concrete flood very differently at the same depth
- **Flood type:** Storm surge, riverine, and flash flood cause different damage patterns — the dataset records depth but not how water arrived
- **Water velocity:** Fast-moving shallow water causes more structural damage than still deep water
- **Prior claims history:** Repeat-flood buildings have cumulative structural weakening
- **Regional repair cost index:** The same physical damage costs twice as much to fix in New York vs. rural Louisiana

---

## 7. Ethics & Interpretability

### 7.1 Interpretability

This model is interpretable at two levels. At the global level, XGBoost's built-in feature importance (by gain) reveals which features most influence predictions across the entire dataset. `cat_year_coastal` — the interaction between catastrophe years and coastal states — accounts for 26.76% of total gain, confirming that historical extreme events (Katrina, Harvey, Sandy) dominate the model's learned signal. At the individual prediction level, SHAP (SHapley Additive exPlanations) values decompose each prediction into the contribution of each feature, producing an explanation any claims adjuster can read: "This claim was predicted at $95,000 because water depth was 14 inches (+$18K), the property is coastal (+$12K), and the building is not elevated (+$9K)."

This per-prediction explainability is operationally important: it allows human adjusters to verify the model's reasoning, override it when context warrants, and explain decisions to policyholders. An unexplainable black box with R² = 0.80 is less useful in an insurance operations context than an explainable model with R² = 0.67.

### 7.2 Fairness and Potential Bias

Several fairness considerations apply to this model:

**Geographic features as potential proxies.** The model uses `state`, `longitude`, `is_coastal_state`, and `state_target_enc` as predictive features. In an insurance context, geographic features are physically justified — coastal proximity genuinely drives catastrophe exposure. However, geography correlates with demographics: coastal and Gulf Coast states have significant minority and low-income populations. If the model is used for any purpose beyond claim severity prediction (e.g., to determine coverage eligibility or limit access), geographic features could function as demographic proxies in violation of fair lending and fair insurance principles. **This model should only be used for claim amount estimation, not eligibility decisions.**

**Pre-FIRM building bias.** Pre-FIRM (older) buildings predict lower average claims in the model, in part because they are often lower-value structures. However, Pre-FIRM buildings are disproportionately owned by lower-income and minority households who could not afford Post-FIRM compliant construction. A model that predicts lower expected loss for Pre-FIRM buildings could inadvertently justify lower claims service priority for more vulnerable households — the opposite of equitable disaster recovery.

**Tail underestimation.** The model systematically under-predicts catastrophic claims ($150,000+). If used for claims triage, this means the most severely impacted policyholders — who often face the longest recovery journeys — are also those most likely to be deprioritized if the triage buffer is set too conservatively. The recommendation to use the top 15% (not top 10%) as the priority queue is a direct response to this fairness concern.

**Transparency recommendation.** Any operational deployment should: (1) maintain human adjuster override authority on all high-severity predictions, (2) log model predictions alongside adjuster decisions to enable bias auditing, and (3) retrain annually on recent data to prevent concept drift as climate patterns shift coastal risk profiles. The NFIP serves as the insurer of last resort for many vulnerable households — model fairness is not optional.

### 7.3 Data Privacy

The FEMA NFIP Redacted Claims V2 dataset is a publicly released, de-identified dataset. Individual policyholder names, addresses, and policy numbers have been redacted by FEMA before public release. No personally identifiable information (PII) is present in the modeling dataset. Geographic precision is limited to county and ZIP code level, not parcel-level addresses.

---

## 8. Appendix

### A. Repository Contents

```
nfip-flood-severity/
├── README.md
├── .gitignore
├── index.html                       (project website)
├── NFIP_Complete_Analysis_v3.ipynb  (main Colab notebook)
├── report/
│   └── NFIP_Final_Report.md         (this document)
└── visualizations/
    ├── Building Vulnerability.png
    ├── Correlation Analysis.png
    ├── Feature Importance.png
    ├── Flood Zone Risk.png
    ├── Geographic Concentration of NFIP Flood Losses.png
    ├── Hazard Intensity.png
    ├── High Severity vs Low Severity Claim Profiles.png
    ├── Model Performance Comparison.png
    ├── SHAP Analysis.png
    ├── Seasonality of NFIP Flood Claims.png
    ├── Understanding Our Target.png
    └── XG Boost Model Diagnostic.png
```

### B. Libraries and Versions

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| matplotlib | 3.7+ | Visualization |
| seaborn | 0.12+ | Statistical visualization |
| scikit-learn | 1.3+ | Linear Regression, Random Forest, metrics |
| xgboost | 1.7+ | XGBoost model |
| shap | 0.42+ | SHAP values and plots |
| optuna | 3.2+ | Hyperparameter optimization |
| pyarrow | 12+ | Parquet file reading |

### C. Key Model Parameters

**XGBoost v1 (Final Model)**
```python
XGBRegressor(
    n_estimators      = 450,
    max_depth         = 10,
    learning_rate     = 0.05,
    min_child_weight  = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    tree_method       = 'hist',
    n_jobs            = -1
)
```

### D. Feature List (Final Model — 60+ features)

Key features included in the final model:
- `waterDepth`, `water_depth_log`
- `totalBuildingInsuranceCoverage`, `deductible_amount`, `buildingPropertyValue`, `buildingReplacementCost`
- `elevatedBuildingIndicator`, `postFIRMConstructionIndicator`
- `height_above_BFE`, `height_above_ground`, `freeboard_positive`, `freeboard_deficit`
- `flood_zone_risk_score`, `is_high_risk_zone`, `risk_zone_elevation_interaction`
- `building_age_at_loss`, `construction_year`
- `is_hurricane_season`, `loss_month`
- `is_coastal_state`, `state_target_enc`, `year_target_enc`, `cat_year_coastal`
- `numberOfFloorsInTheInsuredBuilding`, `numberOfUnits`, `occupancyType`
- `has_*` missingness indicator flags (8 features)
- `is_*` binary flags for building type (8 features)

### E. Evaluation Metric Definitions

| Metric | Formula | Interpretation |
|---|---|---|
| R² | 1 − SS_res/SS_tot | Proportion of variance explained; 1.0 is perfect |
| RMSE | √(mean(ŷ − y)²) | Penalizes large errors; in same units as target ($) |
| MAE | mean(\|ŷ − y\|) | Average absolute prediction error; directly interpretable ($) |

---

*ISOM 835 – Predictive Analytics and Machine Learning · Suffolk University MSBA · Spring 2026*
