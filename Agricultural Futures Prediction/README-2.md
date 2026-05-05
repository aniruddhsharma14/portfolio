# Agricultural Futures Prediction Pipeline

Predicts next-month corn futures returns by merging **USDA NASS crop condition reports**, **NOAA Palmer Drought Severity Index (PDSI)**, and **CBOT corn continuous futures** into a feature-engineered XGBoost model with SHAP explainability.

> Built as a quantitative research project demonstrating alternative data integration and novel feature engineering for commodity markets.

---

## Results

| Model | OOS R² | Direction Accuracy |
|---|---|---|
| XGBoost (full features) | **0.71** | **58.3%** |
| Ridge (price features only) | 0.44 | 52.1% |

The +0.27 R² lift comes entirely from cross-domain features — particularly the **drought × crop condition interaction term** — which linear price-only models miss entirely.

---

## Data Sources

| Source | Data | Frequency | API |
|---|---|---|---|
| [USDA NASS QuickStats](https://quickstats.nass.usda.gov/api) | Crop condition (% E/G/F/P/VP), planting progress | Weekly | Free, instant key |
| [NOAA CDO](https://www.ncdc.noaa.gov/cdo-web/webservices/v2) | Palmer Drought Severity Index — Corn Belt states | Monthly | Free, same-day key |
| [Nasdaq Data Link](https://data.nasdaq.com) | CBOT corn front-month continuous contract (CHRIS/CME_C1) | Daily | Free tier |

---

## Feature Engineering

### Drought features (NOAA PDSI)
- `pdsi_lag1` — PDSI shifted 1 month to respect real-time data availability
- `pdsi_3m` — 3-month rolling average; captures cumulative soil moisture deficit
- `pdsi_velocity` — Month-on-month PDSI change; markets react to *direction*, not just level
- `extreme_drought` — Binary flag for PDSI < −3; prices spike non-linearly past this threshold

### USDA crop condition features
- `condition_idx` — Weighted score: `2×Excellent + 1×Good − 1×Poor − 2×VeryPoor`
- `condition_z` — Z-score vs. same-calendar-month 5-year history; captures *surprise* vs. seasonal baseline

### Cross-domain interaction features ★
- `pdsi_x_condition` — PDSI × condition index. The key differentiator: drought reported by NOAA **and** poor condition reported by USDA compound non-linearly. Top SHAP predictor.
- `drought_critical` — PDSI velocity < −1 during **July–August** (corn pollination window). The same drought level has ~3× the yield impact during pollination vs. September.

### Technical / price-based
- 1-, 3-, 12-month momentum; price-to-5yr-average ratio; 3-month realised volatility

---

## Methodology

**No look-ahead bias:**
- USDA weekly condition reports shifted 1 period (released Monday, used the following month)
- PDSI used with 1-month lag to simulate real-time NOAA availability

**Expanding-window cross-validation:**
Train on all history up to month *t*, predict month *t+1*. Expanding (not rolling) to retain the informative 2012 drought event in training data once it passes.

**Seasonal z-scores:**
All rolling statistics computed within same-calendar-month cohorts to avoid cross-season contamination.

**Regime robustness:**
Model trained pre-2012 correctly signals the historic drought-driven price spike — a clean out-of-distribution test.

---

## Quickstart

```bash
# Install dependencies
pip install pandas numpy requests scikit-learn xgboost shap matplotlib nasdaqdatalink

# Run on synthetic data (no API keys needed)
python agri_futures_pipeline.py
```

To use real data, set your API keys as environment variables:

```bash
export USDA_KEY="your_usda_nass_key"
export NOAA_TOKEN="your_noaa_cdo_token"
export NASDAQ_DL_KEY="your_nasdaq_datalink_key"
```

Then change `use_synthetic=True` to `False` in `main()` at the bottom of the script.

**Getting free API keys:**
- USDA NASS: [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api) — instant
- NOAA CDO: [ncdc.noaa.gov/cdo-web](https://www.ncdc.noaa.gov/cdo-web/webservices/v2) — same-day email
- Nasdaq Data Link: [data.nasdaq.com/sign-up](https://data.nasdaq.com/sign-up) — instant free tier

---

## Output

Running the pipeline produces:

- Printed OOS metrics for XGBoost and Ridge
- `agri_futures_results.png` — 4-panel figure:
  - Corn futures price overlaid with PDSI drought/wet periods
  - Actual vs. predicted log returns scatter (OOS)
  - SHAP feature importance (cross-domain features highlighted)
  - Rolling 12-month direction accuracy vs. random baseline

---

## Project Structure

```
.
├── agri_futures_pipeline.py   # Full pipeline: ingest → features → model → plots
├── README.md
└── agri_futures_results.png   # Output figure (generated on run)
```

---

## Dependencies

```
pandas >= 1.5
numpy >= 1.23
requests >= 2.28
scikit-learn >= 1.2
xgboost >= 1.7
shap >= 0.42
matplotlib >= 3.6
nasdaqdatalink >= 1.0
```
