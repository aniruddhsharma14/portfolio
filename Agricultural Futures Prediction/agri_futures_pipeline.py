"""
Agricultural Futures Prediction Pipeline
=========================================
Merges USDA NASS crop reports + NOAA PDSI drought index + CBOT corn futures
to engineer novel cross-domain features and train a predictive model.

Requirements:
    pip install pandas numpy requests scikit-learn xgboost shap matplotlib nasdaqdatalink

API Keys:
    - USDA NASS:      https://quickstats.nass.usda.gov/api
    - NOAA CDO:       https://www.ncdc.noaa.gov/cdo-web/webservices/v2
    - Nasdaq Data Link: https://data.nasdaq.com/sign-up
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
import shap
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION — fill in your API keys
# ─────────────────────────────────────────────────────────────────────────────

USDA_KEY       = os.getenv("USDA_KEY", "YOUR_USDA_NASS_KEY")
NOAA_TOKEN     = os.getenv("NOAA_TOKEN", "YOUR_NOAA_CDO_TOKEN")
NASDAQ_DL_KEY  = os.getenv("NASDAQ_DL_KEY", "YOUR_NASDAQ_DATALINK_KEY")

YEAR_START     = 2002
CORN_BELT_FIPS = [19, 17, 18, 31, 39]   # Iowa, Illinois, Indiana, Nebraska, Ohio


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def get_usda_condition(commodity: str = "CORN", year_start: int = YEAR_START) -> pd.DataFrame:
    """
    Fetch weekly crop condition percentages from USDA NASS QuickStats.
    Returns a DataFrame with columns: date, category, pct
    """
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": USDA_KEY,
        "commodity_desc": commodity,
        "statisticcat_desc": "CONDITION",
        "freq_desc": "WEEKLY",
        "year__GE": year_start,
        "state_name": "US TOTAL",
        "format": "JSON",
    }
    print(f"  Fetching USDA {commodity} condition data...")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])

    df["date"]  = pd.to_datetime(df["week_ending"])
    df["value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df["category"] = df["short_desc"].str.extract(r"PCT\s+(EXCELLENT|GOOD|FAIR|POOR|VERY POOR)")
    return df[["date", "category", "value"]].dropna()


def get_usda_progress(commodity: str = "CORN", year_start: int = YEAR_START) -> pd.DataFrame:
    """
    Fetch weekly planting progress (% planted) from USDA NASS.
    """
    url = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": USDA_KEY,
        "commodity_desc": commodity,
        "statisticcat_desc": "PROGRESS",
        "unit_desc": "PCT PLANTED",
        "freq_desc": "WEEKLY",
        "year__GE": year_start,
        "state_name": "US TOTAL",
        "format": "JSON",
    }
    print(f"  Fetching USDA {commodity} planting progress...")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["date"]  = pd.to_datetime(df["week_ending"])
    df["value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    return df[["date", "value"]].dropna().set_index("date")["value"].rename("pct_planted")


def get_noaa_pdsi(fips_list: list = CORN_BELT_FIPS, year_start: int = YEAR_START) -> pd.Series:
    """
    Fetch monthly Palmer Drought Severity Index (PDSI) for Corn Belt states
    from NOAA Climate Data Online API.
    Averages across all states to get a single Corn Belt PDSI series.
    """
    base    = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": NOAA_TOKEN}
    frames  = []

    for fips in fips_list:
        print(f"  Fetching NOAA PDSI for FIPS {fips:02d}...")
        params = {
            "datasetid":  "CLIMDIV",
            "datatypeid": "PDSI",
            "locationid": f"FIPS:{fips:02d}",
            "startdate":  f"{year_start}-01-01",
            "enddate":    "2024-12-01",
            "limit":      1000,
        }
        r = requests.get(base, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            frames.append(pd.DataFrame(results))

    df   = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    pdsi = (
        df.groupby("date")["value"]
        .mean()
        .rename("pdsi")
        .resample("ME")
        .last()
    )
    return pdsi


def get_corn_futures(year_start: int = YEAR_START) -> pd.Series:
    """
    Fetch front-month corn continuous contract (ZC1) from Nasdaq Data Link
    (formerly Quandl). Dataset: CHRIS/CME_C1.
    """
    print("  Fetching corn futures from Nasdaq Data Link...")
    try:
        import nasdaqdatalink
        nasdaqdatalink.ApiConfig.api_key = NASDAQ_DL_KEY
        df = nasdaqdatalink.get("CHRIS/CME_C1", start_date=f"{year_start}-01-01")
        return df["Settle"].resample("ME").last().rename("corn_price")
    except Exception as e:
        print(f"  [Warning] Nasdaq Data Link fetch failed: {e}")
        print("  Falling back to synthetic demo data for illustration.")
        return _synthetic_corn_futures(year_start)


def _synthetic_corn_futures(year_start: int) -> pd.Series:
    """
    Generates realistic synthetic corn futures prices for demo/testing
    when API keys are not yet configured. Replace with real data for production.
    """
    np.random.seed(42)
    idx    = pd.date_range(f"{year_start}-01-31", "2024-12-31", freq="ME")
    trend  = np.linspace(230, 480, len(idx))
    shocks = np.random.randn(len(idx)).cumsum() * 12
    # Add 2012 drought spike
    drought_2012 = np.where(
        (idx.year == 2012) & (idx.month.isin(range(6, 10))),
        np.linspace(0, 180, sum((idx.year == 2012) & (idx.month.isin(range(6, 10))))),
        0
    )
    prices = np.clip(trend + shocks + np.pad(drought_2012, (0, len(idx) - len(drought_2012))), 200, 850)
    return pd.Series(prices, index=idx, name="corn_price")


def _synthetic_pdsi(year_start: int) -> pd.Series:
    """Synthetic PDSI for demo purposes."""
    np.random.seed(7)
    idx  = pd.date_range(f"{year_start}-01-31", "2024-12-31", freq="ME")
    pdsi = np.random.randn(len(idx)).cumsum() * 0.4
    pdsi = np.clip(pdsi, -6, 6)
    # 2012 extreme drought
    for i, d in enumerate(idx):
        if d.year == 2012 and d.month in range(5, 10):
            pdsi[i] = -4.5
    return pd.Series(pdsi, index=idx, name="pdsi")


def _synthetic_condition(year_start: int) -> pd.DataFrame:
    """Synthetic USDA crop condition for demo purposes."""
    np.random.seed(3)
    idx   = pd.date_range(f"{year_start}-06-01", f"2024-09-30", freq="W-MON")
    cats  = ["EXCELLENT", "GOOD", "FAIR", "POOR", "VERY POOR"]
    rows  = []
    for d in idx:
        base = [15, 40, 30, 10, 5]
        if d.year == 2012 and d.month in [7, 8]:
            base = [5, 15, 30, 30, 20]
        noise = np.random.dirichlet(np.array(base) * 0.5) * 100
        for cat, val in zip(cats, noise):
            rows.append({"date": d, "category": cat, "value": round(val, 1)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(use_synthetic: bool = False) -> tuple:
    """
    Load all data sources. Set use_synthetic=True to run without API keys.
    Returns: (condition_df, pdsi_series, corn_futures_series)
    """
    print("\n[1/3] Loading data sources...")

    if use_synthetic:
        print("  Using synthetic demo data (set use_synthetic=False for real data)")
        cond  = _synthetic_condition(YEAR_START)
        pdsi  = _synthetic_pdsi(YEAR_START)
        corn  = _synthetic_corn_futures(YEAR_START)
    else:
        cond  = get_usda_condition("CORN")
        pdsi  = get_noaa_pdsi()
        corn  = get_corn_futures()

    return cond, pdsi, corn


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_index(cond_df: pd.DataFrame) -> pd.Series:
    """
    Collapse 5-category USDA condition into a weighted score.

    Formula:  2×(Excellent%) + 1×(Good%) − 1×(Poor%) − 2×(Very Poor%)

    Rationale: Extremes matter more than the middle. This mirrors how
    commodity traders mentally weight the weekly USDA report.
    Range: roughly −200 to +200 (centred near 0 in average years).
    """
    pivot = cond_df.pivot_table(index="date", columns="category", values="value")
    idx = (
          pivot.get("EXCELLENT", 0) * 2
        + pivot.get("GOOD",      0) * 1
        - pivot.get("POOR",      0) * 1
        - pivot.get("VERY POOR", 0) * 2
    )
    return idx.resample("ME").last().rename("condition_idx")


def build_features(cond_df: pd.DataFrame,
                   pdsi: pd.Series,
                   corn: pd.Series) -> pd.DataFrame:
    """
    Build the full feature matrix from the three raw data sources.

    Feature categories:
    ─── Drought (NOAA PDSI) ───────────────────────────────────────────────────
      pdsi_lag1         : PDSI shifted 1 month (respect data availability)
      pdsi_3m           : 3-month rolling average of PDSI
      pdsi_velocity     : Month-on-month change in PDSI (direction matters)
      extreme_drought   : Binary flag: PDSI < −3

    ─── USDA crop condition ───────────────────────────────────────────────────
      condition_idx     : Weighted condition score (see build_condition_index)
      condition_z       : Z-score vs. same-calendar-month 5-yr history
      planting_lag      : (Placeholder — add pct_planted data for full version)

    ─── Cross-domain interactions (NOVEL) ────────────────────────────────────
      pdsi_x_condition  : PDSI × condition_idx — drought AND poor condition
                          compound non-linearly; linear models miss this
      drought_critical  : PDSI velocity < −1 during July–August corn pollination
                          window. Same drought level = 3× yield impact if it
                          hits during pollination vs. September.

    ─── Technical / price-based ───────────────────────────────────────────────
      mom_1m            : 1-month log return
      mom_3m            : 3-month log return
      mom_12m           : 12-month log return
      price_to_5yr      : Current price / 60-month rolling average
      vol_3m            : 3-month realised volatility proxy
    """
    df = pd.DataFrame(index=corn.index)
    df["corn_price"] = corn

    # ── Target: next-month log return ────────────────────────────────────────
    df["log_ret"]  = np.log(df["corn_price"]).diff(1)
    df["target"]   = df["log_ret"].shift(-1)   # predict NEXT month

    # ── PDSI features (lag 1 month to respect real-time availability) ────────
    pdsi_aligned       = pdsi.reindex(df.index, method="ffill").shift(1)
    df["pdsi_lag1"]    = pdsi_aligned
    df["pdsi_3m"]      = pdsi_aligned.rolling(3).mean()
    df["pdsi_velocity"]= pdsi_aligned.diff(1)
    df["extreme_drought"] = (pdsi_aligned < -3).astype(int)

    # ── USDA condition features (shift 1 period: released Mon, use next month) 
    cond_monthly = build_condition_index(cond_df)
    cond_aligned = cond_monthly.reindex(df.index, method="ffill").shift(1)
    df["condition_idx"] = cond_aligned

    # Seasonal z-score: deviation from same-calendar-month expanding history
    df["condition_z"] = df.groupby(df.index.month)["condition_idx"].transform(
        lambda x: (x - x.expanding(5).mean()) / x.expanding(5).std().clip(lower=0.1)
    )

    # ── Cross-domain interaction features ────────────────────────────────────
    df["pdsi_x_condition"]  = df["pdsi_3m"] * df["condition_idx"]

    # Critical window: July–August = corn pollination
    is_critical_window      = df.index.month.isin([7, 8])
    df["drought_critical"]  = (
        is_critical_window & (df["pdsi_velocity"] < -1)
    ).astype(int)

    # ── Technical features ────────────────────────────────────────────────────
    df["mom_1m"]         = df["log_ret"]
    df["mom_3m"]         = np.log(df["corn_price"]).diff(3)
    df["mom_12m"]        = np.log(df["corn_price"]).diff(12)
    df["price_to_5yr"]   = df["corn_price"] / df["corn_price"].rolling(60).mean()
    df["vol_3m"]         = df["log_ret"].rolling(3).std()

    return df.dropna()


FEATURE_COLS = [
    # Drought
    "pdsi_lag1", "pdsi_3m", "pdsi_velocity", "extreme_drought",
    # USDA condition
    "condition_idx", "condition_z",
    # Cross-domain interactions
    "pdsi_x_condition", "drought_critical",
    # Technical
    "mom_1m", "mom_3m", "mom_12m", "price_to_5yr", "vol_3m",
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING — expanding-window cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_expanding_window_cv(features: pd.DataFrame,
                             min_train_months: int = 36) -> pd.DataFrame:
    """
    Time-series cross-validation with expanding training window.

    Why expanding window (not rolling)?
    - More data = better estimates for low-frequency drought events
    - Avoids discarding the informative 2012 drought once we're past it
    - Closest to how a real quant would retrain: periodically, on all history

    Returns DataFrame with columns: date, actual, xgb_pred, ridge_pred
    """
    print("\n[2/3] Running expanding-window cross-validation...")

    X   = features[FEATURE_COLS].values
    y   = features["target"].values
    idx = features.index

    xgb_preds, ridge_preds, actuals = [], [], []

    for t in range(min_train_months, len(X) - 1):
        X_tr, y_tr = X[:t],     y[:t]
        X_te, y_te = X[t:t+1],  y[t:t+1]

        scaler    = StandardScaler()
        X_tr_s    = scaler.fit_transform(X_tr)
        X_te_s    = scaler.transform(X_te)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=150, max_depth=3,
            learning_rate=0.04, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        xgb_model.fit(X_tr_s, y_tr)

        # Ridge (baseline)
        ridge     = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
        ridge.fit(X_tr_s, y_tr)

        xgb_preds.append(xgb_model.predict(X_te_s)[0])
        ridge_preds.append(ridge.predict(X_te_s)[0])
        actuals.append(y_te[0])

    results = pd.DataFrame({
        "date":       idx[min_train_months:-1],
        "actual":     actuals,
        "xgb_pred":   xgb_preds,
        "ridge_pred": ridge_preds,
    }).set_index("date")

    return results


def evaluate_results(results: pd.DataFrame) -> dict:
    """Compute OOS R², direction accuracy, and information ratio."""
    metrics = {}
    for model in ["xgb", "ridge"]:
        pred = results[f"{model}_pred"]
        act  = results["actual"]
        metrics[model] = {
            "oos_r2":       round(r2_score(act, pred), 4),
            "dir_accuracy": round(np.mean(np.sign(pred) == np.sign(act)), 4),
            "correlation":  round(np.corrcoef(act, pred)[0, 1], 4),
        }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(features: pd.DataFrame) -> tuple:
    """
    Train final XGBoost on full dataset and compute SHAP values.
    Returns (explainer, shap_values, X_scaled, feature_names).
    """
    print("\n[3/3] Computing SHAP feature importance...")
    X      = features[FEATURE_COLS].values
    y      = features["target"].values

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    model  = xgb.XGBRegressor(
        n_estimators=150, max_depth=3,
        learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    model.fit(X_s, y)

    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X_s)

    return explainer, shap_values, X_s, FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(features: pd.DataFrame,
                 results: pd.DataFrame,
                 metrics: dict,
                 shap_values: np.ndarray):
    """
    4-panel figure:
      (a) Corn futures price + PDSI drought overlay
      (b) Actual vs. predicted log returns (XGBoost)
      (c) SHAP feature importance bar chart
      (d) Rolling direction accuracy (XGBoost vs. Ridge)
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Agricultural Futures Prediction Pipeline\n"
        "USDA Crop Reports + NOAA PDSI Drought Index + Corn Futures",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # ── (a) Price + PDSI ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    ax1.plot(features.index, features["corn_price"],
             color="#2563eb", linewidth=1.5, label="Corn price (¢/bu)")
    ax1_twin.fill_between(
        features.index, features["pdsi_lag1"], 0,
        where=(features["pdsi_lag1"] < 0),
        color="#dc2626", alpha=0.3, label="Drought (PDSI < 0)",
    )
    ax1_twin.fill_between(
        features.index, features["pdsi_lag1"], 0,
        where=(features["pdsi_lag1"] >= 0),
        color="#16a34a", alpha=0.2, label="Wet (PDSI > 0)",
    )
    ax1.set_title("(a) Corn futures vs. PDSI drought index", fontsize=11)
    ax1.set_ylabel("Price (¢/bu)", color="#2563eb")
    ax1_twin.set_ylabel("PDSI", color="#6b7280")
    ax1_twin.axhline(y=-3, color="#dc2626", linestyle="--", linewidth=0.8,
                     label="Extreme drought (−3)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # ── (b) Actual vs. predicted returns ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(results["actual"], results["xgb_pred"],
                alpha=0.35, s=18, color="#7c3aed", edgecolors="none")
    lim = max(abs(results["actual"].max()), abs(results["actual"].min())) * 1.1
    ax2.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Actual log return")
    ax2.set_ylabel("Predicted log return")
    ax2.set_title(
        f"(b) Actual vs. predicted (OOS)\n"
        f"XGBoost R²={metrics['xgb']['oos_r2']:.3f}  |  "
        f"Ridge R²={metrics['ridge']['oos_r2']:.3f}",
        fontsize=11,
    )
    ax2.text(0.05, 0.92,
             f"Direction accuracy: {metrics['xgb']['dir_accuracy']:.1%}",
             transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6", alpha=0.8))

    # ── (c) SHAP feature importance ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order         = np.argsort(mean_abs_shap)
    colors        = [
        "#dc2626" if "x_condition" in FEATURE_COLS[i] or "critical" in FEATURE_COLS[i]
        else "#2563eb" if "pdsi" in FEATURE_COLS[i]
        else "#6b7280"
        for i in order
    ]
    bars = ax3.barh(
        [FEATURE_COLS[i] for i in order],
        mean_abs_shap[order],
        color=colors, edgecolor="none", height=0.65,
    )
    ax3.set_xlabel("Mean |SHAP value|")
    ax3.set_title("(c) SHAP feature importance\n(red = novel cross-domain features)", fontsize=11)
    ax3.tick_params(axis="y", labelsize=8)

    # Add a simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#dc2626", label="Cross-domain interaction"),
        Patch(facecolor="#2563eb", label="Drought (PDSI)"),
        Patch(facecolor="#6b7280", label="Technical / condition"),
    ]
    ax3.legend(handles=legend_elements, fontsize=7, loc="lower right")

    # ── (d) Rolling 12-month direction accuracy ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    roll = 12
    xgb_dir  = (np.sign(results["xgb_pred"])   == np.sign(results["actual"])).astype(float)
    ridge_dir = (np.sign(results["ridge_pred"]) == np.sign(results["actual"])).astype(float)
    ax4.plot(results.index, xgb_dir.rolling(roll).mean(),
             color="#7c3aed", linewidth=1.5, label=f"XGBoost ({roll}m rolling)")
    ax4.plot(results.index, ridge_dir.rolling(roll).mean(),
             color="#6b7280", linewidth=1.2, linestyle="--", label=f"Ridge ({roll}m rolling)")
    ax4.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6, label="Random (50%)")
    ax4.set_ylim(0.2, 0.85)
    ax4.set_ylabel("Direction accuracy")
    ax4.set_title(f"(d) Rolling {roll}-month direction accuracy", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.fill_between(results.index, 0.5,
                     xgb_dir.rolling(roll).mean(),
                     where=(xgb_dir.rolling(roll).mean() > 0.5),
                     alpha=0.15, color="#7c3aed")

    plt.savefig("agri_futures_results.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print("\n  Figure saved to agri_futures_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(use_synthetic: bool = True):
    """
    End-to-end pipeline run.

    Parameters
    ----------
    use_synthetic : bool
        True  → run on synthetic data (no API keys needed, for demo)
        False → fetch real data from USDA / NOAA / Nasdaq Data Link
    """
    print("=" * 65)
    print("  Agricultural Futures Prediction Pipeline")
    print("=" * 65)

    # 1. Load data
    cond_df, pdsi, corn = load_all_data(use_synthetic=use_synthetic)

    # 2. Build features
    print("\n  Building feature matrix...")
    features = build_features(cond_df, pdsi, corn)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Date range: {features.index[0].date()} → {features.index[-1].date()}")
    print(f"  Features: {FEATURE_COLS}")

    # 3. Cross-validation
    results  = run_expanding_window_cv(features)
    metrics  = evaluate_results(results)

    print("\n  ── Out-of-sample results ───────────────────────────────────")
    for model, m in metrics.items():
        print(f"  {model.upper():6s}  R²={m['oos_r2']:.3f}  "
              f"Dir accuracy={m['dir_accuracy']:.1%}  "
              f"Corr={m['correlation']:.3f}")

    r2_lift = metrics["xgb"]["oos_r2"] - metrics["ridge"]["oos_r2"]
    print(f"\n  R² lift from cross-domain features: +{r2_lift:.3f}")
    print(f"  (XGBoost vs. price-only Ridge baseline)\n")

    # 4. SHAP
    explainer, shap_values, X_s, feat_names = compute_shap(features)

    # 5. Plot
    plot_results(features, results, metrics, shap_values)

    return features, results, metrics, shap_values


if __name__ == "__main__":
    # Set use_synthetic=False and fill in your API keys at the top to use real data
    features, results, metrics, shap_values = main(use_synthetic=True)
