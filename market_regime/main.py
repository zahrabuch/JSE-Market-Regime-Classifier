"""
Market regime detection for the JSE All Jamaican Composite index.

Pipeline:
    1. Load + clean raw CSV
    2. Engineer technical/volatility/volume features
    3. Scale features
    4. Fit a Gaussian Mixture Model, selecting k via BIC
    5. Persist scaler + model for reuse on new data

Run as a script to fit and save artifacts:
    python regime_pipeline.py

Use `predict_regime()` afterwards to score new data with the saved artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("JSE All Jamaican Composite Historical Data.csv")
SCALER_PATH = Path("scaler.pkl")
MODEL_PATH = Path("market_regime_gmm.pkl")

# Number of regimes to fit. BIC is still computed and logged as a diagnostic
# (see select_k_by_bic) so you can sanity-check this choice, but it does not
# override it -- the final model always uses N_REGIMES.
N_REGIMES = 4

FEATURE_COLUMNS = [
    "Daily Return",
    "5 Day Return",
    "20 Day Return",
    "Dist_SMA20",
    "Dist_SMA200",
    "RSI",
    "MACD",
    "Volatility 20",
    "ATR",
    "Relative Volume",
]


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_csv(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def convert_volume(volume) -> float:
    """Convert strings like '1.2M', '350K', '2B' into raw floats."""
    if pd.isna(volume):
        return np.nan

    volume = str(volume).replace(",", "").strip().upper()
    multipliers = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}

    if volume and volume[-1] in multipliers:
        return float(volume[:-1]) * multipliers[volume[-1]]

    return float(volume)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse numeric columns, fix volume suffixes, sort by date, set index."""
    df = df.copy()

    df["Change %"] = df["Change %"].str.replace("%", "", regex=False)

    numeric_cols = ["Price", "Open", "High", "Low", "Change %"]
    for col in numeric_cols:
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Vol."] = df["Vol."].apply(convert_volume)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")
    df = df.rename(columns={"Price": "Close"})

    return df


def winsorize_close(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Clip (not drop) extreme Close values.

    Dropping rows outright breaks the time index that rolling-window features
    depend on (SMA/ATR/volatility would silently span gaps as if no time had
    passed). Clipping keeps every date in place.
    """
    df = df.copy()
    lower, upper = df["Close"].quantile([lower_q, upper_q])
    df["Close"] = df["Close"].clip(lower, upper)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Daily Return"] = df["Close"].pct_change()
    df["5 Day Return"] = df["Close"].pct_change(5)
    df["10 Day Return"] = df["Close"].pct_change(10)
    df["20 Day Return"] = df["Close"].pct_change(20)

    for window in (20, 50, 100, 200):
        sma = df["Close"].rolling(window).mean()
        df[f"SMA{window}"] = sma
        df[f"Dist_SMA{window}"] = (df["Close"] - sma) / sma

    df["RSI"] = RSIIndicator(df["Close"]).rsi()

    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Sig"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    df["Volatility 20"] = df["Daily Return"].rolling(20).std()
    df["Volatility 50"] = df["Daily Return"].rolling(50).std()

    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"])
    df["ATR"] = atr.average_true_range()

    df["HL Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Parkinson"] = np.log(df["High"] / df["Low"]) ** 2
    df["Parkinson 20"] = df["Parkinson"].rolling(20).mean()

    df["OC Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    df["Body"] = (df["Open"] - df["Close"]).abs()
    df["Upper Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df["Volume SMA 20"] = df["Vol."].rolling(20).mean()
    df["Relative Volume"] = df["Vol."] / df["Volume SMA 20"]
    df["Volume ROC"] = df["Vol."].pct_change(10)

    obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Vol."])
    df["OBV"] = obv.on_balance_volume()

    return df.dropna().reset_index(drop=False)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on the given features and return (scaled_df, scaler)."""
    X = df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Model selection + fitting
# ---------------------------------------------------------------------------

def bic_diagnostic(X_scaled: pd.DataFrame, k_range: range = range(3, 7)) -> dict[int, float]:
    """
    Fit a GMM for each k in k_range and log its BIC, purely as a diagnostic.

    This does NOT choose k for you -- it's here so you can compare your
    chosen N_REGIMES against neighboring values and confirm it's reasonable
    (lower BIC = better fit relative to model complexity), while still
    letting you pick k based on interpretability, not just the statistic.
    """
    bic_scores: dict[int, float] = {}

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(X_scaled)
        bic_scores[k] = gmm.bic(X_scaled)
        marker = "  <- chosen N_REGIMES" if k == N_REGIMES else ""
        logger.info("k=%d  BIC=%.1f%s", k, bic_scores[k], marker)

    return bic_scores


def fit_gmm(X_scaled: pd.DataFrame, n_components: int) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    return gmm


def summarize_regimes(df: pd.DataFrame, regime_labels: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["Regime"] = regime_labels
    summary_cols = ["Daily Return", "Volatility 20", "RSI", "ATR", "Dist_SMA200", "Relative Volume"]
    return df.groupby("Regime")[summary_cols].mean()


# ---------------------------------------------------------------------------
# Inference on new data
# ---------------------------------------------------------------------------

def predict_regime(
    new_raw_df: pd.DataFrame,
    scaler_path: Path = SCALER_PATH,
    model_path: Path = MODEL_PATH,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Score new raw OHLCV data with the previously fit scaler + GMM.

    `new_raw_df` should have the same raw columns as the training CSV
    (Date, Price, Open, High, Low, Vol., Change %), and ideally include
    enough trailing history for the rolling-window features (200+ rows)
    to be valid for the most recent date.
    """
    scaler: StandardScaler = joblib.load(scaler_path)
    gmm: GaussianMixture = joblib.load(model_path)

    cleaned = clean_data(new_raw_df)
    featured = create_features(cleaned)

    X = featured[feature_columns]
    X_scaled = scaler.transform(X)

    featured["Regime"] = gmm.predict(X_scaled)
    proba = gmm.predict_proba(X_scaled)
    for i in range(proba.shape[1]):
        featured[f"Regime_{i}_Prob"] = proba[:, i]

    return featured


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Loading and cleaning data")
    raw = load_csv()
    cleaned = clean_data(raw)
    cleaned = winsorize_close(cleaned)

    logger.info("Engineering features")
    featured = create_features(cleaned)

    logger.info("Scaling features")
    X_scaled, scaler = fit_scaler(featured)

    logger.info("Computing BIC diagnostic across k=3..6")
    bic_diagnostic(X_scaled)

    logger.info("Fitting final GMM with N_REGIMES=%d", N_REGIMES)
    gmm = fit_gmm(X_scaled, n_components=N_REGIMES)
    labels = gmm.predict(X_scaled)

    summary = summarize_regimes(featured, labels)
    logger.info("Regime summary:\n%s", summary)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(gmm, MODEL_PATH)
    logger.info("Saved scaler -> %s, model -> %s", SCALER_PATH, MODEL_PATH)


if __name__ == "__main__":
    main()
