import joblib
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import plotly.express as px
import plotly.graph_objects as go



def load_csv():
    return pd.read_csv("JSE All Jamaican Composite Historical Data.csv")


def convert_volume(volume):
    if pd.isna(volume):
        return np.nan

    volume = str(volume).replace(",", "").strip().upper()

    if volume.endswith("M"):
        return float(volume[:-1]) * 1_000_000
    elif volume.endswith("K"):
        return float(volume[:-1]) * 1_000
    elif volume.endswith("B"):
        return float(volume[:-1]) * 1_000_000_000

    return float(volume)


def clean_data(df):

    df["Change %"] = df["Change %"].str.replace("%", "", regex=False)

    numeric_columns = [
        "Price",
        "Open",
        "High",
        "Low",
        "Change %"
    ]

    for col in numeric_columns:
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Vol."] = df["Vol."].apply(convert_volume)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
          .reset_index(drop=True)
          .set_index("Date")
    )

    df = df.rename(columns={"Price": "Close"})

    return df


def create_features(df):

    df["Daily Return"] = df["Close"].pct_change()
    df["5 Day Return"] = df["Close"].pct_change(5)
    df["10 Day Return"] = df["Close"].pct_change(10)
    df["20 Day Return"] = df["Close"].pct_change(20)

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["Dist_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"]
    df["Dist_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"]
    df["Dist_SMA100"] = (df["Close"] - df["SMA100"]) / df["SMA100"]
    df["Dist_SMA200"] = (df["Close"] - df["SMA200"]) / df["SMA200"]

    df["RSI"] = RSIIndicator(df["Close"]).rsi()

    macd = MACD(df["Close"])

    df["MACD"] = macd.macd()
    df["MACD_Sig"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    df["Volatility 20"] = df["Daily Return"].rolling(20).std()
    df["Volatility 50"] = df["Daily Return"].rolling(50).std()

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["ATR"] = atr.average_true_range()

    df["HL Range"] = (df["High"] - df["Low"]) / df["Close"]

    df["Parkinson"] = np.log(df["High"] / df["Low"]) ** 2
    df["Parkinson 20"] = df["Parkinson"].rolling(20).mean()

    df["OC Return"] = (df["Close"] - df["Open"]) / df["Open"]

    df["Gap"] = (
        (df["Open"] - df["Close"].shift(1))
        / df["Close"].shift(1)
    )

    df["Body"] = (df["Open"] - df["Close"]).abs()

    df["Upper Wick"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    )

    df["Lower Wick"] = (
        df[["Open", "Close"]].min(axis=1) - df["Low"]
    )

    df["Volume SMA 20"] = df["Vol."].rolling(20).mean()

    df["Relative Volume"] = (
        df["Vol."] / df["Volume SMA 20"]
    )

    df["Volume ROC"] = df["Vol."].pct_change(10)

    obv = OnBalanceVolumeIndicator(
        close=df["Close"],
        volume=df["Vol."]
    )

    df["OBV"] = obv.on_balance_volume()

    df = df.dropna().reset_index(drop=True)

    return df


def scale_features(df):

    features = [
        "Daily Return",
        "5 Day Return",
        "20 Day Return",
        "Dist_SMA20",
        "Dist_SMA200",
        "RSI",
        "MACD",
        "Volatility 20",
        "ATR",
        "Relative Volume"
    ]

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=features,
        index=df.index
    )

    return X_scaled, scaler


def create_gmm(scaled_df):

    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=42
    )

    gmm.fit(scaled_df)

    return gmm


df = load_csv()

clean_df = clean_data(df)

clean_df = clean_df[
    clean_df["Close"].between(
        clean_df["Close"].quantile(0.01),
        clean_df["Close"].quantile(0.99)
    )
]

feat_df = create_features(clean_df)

X_scaled, scaler = scale_features(feat_df)

print(X_scaled.head())

bics = []

for k in range(3, 6):

    gmm = GaussianMixture(
        n_components=k,
        random_state=42
    )

    labels = gmm.fit_predict(X_scaled)

    feat_df["Regime"] = labels

    print(feat_df["Regime"].value_counts())

    print(
        feat_df.groupby("Regime")[[
            "Daily Return",
            "Volatility 20",
            "RSI",
            "Dist_SMA200"
        ]].mean()
    )

    bics.append(gmm.bic(X_scaled))

print(bics)

gmm = create_gmm(X_scaled)

feat_df["Regime"] = gmm.predict(X_scaled)

print(
    feat_df.groupby("Regime")[[
        "Daily Return",
        "Volatility 20",
        "RSI",
        "ATR",
        "Dist_SMA200",
        "Relative Volume"
    ]].mean()
)

joblib.dump(gmm, "market_regime_gmm.pkl")
joblib.dump(scaler, "scaler.pkl")

df = feat_df.copy()

stats = df.groupby("Regime").agg({
    "Daily Return": "mean",
    "Volatility 20": "mean",
    "RSI": "mean",
    "Dist_SMA200": "mean"
})

bull = stats["Daily Return"].idxmax()
bear = stats["Daily Return"].idxmin()

remaining = [r for r in stats.index if r not in [bull, bear]]

remaining = sorted(
    remaining,
    key=lambda x: stats.loc[x, "Volatility 20"]
)

sideways = remaining[0]
momentum = remaining[1]

regime_names = {
    bull: "Bull",
    bear: "Bear",
    sideways: "Sideways",
    momentum: "Momentum"
}

df["Regime Name"] = df["Regime"].map(regime_names)

df["Classifier Regime"] = df["Regime Name"].replace({
    "Bull": "Bullish",
    "Momentum": "Bullish"
})

summary = df.groupby("Regime Name").agg({
    "Daily Return": ["mean", "std"],
    "Volatility 20": "mean",
    "RSI": "mean",
    "Dist_SMA200": "mean",
    "Close": "count"
})

print(summary)

summary.to_csv("regime_summary.csv")

fig = px.scatter(
    df,
    x= df.index,
    y="Close",
    color="Regime Name",
    title="Market Regimes"
)

fig.show()

transition = pd.crosstab(
    df["Regime Name"],
    df["Regime Name"].shift(-1),
    normalize="index"
)

print(transition)

transition.to_csv("transition_matrix.csv")

fig = go.Figure(
    data=go.Heatmap(
        z=transition.values,
        x=transition.columns,
        y=transition.index
    )
)

fig.update_layout(title="Regime Transition Matrix")

fig.show()

df["Next Regime"] = df["Classifier Regime"].shift(-1)

df = df.dropna()

features = [
    "Daily Return",
    "5 Day Return",
    "20 Day Return",
    "Volatility 20",
    "RSI",
    "MACD",
    "Dist_SMA20",
    "Dist_SMA200",
    "Relative Volume"
]

X = df[features]
y = df["Next Regime"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))

importance = (
    pd.Series(
        rf.feature_importances_,
        index=features
    )
    .sort_values()
)

fig = px.bar(
    importance,
    orientation="h",
    title="Feature Importance"
)

fig.show()

joblib.dump(gmm, "gmm.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(rf, "next_regime_classifier.pkl")

bull_name = summary["Daily Return"]["mean"].idxmax()

weights = {
    "Bull": 1.0,
    "Momentum": 0.75,
    "Sideways": 0.25,
    "Bear": 0.0
}

df["Strategy Return"] = (
    df["Daily Return"] *
    df["Regime Name"].map(weights)
)

df["Buy and Hold"] = (1 + df["Daily Return"]).cumprod()
df["Strategy Equity"] = (1 + df["Strategy Return"]).cumprod()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Buy and Hold"],
        name="Buy & Hold"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Strategy Equity"],
        name="Regime Strategy"
    )
)

fig.update_layout(title="Strategy vs Buy & Hold")

fig.show()
