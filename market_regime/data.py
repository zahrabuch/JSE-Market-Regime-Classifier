import pandas as pd
import numpy as np
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

df = pd.read_csv("JSE All Jamaican Composite Historical Data.csv")

print(df.info())

df["Change %"] = df["Change %"].str.replace("%", "", regex=False)

num_col = ["Price", "Open", "High", "Low", "Change %"]

for col in num_col:
    df[col] = df[col].str.replace(",", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.info())


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


df["Vol."] = df["Vol."].apply(convert_volume)


def clean_data(df):
    df["Change %"] = df["Change %"].str.replace("%", "", regex=False)

    num_col = ["Price", "Open", "High", "Low", "Change %"]

    for col in num_col:
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Vol."] = df["Vol."].apply(convert_volume)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")

    df["Daily Return"] = df["Price"].pct_change()
    df = df.rename(columns={"Price": "Close"})

    return df

def create_features(df):
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

    df["HL Range"] = (df["High"] - df["Low"])/df["Close"]

    df["Parkinson"] = (np.log(df["High"]/df["Low"])**2)
    df["Parkinson 20"] = df["Parkinson"].rolling(20).mean()

    df["OC Return"] = (df["Close"] - df["Open"])/df["Open"]

    df["Gap"] = (df["Open"] - df["Close"].shift(1))/df["Close"].shift(1)

    df["Body"] = (df["Open"] - df["Close"]).abs()
    df["Upper Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df["Volume SMA 20"] = df["Vol."].rolling(20).mean()
    df["Relative Volume"] = df["Vol."]/df["Volume SMA 20"]
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
        "Dist_SMA20",
        "Dist_SMA50",
        "Dist_SMA100",
        "Dist_SMA200",
        "Daily Return",
        "5 Day Return",
        "10 Day Return",
        "20 Day Return",
        "RSI",
        "MACD",
        "MACD_Sig",
        "Volatility 20",
        "ATR",
        "HL Range",
        "OC Return",
        "Gap",
        "Body",
        "Upper Wick",
        "Lower Wick",
        "Relative Volume",
        "OBV"
    ]

    X = df[features]
    scaler = StandardScaler()
    joblib.dump(scaler, "scaler.pkl")
    """
    scaler = joblib.load("scaler.pkl")
    X_new_scaled = scaler.transform(X_new)
    """
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=features,
        index=df.index
    )

    return X_scaled


def create_gmm():
    n_clusters = 4

    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=42
    )

df = load_csv()
print(df.info())
clean_df = clean_data(df)
print(clean_df.info())
feat_df = create_features(clean_df)
print(feat_df.info())
scaled_feat = scale_features(feat_df)
print(scaled_feat.head())
