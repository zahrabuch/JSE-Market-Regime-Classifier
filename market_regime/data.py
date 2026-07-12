import pandas as pd
import numpy as np

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


df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")
df["Daily Return"] = df["Price"].pct_change()
print(df.info())
df = df.rename(columns={"Price": "Close"})


#FEATURES
def create_features(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["Dist_SMA20"] = (df["Close"] - df["SMA20"]) / df["SMA20"]
    df["Dist_SMA50"] = (df["Close"] - df["SMA50"]) / df["SMA50"]
    df["Dist_SMA200"] = (df["Close"] - df["SMA200"]) / df["SMA200"]

