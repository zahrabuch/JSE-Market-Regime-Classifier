import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

#FEATURES
def create_features(df):
    df["5 Day Return"] = df["Price"].pct_change(5)
    df["10 Day Return"] = df["Price"].pct_change(10)
    df["20 Day Return"] = df["Price"].pct_change(20)

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA50"] = df["Close"].rolling(100).mean()
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
