import os
import sys
import time
import logging
import argparse
import datetime
import urllib3
import pytz
from bs4 import BeautifulSoup

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8-sig")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8-sig")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from google import genai
from google.genai import errors as genai_errors


# ============================================================
# 股票設定
# ============================================================

STOCKS = {
    "2409.TW": {
        "name": "友達",
        "short_name": "2409.TW",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "6285.TW": {
        "name": "啟碁",
        "short_name": "6285.TW",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "2327.TW": {
        "name": "國巨",
        "short_name": "2327.TW",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "2408.TW": {
        "name": "南亞科",
        "short_name": "2408.TW",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "6451.TW": {
        "name": "訊芯-KY",
        "short_name": "6451.TW",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "3081.TWO": {
        "name": "聯亞",
        "short_name": "3081.TWO",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "FCX": {
        "name": "自由港麥克莫蘭",
        "short_name": "FCX",
        "currency": "USD",
        "currency_label": "美元",
        "market": "美國",
    },
    "MRNA": {
        "name": "莫德納",
        "short_name": "MRNA",
        "currency": "USD",
        "currency_label": "美元",
        "market": "美國",
    },
    "SPCX": {
        "name": "SpaceX",
        "short_name": "SPCX",
        "currency": "USD",
        "currency_label": "美元",
        "market": "美國",
    }
}

HISTORY_PERIOD = "2y"

# 布林通道
BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0
BB_DDOF = 0

# 長期均線
MA_LONG_WINDOW = 200
RISK_THRESHOLD_RATIO = 0.95

# KD
KD_WINDOW = 9
KD_ALPHA = 1 / 3
KD_INITIAL = 50.0

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

def _load_env_file():
    env_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.expanduser("~/.env"),
        ".env",
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'").strip('"')
                            if k and not os.environ.get(k):
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Telegram
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 重試
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 相依套件檢查
# ============================================================

def check_dependencies():
    """
    yfinance repair=True 可能需要 SciPy。
    在下載前主動檢查，避免重試三次才發現缺少套件。
    """

    try:
        import scipy

        logger.info(
            "SciPy 已載入，版本：%s",
            scipy.__version__,
        )

    except ModuleNotFoundError as error:
        raise RuntimeError(
            "目前使用 yfinance repair=True，"
            "但執行環境未安裝 scipy。"
        ) from error


# ============================================================
# 行情下載
# ============================================================

def download_stock_data(ticker):
    """
    下載單一股票日線。
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "正在下載 %s 歷史日線，第 %d 次嘗試",
                ticker,
                attempt,
            )

            raw_df = yf.download(
                ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                back_adjust=False,
                actions=False,
                repair=False,
                keepna=False,
                progress=False,
                threads=False,
                timeout=30,
                multi_level_index=False,
            )

            df = normalize_yfinance_columns(
                raw_df,
                ticker,
            )

            validate_stock_data(
                df,
                ticker,
            )

            logger.info(
                "%s 行情下載成功，共 %d 筆，最新日期：%s",
                ticker,
                len(df),
                df.index[-1].strftime("%Y-%m-%d"),
            )

            return df

        except Exception as error:
            last_error = error

            logger.warning(
                "%s 第 %d 次行情下載或驗證失敗：%s",
                ticker,
                attempt,
                repr(error),
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"{ticker} 行情下載重試後仍失敗：{last_error}"
    )


def normalize_yfinance_columns(df, ticker):
    """
    將 yfinance 回傳資料整理成：
    Open、High、Low、Close、Volume。
    """

    if df is None or df.empty:
        raise ValueError(
            f"{ticker}：Yahoo Finance 回傳空資料"
        )

    df = df.copy()

    logger.info(
        "%s 原始欄位：%s",
        ticker,
        list(df.columns),
    )

    if isinstance(df.columns, pd.MultiIndex):
        logger.info(
            "%s 偵測到 MultiIndex 欄位",
            ticker,
        )

        # 常見格式：(Price, Ticker)
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(
                ticker,
                axis=1,
                level=-1,
                drop_level=True,
            )

        # 另一格式：(Ticker, Price)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(
                ticker,
                axis=1,
                level=0,
                drop_level=True,
            )

        else:
            # 移除只有一個唯一值的層級
            for level in reversed(
                range(df.columns.nlevels)
            ):
                unique_values = (
                    df.columns
                    .get_level_values(level)
                    .unique()
                )

                if len(unique_values) == 1:
                    df.columns = (
                        df.columns.droplevel(level)
                    )

    df.columns = [
        str(column).strip().title()
        for column in df.columns
    ]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{ticker} 缺少必要行情欄位："
            f"{missing_columns}；"
            f"實際欄位：{list(df.columns)}"
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df.loc[~df.index.isna()]
    df = df.sort_index()

    df = df.loc[
        ~df.index.duplicated(keep="last")
    ]

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    if df.empty:
        raise ValueError(
            f"{ticker} 行情標準化後沒有有效資料"
        )

    return df


def validate_stock_data(df, ticker):
    """
    驗證資料量及 OHLC 邏輯。
    """
    # 修正：移除 MA_LONG_WINDOW 的強制限制，僅確保短天期指標能計算
    minimum_rows = max(
        BB_WINDOW,
        KD_WINDOW,
    ) + 10

    if len(df) < minimum_rows:
        raise ValueError(
            f"{ticker} 有效日線只有 {len(df)} 筆，"
            f"至少需要 {minimum_rows} 筆"
        )

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    non_positive_rows = (
        df[price_columns] <= 0
    ).any(axis=1)

    if non_positive_rows.any():
        raise ValueError(
            f"{ticker} 發現 "
            f"{int(non_positive_rows.sum())} 筆"
            "小於或等於零的價格"
        )

    invalid_high = (
        (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["High"] < df["Low"])
    )

    invalid_low = (
        (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
        | (df["Low"] > df["High"])
    )

    invalid_ohlc = invalid_high | invalid_low

    if invalid_ohlc.any():
        invalid_dates = [
            date.strftime("%Y-%m-%d")
            for date in df.index[invalid_ohlc][:5]
        ]

        raise ValueError(
            f"{ticker} 發現 "
            f"{int(invalid_ohlc.sum())} 筆"
            "不符合 OHLC 邏輯的行情；"
            f"前幾筆日期：{invalid_dates}"
        )


# ============================================================
# 技術指標
# ============================================================

def calculate_indicators(df, ticker):
    """
    計算：
    1. 布林通道
    2. 200 日均線
    3. KD
    4. 黃金交叉及死亡交叉
    """

    df = df.copy()

    # --------------------------------------------------------
    # 布林通道
    # --------------------------------------------------------

    df["MA20"] = df["Close"].rolling(
        window=BB_WINDOW,
        min_periods=BB_WINDOW,
    ).mean()

    df["BB_STD"] = df["Close"].rolling(
        window=BB_WINDOW,
        min_periods=BB_WINDOW,
    ).std(ddof=BB_DDOF)

    df["Upper_Band"] = (
        df["MA20"]
        + BB_STD_MULTIPLIER * df["BB_STD"]
    )

    df["Lower_Band"] = (
        df["MA20"]
        - BB_STD_MULTIPLIER * df["BB_STD"]
    )

    band_range = (
        df["Upper_Band"]
        - df["Lower_Band"]
    )

    df["BB_Width_Pct"] = (
        band_range
        / df["MA20"]
        * 100
    )

    df["BB_Position_Pct"] = (
        (df["Close"] - df["Lower_Band"])
        / band_range.replace(0, np.nan)
        * 100
    )

    # --------------------------------------------------------
    # 長期均線
    # --------------------------------------------------------

    df["MA200"] = df["Close"].rolling(
        window=MA_LONG_WINDOW,
        min_periods=MA_LONG_WINDOW,
    ).mean()

    df["Risk_Threshold"] = (
        df["MA200"]
        * RISK_THRESHOLD_RATIO
    )

    # --------------------------------------------------------
    # KD
    # --------------------------------------------------------

    df["Min_Low_9"] = df["Low"].rolling(
        window=KD_WINDOW,
        min_periods=KD_WINDOW,
    ).min()

    df["Max_High_9"] = df["High"].rolling(
        window=KD_WINDOW,
        min_periods=KD_WINDOW,
    ).max()

    price_range_9 = (
        df["Max_High_9"]
        - df["Min_Low_9"]
    )

    df["RSV"] = (
        (df["Close"] - df["Min_Low_9"])
        / price_range_9.replace(0, np.nan)
        * 100
    )

    k_values = []
    d_values = []

    current_k = KD_INITIAL
    current_d = KD_INITIAL

    for rsv in df["RSV"]:
        if pd.isna(rsv):
            k_values.append(current_k)
            d_values.append(current_d)
            continue

        current_k = (
            (1 - KD_ALPHA) * current_k
            + KD_ALPHA * float(rsv)
        )

        current_d = (
            (1 - KD_ALPHA) * current_d
            + KD_ALPHA * current_k
        )

        k_values.append(current_k)
        d_values.append(current_d)

    df["K"] = k_values
    df["D"] = d_values

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["EMA_Fast"] = df["Close"].ewm(
        span=MACD_FAST, 
        adjust=False
    ).mean()

    df["EMA_Slow"] = df["Close"].ewm(
        span=MACD_SLOW, 
        adjust=False
    ).mean()

    df["DIF"] = df["EMA_Fast"] - df["EMA_Slow"]

    df["MACD"] = df["DIF"].ewm(
        span=MACD_SIGNAL, 
        adjust=False
    ).mean()

    df["OSC"] = df["DIF"] - df["MACD"]
    
    # --------------------------------------------------------
    # 波動率與結構性斷裂檢測 (ATR & Breakout Regime)
    # --------------------------------------------------------

    df["Prev_Close"] = df["Close"].shift(1)
    
    # 計算真實波動區間 (TR)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            (df["High"] - df["Prev_Close"]).abs(),
            (df["Low"] - df["Prev_Close"]).abs()
        )
    )
    
    # 14日平均真實波動 (ATR)
    df["ATR"] = df["TR"].rolling(
        window=14, 
        min_periods=14
    ).mean()

    # 爆量檢測：成交量大於 20 日均量的 2.5 倍
    df["Vol_MA20"] = df["Volume"].rolling(
        window=20, 
        min_periods=20
    ).mean()
    df["Vol_Spike"] = df["Volume"] > (df["Vol_MA20"] * 2.5)

    # 異常跳空檢測：缺口大於 1 倍昨日 ATR
    df["Gap_Up"] = (df["Low"] - df["High"].shift(1)) > df["ATR"].shift(1)
    df["Gap_Down"] = (df["Low"].shift(1) - df["High"]) > df["ATR"].shift(1)

    # 狀態標籤：同時滿足爆量與跳空
    df["Breakout_Regime"] = df["Vol_Spike"] & (df["Gap_Up"] | df["Gap_Down"])

    # --- 原有的 required_columns 與 dropna 區塊 ---
    # 請記得在 required_columns 列表加入 "ATR", "Breakout_Regime"
    
    # --------------------------------------------------------
    # 交叉
    # --------------------------------------------------------

    df["Golden_Cross"] = (
        (df["K"] > df["D"])
        & (df["K"].shift(1) <= df["D"].shift(1))
    )

    df["Death_Cross"] = (
        (df["K"] < df["D"])
        & (df["K"].shift(1) >= df["D"].shift(1))
    )

    required_columns = [
            "MA20",
            "BB_STD",
            "Upper_Band",
            "Lower_Band",
            "BB_Width_Pct",
            "BB_Position_Pct",
            "RSV",
            "K",
            "D",
            "DIF",
            "MACD",
            "ATR",
            "Vol_Spike",
            "Breakout_Regime",
            "OSC"
        ]

    df = df.dropna(
        subset=required_columns
    )

    if len(df) < 2:
        raise ValueError(
            f"{ticker} 技術指標計算後"
            "不足兩筆有效資料"
        )

    validate_latest_indicators(
        df.iloc[-1],
        ticker,
    )

    return df


def validate_latest_indicators(latest, ticker):
    columns = [
        "Close",
        "MA20",
        "BB_STD",
        "Upper_Band",
        "Lower_Band",
        "BB_Width_Pct",
        "BB_Position_Pct",
        "RSV",
        "K",
        "D",
        "DIF",
        "MACD",
        "ATR",
        "Vol_Spike",
        "Breakout_Regime",
        "OSC"
    ]

    for column in columns:
        value = latest[column]

        if pd.isna(value):
            raise ValueError(
                f"{ticker} 最新交易日的 "
                f"{column} 為空值"
            )

        if not np.isfinite(float(value)):
            raise ValueError(
                f"{ticker} 最新交易日的 "
                f"{column} 不是有限數值"
            )


# ============================================================
# 技術訊號
# ============================================================

def get_kd_status(latest):
    status = (
        f"RSV {latest['RSV']:.1f}｜"
        f"K {latest['K']:.1f}｜"
        f"D {latest['D']:.1f}"
    )

    if bool(latest["Golden_Cross"]):
        status += "｜今日黃金交叉"

    elif bool(latest["Death_Cross"]):
        status += "｜今日死亡交叉"

    elif latest["K"] > latest["D"]:
        status += "｜K高於D，多方排列延續"

    elif latest["K"] < latest["D"]:
        status += "｜K低於D，空方排列延續"

    else:
        status += "｜K與D接近"

    if (
        latest["K"] >= 80
        and latest["D"] >= 80
    ):
        status += "｜KD位於超買區"

    elif (
        latest["K"] <= 20
        and latest["D"] <= 20
    ):
        status += "｜KD位於超賣區"

    elif latest["K"] >= 80:
        status += "｜K進入超買區"

    elif latest["K"] <= 20:
        status += "｜K進入超賣區"

    return status

def get_macd_status(latest, previous):
    dif = float(latest["DIF"])
    macd = float(latest["MACD"])
    osc = float(latest["OSC"])
    prev_osc = float(previous["OSC"])
    prev_dif = float(previous["DIF"])
    prev_macd = float(previous["MACD"])

    status = (
        f"DIF {dif:.2f}｜"
        f"MACD {macd:.2f}｜"
        f"OSC {osc:.2f}"
    )

    # 判斷交叉
    if dif > macd and prev_dif <= prev_macd:
        status += "｜今日黃金交叉"
    elif dif < macd and prev_dif >= prev_macd:
        status += "｜今日死亡交叉"
    elif dif > 0 and macd > 0:
        status += "｜雙線零軸之上"
    elif dif < 0 and macd < 0:
        status += "｜雙線零軸之下"
    else:
        status += "｜多空交戰區"

    # 判斷動能柱狀體變化
    if osc > 0:
        if osc > prev_osc:
            status += "｜紅柱放大，多方動能增強"
        else:
            status += "｜紅柱縮小，多方動能減弱"
    elif osc < 0:
        if osc < prev_osc:
            status += "｜綠柱放大，空方動能增強"
        else:
            status += "｜綠柱縮小，空方動能減弱"

    return status

def get_bollinger_status(latest):
    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]

    if close > upper:
        return (
            "收盤突破上軌，趨勢偏強，"
            "但短線乖離擴大"
        )

    if close < lower:
        return (
            "收盤跌破下軌，技術結構偏弱，"
            "同時留意超跌反彈"
        )

    if close >= middle:
        return (
            "收盤位於中軌之上，"
            "短線價格結構偏多"
        )

    return (
        "收盤位於中軌之下，"
        "短線價格結構仍偏弱"
    )


def get_technical_conclusion(latest):
    close = float(latest["Close"])
    middle = float(latest["MA20"])
    upper = float(latest["Upper_Band"])
    lower = float(latest["Lower_Band"])
    k_value = float(latest["K"])
    d_value = float(latest["D"])
    risk_threshold = latest["Risk_Threshold"]

    # 擷取突破狀態參數
    is_breakout = bool(latest.get("Breakout_Regime", False))
    gap_up = bool(latest.get("Gap_Up", False))
    gap_down = bool(latest.get("Gap_Down", False))

    # 1. 優先檢驗結構性斷裂狀態
    if is_breakout:
        if gap_up:
            return (
                "🔥 【結構性突破】出現爆量與顯著向上跳空！\n"
                "市場定價邏輯可能已轉變，傳統震盪指標易出現高檔鈍化。\n"
                "技術面宜轉為趨勢跟隨，並以近期跳空缺口或 ATR 動態防守點作為評估參考。"
            )
        elif gap_down:
            return (
                "⚠️ 【結構性斷裂】出現爆量與顯著向下跳空！\n"
                "底部支撐結構可能失效，需留意超賣指標失真風險，\n"
                "應優先審視基本面變數，並嚴格執行風險價值控管機制。"
            )

    # 2. 檢驗長期風險參考
    if not pd.isna(risk_threshold) and close < float(risk_threshold):
        return (
            "🚨 收盤低於200日均線的95%參考值，"
            "長期價格結構偏弱，宜優先關注風險控管。"
        )

    # 3. 常規技術訊號判定
    if close > upper and k_value > d_value and k_value >= 80:
        return (
            "🟠 價格突破布林上軌且KD偏多，"
            "但K值已進入超買區，短線乖離及震盪風險上升。"
        )

    if close < lower and k_value < d_value and k_value <= 20:
        return (
            "🔵 價格跌破布林下軌且KD偏空，"
            "但K值已進入超賣區，反映短線可能存在超跌現象，需留意技術性反彈契機。"
        )

    if close >= middle and k_value > d_value:
        return "🟢 收盤位於布林中軌之上，且K高於D，短線多方動能占優。"

    if close < lower and k_value < d_value:
        return "🔴 收盤低於布林下軌，且K低於D，技術結構偏弱。"

    if close < middle and k_value > d_value:
        return (
            "🟡 KD動能改善，但收盤仍低於布林中軌，"
            "較接近弱勢反彈，尚未確認趨勢轉強。"
        )

    if close >= middle and k_value < d_value:
        return "🟡 價格仍在布林中軌之上，但KD動能轉弱，留意短線整理。"

    return "⚪ 布林通道與KD訊號分歧，目前以中性整理看待。"


# ============================================================
# 單一股票報告
# ============================================================

def build_stock_report(
    ticker,
    stock_info,
    df,
):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    latest_date = (
        df.index[-1].strftime("%Y-%m-%d")
    )

    daily_change = (
        latest["Close"]
        - previous["Close"]
    )

    daily_change_pct = (
        latest["Close"]
        / previous["Close"]
        - 1
    ) * 100

    if pd.isna(latest["Volume"]):
        volume_text = "資料缺漏"
    else:
        volume_text = (
            f"{int(latest['Volume']):,} 股"
        )

    currency_label = (
        stock_info["currency_label"]
    )

    ma200_val = latest['MA200']
    risk_threshold_val = latest['Risk_Threshold']

    ma200_str = f"{float(ma200_val):.2f}" if not pd.isna(ma200_val) else "資料不足"
    risk_threshold_str = f"{float(risk_threshold_val):.2f}" if not pd.isna(risk_threshold_val) else "資料不足"

    atr_val = latest.get('ATR', np.nan)
    atr_str = f"{float(atr_val):.2f}" if not pd.isna(atr_val) else "資料不足"
    regime_str = "觸發" if bool(latest.get("Breakout_Regime", False)) else "未觸發"

    report = f"""【{stock_info['name']}｜{ticker}】
📅 交易日期：{latest_date}
💰 收盤：{latest['Close']:.2f} {currency_label}
📉 漲跌：{daily_change:+.2f} {currency_label}（{daily_change_pct:+.2f}%）
📦 成交量：{volume_text}

📊 布林通道
上軌：{latest['Upper_Band']:.2f}
中軌：{latest['MA20']:.2f}
下軌：{latest['Lower_Band']:.2f}
帶寬：{latest['BB_Width_Pct']:.2f}%
通道位置：{latest['BB_Position_Pct']:.2f}%
訊號：{get_bollinger_status(latest)}

📈 KD指標
{get_kd_status(latest)}

📉 MACD指標
{get_macd_status(latest, previous)}

🛡️ 長期風險與波動率參考
200日均線：{ma200_str}
200日均線95%參考值：{risk_threshold_str}
14日真實波幅 (ATR)：{atr_str}
結構斷裂警示：{regime_str}

💡 技術觀察
{get_technical_conclusion(latest)}"""

    return report

def save_calculation_result(
    ticker,
    df,
):
    output_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA20",
        "BB_STD",
        "Upper_Band",
        "Lower_Band",
        "BB_Width_Pct",
        "BB_Position_Pct",
        "MA200",
        "Risk_Threshold",
        "Min_Low_9",
        "Max_High_9",
        "RSV",
        "K",
        "D",
        "DIF",
        "MACD",
        "OSC",
        "Golden_Cross",
        "Death_Cross",
        "ATR",
        "Vol_Spike",
        "Breakout_Regime",
    ]

    output_df = (
        df[output_columns]
        .tail(60)
        .copy()
    )

    safe_ticker = ticker.replace(".", "_")

    filename = (
        f"{safe_ticker}_calculation.csv"
    )

    output_df.to_csv(
        filename,
        encoding="utf-8-sig",
        index_label="Date",
        float_format="%.6f",
    )

    logger.info(
        "%s 計算明細已儲存：%s",
        ticker,
        filename,
    )

# ============================================================
# Gemini
# ============================================================

def polish_combined_report_with_gemini(
    base_report,
    active_stocks=None,
):
    """
    Gemini 僅整理文字。
    失敗時直接使用 Python 原始報告。
    """

    if not GEMINI_API_KEY:
        logger.warning(
            "未設定 GEMINI_API_KEY，"
            "直接使用 Python 原始報告"
        )

        return base_report

    prompt = f"""
你是一位繁體中文量化報告編輯。

請整理下方已由 Python 完成計算的各檔股票技術摘要。

嚴格規則：
1. 不得修改任何日期或數值。
2. 不得重新計算指標。
3. 不得增加買進、賣出或持有指令。
4. 不得增加目標價、停損價或報酬保證。
5. 必須保留各檔股票各自的交易日期。
6. 必須保留風險聲明。
7. 僅改善文字流暢度及排版。
8. 使用台灣繁體中文。
9. 只輸出整理後完整報告。

原始報告：

{base_report}
"""

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    except Exception as error:
        logger.exception(
            "無法建立 Gemini Client：%s",
            repr(error),
        )

        return base_report

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "正在請求 Gemini 整理技術報告，"
                "第 %d 次嘗試",
                attempt,
            )

            response = (
                client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
            )

            response_text = getattr(
                response,
                "text",
                None,
            )

            if not response_text:
                raise ValueError(
                    "Gemini 回傳內容為空"
                )

            response_text = (
                response_text.strip()
            )

            if active_stocks:
                required_phrases = [
                    stock_info["name"][:2]
                    for stock_info in active_stocks.values()
                ] + ["布林", "KD", "不構成投資建議"]
            else:
                required_phrases = ["布林", "KD", "不構成投資建議"]

            missing_phrases = [
                phrase
                for phrase in required_phrases
                if phrase not in response_text
            ]

            if missing_phrases:
                logger.warning(
                    "Gemini 報告缺少必要內容：%s，"
                    "改用 Python 原始報告",
                    missing_phrases,
                )

                return base_report

            logger.info(
                "Gemini 技術報告整理成功"
            )

            return response_text

        except genai_errors.ServerError as error:
            logger.warning(
                "Gemini 伺服器錯誤，"
                "第 %d 次嘗試：%s",
                attempt,
                repr(error),
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY_SECONDS
                )

        except genai_errors.ClientError as error:
            logger.error(
                "Gemini 用戶端錯誤，"
                "請檢查 API Key、模型及額度：%s",
                repr(error),
            )

            break

        except Exception as error:
            logger.exception(
                "Gemini 未預期錯誤：%s",
                repr(error),
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY_SECONDS
                )
            else:
                break

    logger.warning(
        "Gemini 整理失敗，"
        "改用 Python 原始報告"
    )

    return base_report


# ============================================================
# Telegram
# ============================================================

def send_tg_message(message):
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.info(
            "未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，略過 Telegram 發送"
        )
        return

    url = (
        f"https://api.telegram.org/"
        f"bot{TG_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Telegram 發送失敗，"
            f"HTTP {response.status_code}："
            f"{response.text[:1000]}"
        )

    response_json = response.json()

    if not response_json.get("ok"):
        raise RuntimeError(
            "Telegram API 回報失敗："
            f"{response_json}"
        )

    logger.info(
        "Telegram 技術摘要已成功發送"
    )


# ============================================================
# 交易日檢核（國定假日、週末、天災停市）
# ============================================================

def get_easter_date(year: int) -> datetime.date:
    """計算指定年份的復活節週日 (Gregorian Easter)"""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def get_us_market_holidays(year: int) -> dict[datetime.date, str]:
    """計算美股 (NYSE / NASDAQ) 全年法定休市日"""
    holidays = {}

    # 1. 元旦 (New Year's Day)
    nyd = datetime.date(year, 1, 1)
    if nyd.weekday() == 6:  # 週日 -> 週一補假
        holidays[datetime.date(year, 1, 2)] = "New Year's Day (Observed)"
    elif nyd.weekday() != 5:  # 週六不提前（NYSE慣例）
        holidays[nyd] = "New Year's Day"

    # 2. 馬丁路德金紀念日 (Martin Luther King, Jr. Day - 1月第3個週一)
    jan1 = datetime.date(year, 1, 1)
    first_mon_jan = jan1 + datetime.timedelta(days=(0 - jan1.weekday() + 7) % 7)
    holidays[first_mon_jan + datetime.timedelta(weeks=2)] = "Martin Luther King, Jr. Day"

    # 3. 華盛頓誕辰 / 總統日 (Presidents' Day - 2月第3個週一)
    feb1 = datetime.date(year, 2, 1)
    first_mon_feb = feb1 + datetime.timedelta(days=(0 - feb1.weekday() + 7) % 7)
    holidays[first_mon_feb + datetime.timedelta(weeks=2)] = "Presidents' Day"

    # 4. 耶穌受難日 (Good Friday - 復活節前2天)
    good_friday = get_easter_date(year) - datetime.timedelta(days=2)
    holidays[good_friday] = "Good Friday"

    # 5. 陣亡將士紀念日 (Memorial Day - 5月最後一個週一)
    may31 = datetime.date(year, 5, 31)
    last_mon_may = may31 - datetime.timedelta(days=(may31.weekday() - 0 + 7) % 7)
    holidays[last_mon_may] = "Memorial Day"

    # 6. 六月節全國獨立日 (Juneteenth - 6月19日)
    june19 = datetime.date(year, 6, 19)
    if june19.weekday() == 5:
        holidays[datetime.date(year, 6, 18)] = "Juneteenth (Observed)"
    elif june19.weekday() == 6:
        holidays[datetime.date(year, 6, 20)] = "Juneteenth (Observed)"
    else:
        holidays[june19] = "Juneteenth National Independence Day"

    # 7. 美國獨立日 (Independence Day - 7月4日)
    july4 = datetime.date(year, 7, 4)
    if july4.weekday() == 5:
        holidays[datetime.date(year, 7, 3)] = "Independence Day (Observed)"
    elif july4.weekday() == 6:
        holidays[datetime.date(year, 7, 5)] = "Independence Day (Observed)"
    else:
        holidays[july4] = "Independence Day"

    # 8. 勞動節 (Labor Day - 9月第1個週一)
    sep1 = datetime.date(year, 9, 1)
    first_mon_sep = sep1 + datetime.timedelta(days=(0 - sep1.weekday() + 7) % 7)
    holidays[first_mon_sep] = "Labor Day"

    # 9. 感恩節 (Thanksgiving Day - 11月第4個週四)
    nov1 = datetime.date(year, 11, 1)
    first_thu_nov = nov1 + datetime.timedelta(days=(3 - nov1.weekday() + 7) % 7)
    holidays[first_thu_nov + datetime.timedelta(weeks=3)] = "Thanksgiving Day"

    # 10. 聖誕節 (Christmas Day - 12月25日)
    dec25 = datetime.date(year, 12, 25)
    if dec25.weekday() == 5:
        holidays[datetime.date(year, 12, 24)] = "Christmas Day (Observed)"
    elif dec25.weekday() == 6:
        holidays[datetime.date(year, 12, 26)] = "Christmas Day (Observed)"
    else:
        holidays[dec25] = "Christmas Day"

    return holidays


_TWSE_HOLIDAYS_CACHE = {}

def parse_twse_date(date_str: str) -> datetime.date | None:
    """解析證交所回傳的日期字串 (支援西元或民國格式)"""
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            pass

    parts = date_str.replace("/", "-").split("-")
    if len(parts) == 3 and parts[0].isdigit():
        year = int(parts[0])
        if year < 1900:  # 民國年
            year += 1911
        try:
            return datetime.date(year, int(parts[1]), int(parts[2]))
        except ValueError:
            pass
    return None


def fetch_twse_holidays(year: int = None) -> dict[datetime.date, str]:
    """
    從台灣證券交易所 (TWSE) 官方 API 取得年度休市行事曆
    """
    global _TWSE_HOLIDAYS_CACHE
    if year in _TWSE_HOLIDAYS_CACHE:
        return _TWSE_HOLIDAYS_CACHE[year]

    try:
        url = "https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule?response=json"
        if year:
            url += f"&queryYear={year}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            holidays = {}
            for row in data.get("data", []):
                if len(row) >= 2:
                    date_str, name = str(row[0]).strip(), str(row[1]).strip()
                    desc = str(row[2]).strip() if len(row) > 2 else ""

                    # 過濾純粹標記開市/最後交易日的非休市項目
                    if "開始交易" in name and "無交易" not in name and "休市" not in name and "放假" not in name:
                        continue
                    if "最後交易" in name and "無交易" not in name and "休市" not in name and "放假" not in name:
                        continue

                    dt = parse_twse_date(date_str)
                    if dt:
                        holidays[dt] = f"{name} ({desc})" if desc else name

            if holidays:
                _TWSE_HOLIDAYS_CACHE[year] = holidays
                return holidays
    except Exception as e:
        logger.warning("取得證交所休市日 API 失敗: %s", repr(e))

    return _TWSE_HOLIDAYS_CACHE.get(year, {})


def check_taiwan_typhoon_closure(target_date: datetime.date) -> tuple[bool, str]:
    """
    檢查指定日期台灣是否因天然災害（颱風假）導致台北市停班而使證交所休市。
    依證交所規定：台北市政府宣布停止上班即全日休市。
    """
    today = datetime.date.today()
    if target_date != today:
        return False, ""

    try:
        url = "https://www.dgpa.gov.tw/typh/daily/nds.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, verify=False, timeout=8)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            text = soup.get_text()

            for city in ["臺北市", "台北市"]:
                if city in text:
                    idx = text.find(city)
                    snippet = text[idx : idx + 100]
                    negative_keywords = ["未達停止上班", "未達停班", "照常上班", "正常上班", "未列入警戒區"]
                    has_negative = any(neg in snippet for neg in negative_keywords)

                    if ("停止上班" in snippet or "停班" in snippet) and not has_negative:
                        return True, f"行政院人事行政總處公告【{city}】停止上班（颱風/天災全日停市）"
    except Exception as e:
        logger.warning("查詢人事行政總處天然災害停班公告失敗: %s", repr(e))

    return False, ""


def is_trading_day(target_date: datetime.date, market: str = "tw") -> tuple[bool, str]:
    """
    檢核目標日期在指定市場是否為交易日。
    market: 'tw' (台股), 'us' (美股), 'all' (全市場)
    回傳: (is_open: bool, reason: str)
    """
    weekday = target_date.weekday()
    if weekday == 5:
        return False, f"{target_date} 為週六（週末非交易日）"
    if weekday == 6:
        return False, f"{target_date} 為週日（週末非交易日）"

    market_lower = market.lower()

    if market_lower in ("tw", "taiwan", "台股"):
        is_typhoon, typhoon_reason = check_taiwan_typhoon_closure(target_date)
        if is_typhoon:
            return False, f"{target_date} 台股休市：{typhoon_reason}"

        twse_holidays = fetch_twse_holidays(target_date.year)
        if target_date in twse_holidays:
            return False, f"{target_date} 台股休市：{twse_holidays[target_date]}"

        return True, f"{target_date} 為台股正常交易日"

    elif market_lower in ("us", "usa", "美股"):
        us_holidays = get_us_market_holidays(target_date.year)
        if target_date in us_holidays:
            return False, f"{target_date} 美股休市：{us_holidays[target_date]}"

        return True, f"{target_date} 為美股正常交易日"

    elif market_lower in ("all", "全部"):
        tw_open, tw_reason = is_trading_day(target_date, "tw")
        us_open, us_reason = is_trading_day(target_date, "us")
        if not tw_open and not us_open:
            return False, f"{target_date} 台股與美股皆休市（台股: {tw_reason}; 美股: {us_reason}）"

        return True, f"{target_date} 部份或全部市場開市（台股: {'開市' if tw_open else '休市'}, 美股: {'開市' if us_open else '休市'}）"

    return True, "未知市場，預設正常執行"


# ============================================================
# 主流程
# ============================================================

def run_analysis(market="all", force=False):
    if market.lower() in ("tw", "taiwan", "台股"):
        target_stocks = {
            k: v for k, v in STOCKS.items()
            if v.get("market") == "台灣"
        }
        market_label = "台股"
    elif market.lower() in ("us", "usa", "美股"):
        target_stocks = {
            k: v for k, v in STOCKS.items()
            if v.get("market") == "美國"
        }
        market_label = "美股"
    else:
        target_stocks = STOCKS
        market_label = "全市場"

    logger.info(
        "開始執行 %s 每日技術分析，共 %d 檔標的",
        market_label,
        len(target_stocks),
    )

    # 交易日檢核
    tw_tz = pytz.timezone("Asia/Taipei")
    us_tz = pytz.timezone("America/New_York")
    now_tw = datetime.datetime.now(tw_tz)
    now_us = datetime.datetime.now(us_tz)

    tw_date = now_tw.date()
    # 若台灣已進入週一上午，而紐約尚處於週日夜間，美股目標交易日為該週一 (tw_date)
    if now_tw.weekday() == 0 and now_us.weekday() == 6:
        us_date = now_tw.date()
    else:
        us_date = now_us.date()

    if not force:
        if market.lower() in ("tw", "taiwan", "台股"):
            is_open, reason = is_trading_day(tw_date, market="tw")
            if not is_open:
                logger.info(
                    "📅 交易日檢核：今日 (%s) %s 休市／非交易日 [%s]，停止執行分析。",
                    tw_date,
                    market_label,
                    reason,
                )
                print(
                    f"\n📅 今日 ({tw_date}) {market_label} 為非交易日（{reason}），停止執行分析。\n",
                    flush=True,
                )
                return
            logger.info("📅 交易日檢核：今日 (%s) 為台股正常交易日", tw_date)

        elif market.lower() in ("us", "usa", "美股"):
            is_open, reason = is_trading_day(us_date, market="us")
            if not is_open:
                logger.info(
                    "📅 交易日檢核：今日 (%s) %s 休市／非交易日 [%s]，停止執行分析。",
                    us_date,
                    market_label,
                    reason,
                )
                print(
                    f"\n📅 今日 ({us_date}) {market_label} 為非交易日（{reason}），停止執行分析。\n",
                    flush=True,
                )
                return
            logger.info("📅 交易日檢核：今日 (%s) 為美股正常交易日", us_date)

        else:
            tw_open, tw_reason = is_trading_day(tw_date, market="tw")
            us_open, us_reason = is_trading_day(us_date, market="us")

            if not tw_open and not us_open:
                logger.info(
                    "📅 交易日檢核：今日台股 (%s: %s) 與美股 (%s: %s) 皆休市，停止執行分析。",
                    tw_date,
                    tw_reason,
                    us_date,
                    us_reason,
                )
                print(
                    f"\n📅 今日台美市場皆非交易日（台股: {tw_reason}；美股: {us_reason}），停止執行分析。\n",
                    flush=True,
                )
                return

            if not tw_open:
                logger.info("📅 交易日檢核：今日台股休市 (%s)，僅分析美股標的", tw_reason)
                target_stocks = {
                    k: v for k, v in target_stocks.items()
                    if v.get("market") == "美國"
                }
            elif not us_open:
                logger.info("📅 交易日檢核：今日美股休市 (%s)，僅分析台股標的", us_reason)
                target_stocks = {
                    k: v for k, v in target_stocks.items()
                    if v.get("market") == "台灣"
                }
    else:
        logger.info("⚡ 已啟用 --force 參數，跳過交易日檢核。")

    check_dependencies()

    stock_reports = []
    failed_stocks = []

    for ticker, stock_info in target_stocks.items():
        try:
            logger.info(
                "開始處理 %s %s",
                stock_info["name"],
                ticker,
            )

            df = download_stock_data(ticker)

            df = calculate_indicators(
                df,
                ticker,
            )

            save_calculation_result(
                ticker,
                df,
            )

            latest = df.iloc[-1]

            logger.info(
                "%s 最新交易日=%s，"
                "收盤=%.2f，"
                "布林中軌=%.2f，"
                "K=%.2f，"
                "D=%.2f",
                ticker,
                df.index[-1].strftime("%Y-%m-%d"),
                latest["Close"],
                latest["MA20"],
                latest["K"],
                latest["D"],
            )

            report = build_stock_report(
                ticker,
                stock_info,
                df,
            )

            stock_reports.append(report)

        except Exception as error:
            logger.exception(
                "%s 分析失敗",
                ticker,
            )

            failed_stocks.append(
                f"{ticker}："
                f"{type(error).__name__}，"
                f"{str(error)[:300]}"
            )

    if not stock_reports:
        raise RuntimeError(
            f"{market_label} 標的均分析失敗"
        )

    combined_report = (
        f"📋 每日量化技術摘要（{market_label}）"
        + "\n\n"
        + "\n\n━━━━━━━━━━━━━\n\n".join(
            stock_reports
        )
    )

    if failed_stocks:
        combined_report += (
            "\n\n⚠️ 部分標的分析失敗\n"
            + "\n".join(failed_stocks)
        )

    combined_report += (
        "\n\n📌 日期說明："
        "台股與美股交易時區不同，"
        "各檔股票使用各自最新完成交易日。"
        "\n\n⚠️ 本摘要為公開行情與技術指標整理，"
        "不構成投資建議。"
    )

    final_report = (
        polish_combined_report_with_gemini(
            combined_report,
            active_stocks=target_stocks,
        )
    )

    print(
        "\n"
        + final_report
        + "\n",
        flush=True,
    )

    send_tg_message(
        final_report
    )


# ============================================================
# 程式入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="每日股票量化技術分析"
    )
    parser.add_argument(
        "--market",
        choices=["tw", "us", "all"],
        default="all",
        help="指定分析市場：tw(台股), us(美股), all(全部)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制執行分析，忽略休市日與非交易日檢查",
    )
    cli_args = parser.parse_args()

    try:
        run_analysis(
            market=cli_args.market,
            force=cli_args.force,
        )

    except Exception as error:
        logger.exception(
            "技術分析執行失敗"
        )

        print(
            f"FATAL_ERROR_TYPE="
            f"{type(error).__name__}",
            flush=True,
        )

        print(
            f"FATAL_ERROR_MESSAGE="
            f"{repr(error)}",
            flush=True,
        )

        failure_message = (
            f"⚠️ {cli_args.market} 技術分析執行失敗。\n"
            f"錯誤類型：{type(error).__name__}\n"
            f"錯誤內容：{str(error)[:800]}"
        )

        try:
            if TG_TOKEN and TG_CHAT_ID:
                send_tg_message(
                    failure_message
                )

        except Exception as telegram_error:
            logger.exception(
                "錯誤通知也無法傳送：%s",
                repr(telegram_error),
            )

        raise
