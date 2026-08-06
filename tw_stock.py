import os
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from google import genai
from google.genai import errors as genai_errors


# ============================================================
# 基本參數
# ============================================================

TICKER = "2409.TW"
STOCK_NAME = "友達光電"
HISTORY_PERIOD = "2y"

# 布林通道參數
BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0

# 0 = 母體標準差，通常較接近常見看盤軟體
# 1 = 樣本標準差，為 pandas std() 預設值
BB_DDOF = 0

# 長期均線及風險參考
MA_LONG_WINDOW = 200
RISK_THRESHOLD_RATIO = 0.95

# KD 參數
KD_WINDOW = 9
KD_ALPHA = 1 / 3
KD_INITIAL = 50.0

# API 金鑰及 Telegram 資訊
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Gemini 模型
GEMINI_MODEL = "gemini-2.5-flash"

# 網路重試設定
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 行情下載
# ============================================================

def download_stock_data() -> pd.DataFrame:
    """
    從 Yahoo Finance 下載友達日線。

    auto_adjust=False：
    使用未經股息回溯調整的原始 OHLC。

    repair=True：
    讓 yfinance 嘗試修正部分異常資料。

    multi_level_index=False：
    優先要求單層欄位，避免 Close 等欄位成為 DataFrame。
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "正在下載 %s 歷史日線，第 %d 次嘗試",
                TICKER,
                attempt
            )

            raw_df = yf.download(
                TICKER,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                back_adjust=False,
                actions=False,
                repair=True,
                keepna=False,
                progress=False,
                threads=False,
                timeout=30,
                multi_level_index=False
            )

            df = normalize_yfinance_columns(raw_df)

            validate_stock_data(df)

            logger.info(
                "行情下載成功，共 %d 筆，最新日期：%s",
                len(df),
                df.index[-1].strftime("%Y-%m-%d")
            )

            return df

        except Exception as error:
            last_error = error

            logger.warning(
                "第 %d 次行情下載或驗證失敗：%s",
                attempt,
                repr(error)
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"{TICKER} 行情下載重試後仍失敗：{last_error}"
    )


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 yfinance 回傳資料整理成：
    Open、High、Low、Close、Volume。
    """

    if df is None or df.empty:
        raise ValueError("Yahoo Finance 回傳空資料")

    df = df.copy()

    logger.info(
        "yfinance 原始欄位：%s",
        list(df.columns)
    )

    # 相容仍然回傳 MultiIndex 的 yfinance 版本
    if isinstance(df.columns, pd.MultiIndex):
        logger.info("偵測到 MultiIndex 欄位，開始標準化")

        # 常見格式：(Price, Ticker)
        if TICKER in df.columns.get_level_values(-1):
            df = df.xs(
                TICKER,
                axis=1,
                level=-1,
                drop_level=True
            )

        # 另一種格式：(Ticker, Price)
        elif TICKER in df.columns.get_level_values(0):
            df = df.xs(
                TICKER,
                axis=1,
                level=0,
                drop_level=True
            )

        else:
            # 嘗試移除只有一個唯一值的欄位層級
            for level in reversed(range(df.columns.nlevels)):
                unique_values = (
                    df.columns
                    .get_level_values(level)
                    .unique()
                )

                if len(unique_values) == 1:
                    df.columns = df.columns.droplevel(level)

    df.columns = [
        str(column).strip().title()
        for column in df.columns
    ]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"缺少必要行情欄位：{missing_columns}；"
            f"實際欄位：{list(df.columns)}"
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df.index = pd.to_datetime(
        df.index,
        errors="coerce"
    )

    # 技術指標只需要交易日期，移除時區資訊
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    # 移除無效日期
    df = df.loc[~df.index.isna()]

    # 依日期排序並移除重複日期
    df = df.sort_index()
    df = df.loc[
        ~df.index.duplicated(keep="last")
    ]

    # OHLC 不完整的交易日不參與計算
    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    if df.empty:
        raise ValueError(
            "行情標準化後沒有有效資料"
        )

    logger.info(
        "標準化後欄位：%s",
        list(df.columns)
    )

    return df


def validate_stock_data(df: pd.DataFrame) -> None:
    """
    驗證資料量、價格及 OHLC 邏輯。
    """

    minimum_rows = max(
        MA_LONG_WINDOW,
        BB_WINDOW,
        KD_WINDOW
    ) + 10

    if len(df) < minimum_rows:
        raise ValueError(
            f"有效日線只有 {len(df)} 筆，"
            f"至少需要 {minimum_rows} 筆"
        )

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    non_positive_rows = (
        df[price_columns] <= 0
    ).any(axis=1)

    if non_positive_rows.any():
        raise ValueError(
            f"發現 {int(non_positive_rows.sum())} 筆"
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
            timestamp.strftime("%Y-%m-%d")
            for timestamp in df.index[invalid_ohlc][:5]
        ]

        raise ValueError(
            f"發現 {int(invalid_ohlc.sum())} 筆"
            f"不符合 OHLC 邏輯的行情；"
            f"前幾筆日期：{invalid_dates}"
        )


# ============================================================
# 技術指標計算
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算：
    1. 20 日、2 倍標準差布林通道
    2. 200 日均線及風險參考值
    3. 9 日 RSV 與遞迴平滑 KD
    4. 黃金交叉與死亡交叉
    """

    df = df.copy()

    # --------------------------------------------------------
    # 布林通道
    # --------------------------------------------------------

    df["MA20"] = df["Close"].rolling(
        window=BB_WINDOW,
        min_periods=BB_WINDOW
    ).mean()

    df["BB_STD"] = df["Close"].rolling(
        window=BB_WINDOW,
        min_periods=BB_WINDOW
    ).std(ddof=BB_DDOF)

    df["Upper_Band"] = (
        df["MA20"]
        + BB_STD_MULTIPLIER * df["BB_STD"]
    )

    df["Lower_Band"] = (
        df["
