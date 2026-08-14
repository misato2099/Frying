import os
import time
import logging

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
        "name": "友達光電",
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
                repair=True,
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

    # 檢核長天期資料是否存在，改用 pd.isna 以確保相容性
    if not pd.isna(risk_threshold) and close < float(risk_threshold):
        return (
            "🚨 收盤低於200日均線的95%參考值，"
            "長期價格結構偏弱，宜優先關注風險控管。"
        )

    if (
        close > upper
        and k_value > d_value
        and k_value >= 80
    ):
        return (
            "🟠 價格突破布林上軌且KD偏多，"
            "但K值已進入超買區，"
            "短線乖離及震盪風險上升。"
        )

    if (
        close < lower
        and k_value < d_value
        and k_value <= 20
    ):
        return (
            "🔵 價格跌破布林下軌且KD偏空，"
            "但K值已進入超賣區，"
            "反映短線可能存在超跌現象，需留意技術性反彈契機。"
        )

    if (
        close >= middle
        and k_value > d_value
    ):
        return (
            "🟢 收盤位於布林中軌之上，"
            "且K高於D，短線多方動能占優。"
        )

    if (
        close < lower
        and k_value < d_value
    ):
        return (
            "🔴 收盤低於布林下軌，"
            "且K低於D，技術結構偏弱。"
        )

    if (
        close < middle
        and k_value > d_value
    ):
        return (
            "🟡 KD動能改善，"
            "但收盤仍低於布林中軌，"
            "較接近弱勢反彈，尚未確認趨勢轉強。"
        )

    if (
        close >= middle
        and k_value < d_value
    ):
        return (
            "🟡 價格仍在布林中軌之上，"
            "但KD動能轉弱，留意短線整理。"
        )

    return (
        "⚪ 布林通道與KD訊號分歧，"
        "目前以中性整理看待。"
    )


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

    # 強化空值與型別防呆
    ma200_str = f"{float(ma200_val):.2f}" if not pd.isna(ma200_val) else "資料不足"
    risk_threshold_str = f"{float(risk_threshold_val):.2f}" if not pd.isna(risk_threshold_val) else "資料不足"

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

🛡️ 長期風險參考
200日均線：{ma200_str}
200日均線95%參考值：{risk_threshold_str}

💡 技術觀察
{get_technical_conclusion(latest)}"""

    return report


# ============================================================
# Gemini
# ============================================================

def polish_combined_report_with_gemini(
    base_report,
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

請整理下方已由 Python 完成計算的
友達光電與自由港麥克莫蘭技術摘要。

嚴格規則：
1. 不得修改任何日期或數值。
2. 不得重新計算指標。
3. 不得增加買進、賣出或持有指令。
4. 不得增加目標價、停損價或報酬保證。
5. 必須保留兩檔股票各自的交易日期。
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
                "正在請求 Gemini 整理雙股報告，"
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

            required_phrases = [
                "友達",
                "啟碁",
                "國巨",
                "訊芯-KY",
                "聯亞",
                "FCX",
                "莫德納",
                "SPACX",
                "布林",
                "KD",
                "不構成投資建議",
            ]

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
                "Gemini 雙股報告整理成功"
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
    if not TG_TOKEN:
        raise ValueError(
            "未設定 TELEGRAM_BOT_TOKEN"
        )

    if not TG_CHAT_ID:
        raise ValueError(
            "未設定 TELEGRAM_CHAT_ID"
        )

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
        "Telegram 雙股技術摘要已成功發送"
    )


# ============================================================
# 儲存明細
# ============================================================

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
        "Golden_Cross",
        "Death_Cross",
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
# 主流程
# ============================================================

def run_analysis():
    logger.info(
        "開始執行友達與 FCX 每日技術分析"
    )

    check_dependencies()

    stock_reports = []
    failed_stocks = []

    for ticker, stock_info in STOCKS.items():
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
            "友達與 FCX 均分析失敗"
        )

    combined_report = (
        "📋 每日量化技術摘要\n\n"
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
        "兩檔股票使用各自最新完成交易日。"
        "\n\n⚠️ 本摘要為公開行情與技術指標整理，"
        "不構成投資建議。"
    )

    final_report = (
        polish_combined_report_with_gemini(
            combined_report
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
    try:
        run_analysis()

    except Exception as error:
        logger.exception(
            "雙股技術分析執行失敗"
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
            "⚠️ 友達與FCX每日技術分析執行失敗。\n"
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
