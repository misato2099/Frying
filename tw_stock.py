import os
import time
import html
import logging

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions


# ============================================================
# 基本設定
# ============================================================

TICKER = "2409.TW"
STOCK_NAME = "友達"
HISTORY_PERIOD = "2y"

BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0
BB_DDOF = 0

MA_LONG_WINDOW = 200
RISK_THRESHOLD_RATIO = 0.95

KD_WINDOW = 9
KD_ALPHA = 1 / 3
KD_INITIAL = 50.0

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 下載及整理行情
# ============================================================

def download_stock_data(max_retries=3, retry_delay=5):
    """
    下載友達未調整日線。

    auto_adjust=False：
    使用未經股息回溯調整的原始 OHLC，
    避免除息後布林通道與一般看盤軟體產生差異。

    repair=True：
    讓 yfinance 嘗試修正部分異常行情。
    """

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "正在下載 %s 歷史日線，第 %d 次嘗試",
                TICKER,
                attempt
            )

            df = yf.download(
                TICKER,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                actions=False,
                repair=True,
                keepna=False,
                progress=False,
                threads=False,
                timeout=30
            )

            df = normalize_yfinance_columns(df)

            if df.empty:
                raise ValueError("Yahoo Finance 回傳空資料")

            validate_stock_data(df)

            logger.info(
                "行情下載成功，共 %d 筆，最新日期為 %s",
                len(df),
                df.index[-1].strftime("%Y-%m-%d")
            )

            return df

        except Exception as error:
            last_error = error

            logger.warning(
                "第 %d 次下載失敗：%s",
                attempt,
                error
            )

            if attempt < max_retries:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"{TICKER} 行情下載重試後仍失敗：{last_error}"
    )


def normalize_yfinance_columns(df):
    """
    處理不同 yfinance 版本可能回傳的 MultiIndex 欄位。
    """

    if df is None:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        ticker_levels = []

        for level_number in range(df.columns.nlevels):
            values = df.columns.get_level_values(level_number)

            if TICKER in values:
                ticker_levels.append(level_number)

        if ticker_levels:
            ticker_level = ticker_levels[0]

            df = df.xs(
                TICKER,
                axis=1,
                level=ticker_level,
                drop_level=True
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
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"缺少必要欄位：{missing_columns}，"
            f"實際欄位為：{list(df.columns)}"
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

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.isna()]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    return df


def validate_stock_data(df):
    """
    檢查價格資料是否足夠且符合基本 OHLC 邏輯。
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

    invalid_price = (
        df[price_columns] <= 0
    ).any(axis=1)

    if invalid_price.any():
        raise ValueError(
            f"發現 {int(invalid_price.sum())} 筆"
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
        raise ValueError(
            f"發現 {int(invalid_ohlc.sum())} 筆"
            "不符合 OHLC 邏輯的行情"
        )


# ============================================================
# 技術指標計算
# ============================================================

def calculate_indicators(df):
    """
    計算：
    1. 布林通道 20 日、2 倍母體標準差
    2. 200 日均線及風險參考線
    3. 台灣常用 9 日 RSV、K/D 遞迴平滑
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
    # 長期均線及風險參考值
    # --------------------------------------------------------

    df["MA200"] = df["Close"].rolling(
        window=MA_LONG_WINDOW,
        min_periods=MA_LONG_WINDOW
    ).mean()

    df["Risk_Threshold"] = (
        df["MA200"]
        * RISK_THRESHOLD_RATIO
    )

    # --------------------------------------------------------
    # KD 指標
    # --------------------------------------------------------

    df["Min_Low_9"] = df["Low"].rolling(
        window=KD_WINDOW,
        min_periods=KD_WINDOW
    ).min()

    df["Max_High_9"] = df["High"].rolling(
        window=KD_WINDOW,
        min_periods=KD_WINDOW
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
            # RSV 尚未形成或價格區間為零時，
            # 維持前一日 K、D，不強制把 RSV 改成 50。
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
    # 真正的黃金交叉與死亡交叉
    # --------------------------------------------------------

    df["Golden_Cross"] = (
        (df["K"] > df["D"])
        & (df["K"].shift(1) <= df["D"].shift(1))
    )

    df["Death_Cross"] = (
        (df["K"] < df["D"])
        & (df["K"].shift(1) >= df["D"].shift(1))
    )

    required_latest_columns = [
        "MA20",
        "Upper_Band",
        "Lower_Band",
        "BB_Width_Pct",
        "BB_Position_Pct",
        "MA200",
        "Risk_Threshold",
        "RSV",
        "K",
        "D"
    ]

    df = df.dropna(
        subset=required_latest_columns
    )

    if df.empty:
        raise ValueError(
            "技術指標計算後沒有足夠的有效資料"
        )

    return df


# ============================================================
# 訊號判讀
# ============================================================

def get_kd_status(latest):
    """
    區分：
    1. 當日真正發生交叉
    2. 既有多方或空方排列
    """

    status = (
        f"RSV={latest['RSV']:.1f} / "
        f"K={latest['K']:.1f} / "
        f"D={latest['D']:.1f}"
    )

    if bool(latest["Golden_Cross"]):
        status += "（今日黃金交叉）"
    elif bool(latest["Death_Cross"]):
        status += "（今日死亡交叉）"
    elif latest["K"] > latest["D"]:
        status += "（K高於D，多方排列延續）"
    elif latest["K"] < latest["D"]:
        status += "（K低於D，空方排列延續）"
    else:
        status += "（K與D接近）"

    if latest["K"] >= 80 and latest["D"] >= 80:
        status += "［KD高檔超買］"
    elif latest["K"] <= 20 and latest["D"] <= 20:
        status += "［KD低檔超賣］"
    elif latest["K"] >= 80:
        status += "［K進入超買區］"
    elif latest["K"] <= 20:
        status += "［K進入超賣區］"

    return status


def get_bollinger_status(latest):
    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]

    if close > upper:
        return "收盤突破布林上軌，趨勢偏強，但短線乖離擴大"

    if close < lower:
        return "收盤跌破布林下軌，走勢偏弱，但須留意超跌反彈"

    if close >= middle:
        return "收盤位於布林中軌之上，短線結構偏多"

    return "收盤位於布林中軌之下，短線結構仍偏弱"


def get_technical_conclusion(latest):
    """
    使用明確規則產生可重現的技術觀察，
    不把投資決策完全交給生成式 AI。
    """

    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]
    risk_threshold = latest["Risk_Threshold"]

    if close < risk_threshold:
        return (
            "🚨 風險警示：收盤低於200日均線的95%參考值，"
            "技術結構偏弱，宜優先注意風險控管。"
        )

    if (
        close > upper
        and latest["K"] > latest["D"]
        and latest["K"] >= 80
    ):
        return (
            "🟠 技術觀察：趨勢偏強，但價格突破上軌且K值進入超買區，"
            "短線乖離較大，留意高檔震盪。"
        )

    if (
        close >= middle
        and latest["K"] > latest["D"]
    ):
        return (
            "🟢 技術觀察：收盤位於中軌之上，且K高於D，"
            "短線多方動能占優。"
        )

    if (
        close < lower
        and latest["K"] < latest["D"]
    ):
        return (
            "🔴 技術觀察：收盤低於下軌且K低於D，"
            "技術面偏弱，惟同時留意超跌反彈可能。"
        )

    if (
        close < middle
        and latest["K"] > latest["D"]
    ):
        return (
            "🟡 技術觀察：KD動能改善，但收盤仍低於布林中軌，"
            "目前較接近弱勢反彈，尚未確認趨勢轉強。"
        )

    if (
        close >= middle
        and latest["K"] < latest["D"]
    ):
        return (
            "🟡 技術觀察：價格仍在中軌之上，但KD動能轉弱，"
            "留意短線整理。"
        )

    return (
        "⚪ 技術觀察：布林通道與KD訊號分歧，"
        "目前以中性整理看待。"
    )


# ============================================================
# 報告產生
# ============================================================

def build_base_report(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    latest_date = df.index[-1].strftime("%Y-%m-%d")

    daily_change_pct = (
        latest["Close"]
        / previous["Close"]
        - 1
    ) * 100

    kd_status = get_kd_status(latest)
    bollinger_status = get_bollinger_status(latest)
    technical_conclusion = get_technical_conclusion(latest)

    volume = latest["Volume"]

    if pd.isna(volume):
        volume_text = "資料缺漏"
    else:
        volume_text = f"{int(volume):,} 股"

    report = f"""📋 友達（2409）每日量化診斷
📅 交易日期：{latest_date}
💰 收盤價格：{latest['Close']:.2f} 元（{daily_change_pct:+.2f}%）
📦 成交量：{volume_text}

📊 布林通道
上軌：{latest['Upper_Band']:.2f}
中軌：{latest['MA20']:.2f}
下軌：{latest['Lower_Band']:.2f}
帶寬：{latest['BB_Width_Pct']:.2f}%
通道位置：{latest['BB_Position_Pct']:.2f}%
訊號：{bollinger_status}

📈 KD指標
{kd_status}

🛡️ 長期風險參考
200日均線：{latest['MA200']:.2f}
風險參考值：{latest['Risk_Threshold']:.2f}

💡 綜合判讀
{technical_conclusion}

⚠️ 本摘要為公開行情的技術指標整理，不構成投資建議。"""

    return report, latest


def polish_report_with_gemini(base_report):
    """
    Gemini 只負責精簡及排版，
    不得修改任何數值，也不得新增買賣指令。
    """

    if not GEMINI_API_KEY:
        logger.warning(
            "未設定 GEMINI_API_KEY，直接使用 Python 報告"
        )
        return base_report

    genai.configure(api_key=GEMINI_API_KEY)

    prompt = f"""
你是一位繁體中文量化報告編輯。

請整理下方已由 Python 完成計算的友達光電技術摘要。

嚴格規則：
1. 不得修改任何日期、價格或技術指標數值。
2. 不得自行重新計算數值。
3. 不得輸出明確的買進、賣出或持有指令。
4. 不得加入目標價、停損價或報酬保證。
5. 保留原有段落、Emoji及風險聲明。
6. 僅可改善文字流暢度。
7. 輸出繁體中文。
8. 只輸出整理後報告，不要解釋處理過程。

原始報告：
{base_report}
"""

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    max_retries = 3
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "正在請求 Gemini 整理報告，第 %d 次嘗試",
                attempt
            )

            response = model.generate_content(prompt)

            response_text = getattr(
                response,
                "text",
                None
            )

            if not response_text:
                raise ValueError(
                    "Gemini 回傳內容為空"
                )

            logger.info("Gemini 報告整理成功")

            return response_text.strip()

        except (
            google_exceptions.ServiceUnavailable,
            google_exceptions.ResourceExhausted
        ) as error:
            logger.warning(
                "Gemini 暫時無法使用：%s",
                error
            )

            if attempt < max_retries:
                time.sleep(retry_delay)

        except Exception as error:
            logger.exception(
                "Gemini 發生未預期錯誤：%s",
                error
            )
            break

    logger.warning(
        "Gemini 整理失敗，改用 Python 原始報告"
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

    logger.info(
        "Telegram Token 長度：%d，Chat ID：%s",
        len(TG_TOKEN),
        TG_CHAT_ID
    )

    url = (
        f"https://api.telegram.org/"
        f"bot{TG_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TG_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Telegram 發送失敗，"
            f"HTTP {response.status_code}："
            f"{response.text}"
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
# 儲存計算明細
# ============================================================

def save_calculation_result(df):
    """
    保留最近 60 個交易日的計算明細，
    方便與券商看盤軟體逐日比對。
    """

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
        "RSV",
        "K",
        "D",
        "Golden_Cross",
        "Death_Cross"
    ]

    output_df = df[output_columns].tail(60)

    output_df.to_csv(
        "tw_stock_calculation.csv",
        encoding="utf-8-sig",
        index_label="Date"
    )

    logger.info(
        "計算明細已儲存為 tw_stock_calculation.csv"
    )


# ============================================================
# 主程式
# ============================================================

def run_analysis():
    logger.info(
        "開始執行友達 2409.TW 每日技術分析"
    )

    df = download_stock_data()
    df = calculate_indicators(df)

    save_calculation_result(df)

    base_report, latest = build_base_report(df)

    logger.info(
        "最新交易日：%s，收盤：%.2f，K：%.2f，D：%.2f",
        df.index[-1].strftime("%Y-%m-%d"),
        latest["Close"],
        latest["K"],
        latest["D"]
    )

    final_report = polish_report_with_gemini(
        base_report
    )

    print("\n" + final_report + "\n")

    send_tg_message(final_report)


if __name__ == "__main__":
    try:
        run_analysis()

    except Exception as error:
        logger.exception(
            "友達技術分析執行失敗：%s",
            error
        )

        failure_message = (
            "⚠️ 友達（2409）每日技術分析執行失敗。\n"
            f"錯誤類型：{type(error).__name__}\n"
            f"錯誤內容：{str(error)[:500]}"
        )

        try:
            if TG_TOKEN and TG_CHAT_ID:
                send_tg_message(failure_message)
        except Exception as telegram_error:
            logger.exception(
                "錯誤通知亦無法發送：%s",
                telegram_error
            )

        raise
