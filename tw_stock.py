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
    # 200 日均線及風險參考
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
    # KD
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
            # RSV 尚未形成，或 9 日高低價相同時，
            # 維持前一期 K、D。
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
    # 交叉訊號
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
        "MA200",
        "Risk_Threshold",
        "RSV",
        "K",
        "D"
    ]

    df = df.dropna(
        subset=required_columns
    )

    if len(df) < 2:
        raise ValueError(
            "技術指標計算後不足兩筆有效資料"
        )

    validate_latest_indicators(df.iloc[-1])

    return df


def validate_latest_indicators(latest: pd.Series) -> None:
    """
    確認最新指標皆為有限數字。
    """

    columns = [
        "Close",
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
        "D"
    ]

    for column in columns:
        value = latest[column]

        if pd.isna(value):
            raise ValueError(
                f"最新交易日的 {column} 為空值"
            )

        if not np.isfinite(float(value)):
            raise ValueError(
                f"最新交易日的 {column} 不是有限數值"
            )


# ============================================================
# 訊號解讀
# ============================================================

def get_kd_status(latest: pd.Series) -> str:
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
        status += "（K與D相同或極為接近）"

    if (
        latest["K"] >= 80
        and latest["D"] >= 80
    ):
        status += "［KD位於超買區］"

    elif (
        latest["K"] <= 20
        and latest["D"] <= 20
    ):
        status += "［KD位於超賣區］"

    elif latest["K"] >= 80:
        status += "［K進入超買區］"

    elif latest["K"] <= 20:
        status += "［K進入超賣區］"

    return status


def get_bollinger_status(latest: pd.Series) -> str:
    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]

    if close > upper:
        return (
            "收盤突破布林上軌，趨勢偏強，"
            "但短線乖離及波動風險上升"
        )

    if close < lower:
        return (
            "收盤跌破布林下軌，技術結構偏弱，"
            "同時留意超跌反彈可能"
        )

    if close >= middle:
        return (
            "收盤位於布林中軌之上，"
            "短線價格結構偏多"
        )

    return (
        "收盤位於布林中軌之下，"
        "短線價格結構仍偏弱"
    )


def get_technical_conclusion(latest: pd.Series) -> str:
    """
    使用固定規則產生可重複的技術觀察。
    不輸出個人化買賣指令。
    """

    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]

    k_value = latest["K"]
    d_value = latest["D"]

    risk_threshold = latest["Risk_Threshold"]

    if close < risk_threshold:
        return (
            "🚨 風險警示：收盤低於200日均線的95%參考值，"
            "長期價格結構偏弱，宜優先關注風險控管。"
        )

    if (
        close > upper
        and k_value > d_value
        and k_value >= 80
    ):
        return (
            "🟠 技術觀察：價格突破布林上軌且KD偏多，"
            "但K值已進入超買區，短線乖離較大。"
        )

    if (
        close >= middle
        and k_value > d_value
    ):
        return (
            "🟢 技術觀察：收盤位於布林中軌之上，"
            "且K高於D，短線多方動能占優。"
        )

    if (
        close < lower
        and k_value < d_value
    ):
        return (
            "🔴 技術觀察：收盤低於布林下軌，"
            "且K低於D，技術面偏弱。"
        )

    if (
        close < middle
        and k_value > d_value
    ):
        return (
            "🟡 技術觀察：KD動能改善，"
            "但收盤仍低於布林中軌，"
            "目前較接近弱勢反彈，尚未確認趨勢轉強。"
        )

    if (
        close >= middle
        and k_value < d_value
    ):
        return (
            "🟡 技術觀察：價格仍在布林中軌之上，"
            "但KD動能轉弱，留意短線整理。"
        )

    return (
        "⚪ 技術觀察：布林通道與KD訊號分歧，"
        "目前以中性整理看待。"
    )


# ============================================================
# 報告內容
# ============================================================

def build_base_report(
    df: pd.DataFrame
) -> tuple[str, pd.Series]:
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    latest_date = (
        df.index[-1].strftime("%Y-%m-%d")
    )

    previous_date = (
        df.index[-2].strftime("%Y-%m-%d")
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

    kd_status = get_kd_status(latest)

    bollinger_status = (
        get_bollinger_status(latest)
    )

    conclusion = (
        get_technical_conclusion(latest)
    )

    if pd.isna(latest["Volume"]):
        volume_text = "資料缺漏"
    else:
        volume_text = (
            f"{int(latest['Volume']):,} 股"
        )

    report = f"""📋 友達光電（2409）每日量化診斷

📅 資料日期
最新交易日：{latest_date}
前一交易日：{previous_date}

💰 價格與成交量
收盤：{latest['Close']:.2f} 元
漲跌：{daily_change:+.2f} 元（{daily_change_pct:+.2f}%）
成交量：{volume_text}

📊 布林通道（20日，2倍標準差）
上軌：{latest['Upper_Band']:.2f}
中軌：{latest['MA20']:.2f}
下軌：{latest['Lower_Band']:.2f}
帶寬：{latest['BB_Width_Pct']:.2f}%
通道位置：{latest['BB_Position_Pct']:.2f}%
訊號：{bollinger_status}

📈 KD指標（9日RSV）
{kd_status}

🛡️ 長期風險參考
200日均線：{latest['MA200']:.2f}
200日均線95%參考值：{latest['Risk_Threshold']:.2f}

💡 綜合判讀
{conclusion}

⚠️ 本摘要為公開行情與技術指標整理，不構成投資建議。"""

    return report, latest


# ============================================================
# Google Gemini
# ============================================================

def polish_report_with_gemini(
    base_report: str
) -> str:
    """
    Gemini 僅整理文字及排版。

    如果未設定金鑰、API 失敗、額度不足或模型不可用，
    直接回傳 Python 原始報告。
    """

    if not GEMINI_API_KEY:
        logger.warning(
            "未設定 GEMINI_API_KEY，"
            "直接使用 Python 原始報告"
        )

        return base_report

    prompt = f"""
你是一位繁體中文量化報告編輯。

請整理下方已由 Python 完成計算的友達光電技術摘要。

必須遵守以下規則：
1. 不得修改任何日期、價格或技術指標數值。
2. 不得自行重新計算任何數值。
3. 不得輸出明確的買進、賣出或持有指令。
4. 不得加入目標價、停損價或報酬保證。
5. 保留原有段落、Emoji及風險聲明。
6. 僅可改善文字流暢度與排版。
7. 使用台灣繁體中文。
8. 只輸出整理後的完整報告。
9. 不要解釋你的處理過程。

原始報告如下：

{base_report}
"""

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    except Exception as error:
        logger.exception(
            "無法建立 Gemini Client：%s",
            repr(error)
        )

        return base_report

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "正在請求 Gemini 整理報告，"
                "第 %d 次嘗試",
                attempt
            )

            response = (
                client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
            )

            response_text = getattr(
                response,
                "text",
                None
            )

            if not response_text:
                raise ValueError(
                    "Gemini 回傳內容為空"
                )

            response_text = (
                response_text.strip()
            )

            if not validate_gemini_response(
                base_report,
                response_text
            ):
                logger.warning(
                    "Gemini 報告未通過數值檢查，"
                    "改用 Python 原始報告"
                )

                return base_report

            logger.info(
                "Gemini 報告整理成功"
            )

            return response_text

        except genai_errors.ServerError as error:
            logger.warning(
                "Gemini 伺服器錯誤，"
                "第 %d 次嘗試：%s",
                attempt,
                repr(error)
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY_SECONDS
                )

        except genai_errors.ClientError as error:
            logger.error(
                "Gemini 用戶端錯誤，"
                "請檢查 API Key、模型權限及額度：%s",
                repr(error)
            )

            break

        except Exception as error:
            logger.exception(
                "Gemini 未預期錯誤：%s",
                repr(error)
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


def validate_gemini_response(
    base_report: str,
    gemini_report: str
) -> bool:
    """
    最基本的防護：
    確認 Gemini 沒有遺失重要欄位。

    完整數值仍以 CSV 及 Python 計算結果為準。
    """

    required_phrases = [
        "友達",
        "布林",
        "KD",
        "200日",
        "不構成投資建議"
    ]

    for phrase in required_phrases:
        if phrase not in gemini_report:
            logger.warning(
                "Gemini 回應缺少必要文字：%s",
                phrase
            )

            return False

    return True


# ============================================================
# Telegram
# ============================================================

def send_tg_message(message: str) -> None:
    """
    傳送 Telegram 文字訊息。
    """

    if not TG_TOKEN:
        raise ValueError(
            "未設定 TELEGRAM_BOT_TOKEN"
        )

    if not TG_CHAT_ID:
        raise ValueError(
            "未設定 TELEGRAM_CHAT_ID"
        )

    logger.info(
        "準備傳送 Telegram 訊息，"
        "Token 長度=%d，Chat ID=%s",
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
            f"{response.text[:1000]}"
        )

    try:
        response_json = response.json()

    except ValueError as error:
        raise RuntimeError(
            "Telegram 回傳內容不是有效 JSON："
            f"{response.text[:1000]}"
        ) from error

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

def save_calculation_result(
    df: pd.DataFrame
) -> None:
    """
    保留最近 60 個交易日的計算結果，
    方便與券商軟體或其他資料源逐日比對。
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
        "Min_Low_9",
        "Max_High_9",
        "RSV",
        "K",
        "D",
        "Golden_Cross",
        "Death_Cross"
    ]

    output_df = (
        df[output_columns]
        .tail(60)
        .copy()
    )

    output_df.to_csv(
        "tw_stock_calculation.csv",
        encoding="utf-8-sig",
        index_label="Date",
        float_format="%.6f"
    )

    logger.info(
        "計算明細已儲存："
        "tw_stock_calculation.csv"
    )


# ============================================================
# 主流程
# ============================================================

def run_analysis() -> None:
    logger.info(
        "開始執行友達 2409.TW 每日技術分析"
    )

    df = download_stock_data()

    df = calculate_indicators(df)

    save_calculation_result(df)

    base_report, latest = (
        build_base_report(df)
    )

    logger.info(
        "最新交易日=%s，"
        "收盤=%.2f，"
        "布林中軌=%.2f，"
        "K=%.2f，"
        "D=%.2f",
        df.index[-1].strftime("%Y-%m-%d"),
        latest["Close"],
        latest["MA20"],
        latest["K"],
        latest["D"]
    )

    final_report = (
        polish_report_with_gemini(
            base_report
        )
    )

    print(
        "\n"
        + final_report
        + "\n",
        flush=True
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
            "友達技術分析執行失敗"
        )

        # 確保 GitHub Actions 最下方可清楚看到錯誤
        print(
            f"FATAL_ERROR_TYPE="
            f"{type(error).__name__}",
            flush=True
        )

        print(
            f"FATAL_ERROR_MESSAGE="
            f"{repr(error)}",
            flush=True
        )

        failure_message = (
            "⚠️ 友達光電（2409）每日技術分析執行失敗。\n"
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
                repr(telegram_error)
            )

        raise
