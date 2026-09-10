import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from google import genai
from google.genai import errors as genai_errors


# ============================================================
# 股票與市場設定
# ============================================================

STOCKS = {
    # ---------------- 台股標的 ----------------
    "2409.TW": {
        "name": "友達光電",
        "short_name": "友達",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "6285.TW": {
        "name": "啟碁科技",
        "short_name": "啟碁",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "6451.TW": {
        "name": "訊芯-KY",
        "short_name": "訊芯-KY",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "3081.TWO": {
        "name": "聯亞光電",
        "short_name": "聯亞",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "2327.TW": {
        "name": "國巨",
        "short_name": "國巨",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    "2408.TW": {
        "name": "南亞科技",
        "short_name": "南亞科",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",
    },
    # ---------------- 歐股標的 (可依需求新增/擴充) ----------------
    "ASML.AS": {
        "name": "艾司摩爾",
        "short_name": "ASML",
        "currency": "EUR",
        "currency_label": "歐元",
        "market": "歐洲",
    },
    "TKA.DE": {
        "name": "蒂森克虜伯",
        "short_name": "TKA",
        "currency": "EUR",
        "currency_label": "歐元",
        "market": "歐洲",
    },
    # ---------------- 美股標的 ----------------
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
    },
    "GPRO": {
        "name": "GoPro",
        "short_name": "GPRO",
        "currency": "USD",
        "currency_label": "美元",
        "market": "美國",
    },
}

HISTORY_PERIOD = "2y"

# 布林通道參數
BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0
BB_DDOF = 0

# 長期均線參數（支援次新股自適應）
MA_LONG_WINDOW = 200
RISK_THRESHOLD_RATIO = 0.95

# KD 參數
KD_WINDOW = 9
KD_ALPHA = 1 / 3
KD_INITIAL = 50.0

# MACD 參數
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR 與結構性斷裂參數
ATR_WINDOW = 14
VOL_MA_WINDOW = 20
VOL_SPIKE_RATIO = 2.5
GAP_ATR_RATIO = 1.0

# ------------------------------------------------------------
# 專屬週期分析設定（友達 2409.TW 與 FCX）
# ------------------------------------------------------------
CYCLE_SPECS = {
    "2409.TW": {
        "title": "友達 (2409.TW) 專屬面板景氣與時間週期量化分析",
        "up_cycle_mean": 45.8,
        "up_cycle_std": 25.5,
        "down_cycle_mean": 37.7,
        "down_cycle_std": 5.7,
        "peak_to_peak_mean": 71.7,
        "peak_to_peak_std": 17.8,
        "industry_notes": "面板報價景氣週期、產能利用率（85%頂/70%底）、一年處分三廠邁向輕資產與MicroLED/CPO轉型。",
    },
    "FCX": {
        "title": "自由港麥克莫蘭 (FCX) 專屬商品景氣與時間週期量化分析",
        "up_cycle_mean": 52.0,
        "up_cycle_std": 18.0,
        "down_cycle_mean": 35.0,
        "down_cycle_std": 10.0,
        "peak_to_peak_mean": 88.0,
        "peak_to_peak_std": 15.0,
        "industry_notes": "LME/COMEX全球銅庫存水位、銅精礦加工費(TC/RC)、AI資料中心與電網電氣化長期結構性供需。",
    },
}

# Telegram 訊息上限設定（官方上限為 4096，保留安全緩衝區設為 3800）
MAX_TG_MESSAGE_LENGTH = 3800

# Gemini 設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Telegram 設定
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 重試參數
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3


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
    檢查必要相依套件。
    """
    try:
        import scipy
        logger.info("SciPy 已載入，版本：%s", scipy.__version__)
    except ModuleNotFoundError:
        logger.warning("環境未安裝 scipy，部分 yfinance 修復功能可能受限")


# ============================================================
# 行情下載與標準化（三重容錯自適應機制）
# ============================================================

def get_browser_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }


def download_from_yahoo_v8_api(ticker):
    """
    第三層保底：直接呼叫 Yahoo Finance 官方底層 v8 Chart API。
    繞過 yfinance 封裝庫對於國際代碼 (.TW / .TWO / .AS / .DE / .L) 的解析 bug。
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "2y", "interval": "1d"}
    headers = get_browser_headers()

    response = requests.get(url, params=params, headers=headers, timeout=25)
    if response.status_code != 200:
        raise RuntimeError(f"Yahoo v8 API HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()
    chart = data.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo v8 API 回報錯誤：{error}")

    result = chart.get("result")
    if not result or len(result) == 0:
        raise ValueError(f"{ticker}：Yahoo v8 API 未回傳結果")

    res = result[0]
    timestamps = res.get("timestamp")
    if not timestamps:
        raise ValueError(f"{ticker}：時間戳記為空")

    indicators = res.get("indicators", {}).get("quote", [{}])[0]

    df = pd.DataFrame(
        {
            "Open": indicators.get("open"),
            "High": indicators.get("high"),
            "Low": indicators.get("low"),
            "Close": indicators.get("close"),
            "Volume": indicators.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise ValueError(f"{ticker}：v8 API 解析後無有效報價數據")

    return df


def download_stock_data(ticker):
    """
    自適應三重下載機制：
    1. yf.Ticker(ticker).history() （處理台股與歐股最穩定的 yfinance 原生方法）
    2. yf.download() （關閉 repair=True，避免國際股票被錯誤過濾）
    3. Yahoo Finance v8 底層 JSON API （終極保底，保證 100% 取得）
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        # 方法 1: yf.Ticker.history()
        try:
            logger.info("正在下載 %s 行情 (方法1: Ticker.history，嘗試 %d/%d)", ticker, attempt, MAX_RETRIES)
            t = yf.Ticker(ticker)
            raw_df = t.history(period=HISTORY_PERIOD, interval="1d", auto_adjust=False)
            if raw_df is not None and not raw_df.empty:
                df = normalize_yfinance_columns(raw_df, ticker)
                validate_stock_data(df, ticker)
                logger.info("%s 下載成功 (方法1: Ticker.history，筆數：%d)", ticker, len(df))
                return df
        except Exception as err:
            last_error = err
            logger.warning("%s 方法1 失敗：%s", ticker, str(err)[:150])

        # 方法 2: yf.download (禁用易造成非美股出錯的 repair=True)
        try:
            logger.info("正在下載 %s 行情 (方法2: yf.download，嘗試 %d/%d)", ticker, attempt, MAX_RETRIES)
            raw_df = yf.download(
                ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                repair=False,
                threads=False,
                timeout=25,
            )
            if raw_df is not None and not raw_df.empty:
                df = normalize_yfinance_columns(raw_df, ticker)
                validate_stock_data(df, ticker)
                logger.info("%s 下載成功 (方法2: yf.download，筆數：%d)", ticker, len(df))
                return df
        except Exception as err:
            last_error = err
            logger.warning("%s 方法2 失敗：%s", ticker, str(err)[:150])

        # 方法 3: 官方底層 v8 API 直連（終極保底）
        try:
            logger.info("正在下載 %s 行情 (方法3: Yahoo v8 API 直連，嘗試 %d/%d)", ticker, attempt, MAX_RETRIES)
            raw_df = download_from_yahoo_v8_api(ticker)
            if raw_df is not None and not raw_df.empty:
                df = normalize_yfinance_columns(raw_df, ticker)
                validate_stock_data(df, ticker)
                logger.info("%s 下載成功 (方法3: Yahoo v8 API，筆數：%d)", ticker, len(df))
                return df
        except Exception as err:
            last_error = err
            logger.warning("%s 方法3 失敗：%s", ticker, str(err)[:150])

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(f"{ticker} 行情下載重試後仍失敗：{last_error}")


def normalize_yfinance_columns(df, ticker):
    """
    將回傳資料整理為標準欄位：
    Open、High、Low、Close、Volume。
    相容 yfinance 單層與 MultiIndex 欄位結構。
    """
    if df is None or df.empty:
        raise ValueError(f"{ticker}：回傳空資料")

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            for level in reversed(range(df.columns.nlevels)):
                unique_values = df.columns.get_level_values(level).unique()
                if len(unique_values) == 1:
                    df.columns = df.columns.droplevel(level)

    df.columns = [str(column).strip().title() for column in df.columns]

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{ticker} 缺少必要行情欄位：{missing_columns}")

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df.loc[~df.index.isna()]
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        raise ValueError(f"{ticker} 行情標準化後沒有有效資料")

    return df


def validate_stock_data(df, ticker):
    """
    驗證資料筆數與 OHLC 邏輯。
    針對新上市股票（如 SpaceX / SPCX 上市天數不足 200 日）：
    動態放寬最低門檻至短中期指標所需筆數（35 筆），避免拋出異常中斷執行。
    """
    core_minimum_rows = max(BB_WINDOW, KD_WINDOW, MACD_SLOW, ATR_WINDOW) + 5

    if len(df) < core_minimum_rows:
        raise ValueError(f"{ticker} 有效日線只有 {len(df)} 筆，至少需要 {core_minimum_rows} 筆")

    price_columns = ["Open", "High", "Low", "Close"]
    non_positive_rows = (df[price_columns] <= 0).any(axis=1)

    if non_positive_rows.any():
        raise ValueError(f"{ticker} 發現 {int(non_positive_rows.sum())} 筆 <=0 價格")

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
        invalid_dates = [d.strftime("%Y-%m-%d") for d in df.index[invalid_ohlc][:5]]
        raise ValueError(f"{ticker} 發現異常 OHLC 數據，日期示例：{invalid_dates}")


# ============================================================
# 技術指標與結構性突破計算（新上市股票自適應機制）
# ============================================================

def calculate_indicators(df, ticker):
    """
    計算：
    1. 布林通道 (20, 2, ddof=0)
    2. 長期均線與風險參考（自適應上市天數：>=200日取MA200，<200日取累計/MA60，相容SpaceX）
    3. KD 指標 (9, 3, 3)
    4. MACD 指標 (12, 26, 9)
    5. ATR(14) 與 結構性斷裂/突破檢測 (Breakout Regime)
    """
    df = df.copy()

    # 1. 布林通道
    df["MA20"] = df["Close"].rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    df["BB_STD"] = df["Close"].rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std(ddof=BB_DDOF)
    df["Upper_Band"] = df["MA20"] + BB_STD_MULTIPLIER * df["BB_STD"]
    df["Lower_Band"] = df["MA20"] - BB_STD_MULTIPLIER * df["BB_STD"]
    band_range = df["Upper_Band"] - df["Lower_Band"]
    df["BB_Width_Pct"] = (band_range / df["MA20"]) * 100
    df["BB_Position_Pct"] = ((df["Close"] - df["Lower_Band"]) / band_range.replace(0, np.nan)) * 100

    # 2. 長期均線與風險參考（次新股自適應支援）
    if len(df) >= MA_LONG_WINDOW:
        df["MA_Long"] = df["Close"].rolling(window=MA_LONG_WINDOW, min_periods=MA_LONG_WINDOW).mean()
        df["MA_Long_Label"] = "200日均線"
    else:
        effective_window = min(60, len(df))
        min_p = max(15, effective_window // 2)
        df["MA_Long"] = df["Close"].rolling(window=effective_window, min_periods=min_p).mean()
        df["MA_Long_Label"] = f"上市累計均線({effective_window}日)"
        logger.info("%s 上市天數不足 200 日（共 %d 筆），自動切換為：%s", ticker, len(df), df["MA_Long_Label"].iloc[-1])

    df["MA200"] = df["MA_Long"]
    df["Risk_Threshold"] = df["MA_Long"] * RISK_THRESHOLD_RATIO

    # 3. KD 指標 (9, 3, 3)
    df["Min_Low_9"] = df["Low"].rolling(window=KD_WINDOW, min_periods=KD_WINDOW).min()
    df["Max_High_9"] = df["High"].rolling(window=KD_WINDOW, min_periods=KD_WINDOW).max()
    price_range_9 = df["Max_High_9"] - df["Min_Low_9"]
    df["RSV"] = ((df["Close"] - df["Min_Low_9"]) / price_range_9.replace(0, np.nan)) * 100

    k_values = []
    d_values = []
    current_k = KD_INITIAL
    current_d = KD_INITIAL

    for rsv in df["RSV"]:
        if pd.isna(rsv):
            k_values.append(current_k)
            d_values.append(current_d)
            continue
        current_k = (1 - KD_ALPHA) * current_k + KD_ALPHA * float(rsv)
        current_d = (1 - KD_ALPHA) * current_d + KD_ALPHA * current_k
        k_values.append(current_k)
        d_values.append(current_d)

    df["K"] = k_values
    df["D"] = d_values

    df["Golden_Cross"] = (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))
    df["Death_Cross"] = (df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))

    # 4. MACD 指標 (12, 26, 9)
    ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["MACD"] = df["DIF"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["OSC"] = df["DIF"] - df["MACD"]

    # 5. ATR 與 結構性斷裂/突破檢測
    prev_close = df["Close"].shift(1)
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(window=ATR_WINDOW, min_periods=ATR_WINDOW).mean()

    vol_ma20 = df["Volume"].rolling(window=VOL_MA_WINDOW, min_periods=VOL_MA_WINDOW).mean()
    df["Vol_Spike"] = df["Volume"] > (vol_ma20 * VOL_SPIKE_RATIO)
    prev_atr = df["ATR"].shift(1)
    df["Gap_Up"] = (df["Low"] - prev_high) > (prev_atr * GAP_ATR_RATIO)
    df["Gap_Down"] = (prev_low - df["High"]) > (prev_atr * GAP_ATR_RATIO)
    df["Breakout_Regime"] = df["Vol_Spike"] & (df["Gap_Up"] | df["Gap_Down"])

    required_columns = [
        "MA20", "BB_STD", "Upper_Band", "Lower_Band", "BB_Width_Pct", "BB_Position_Pct",
        "MA_Long", "Risk_Threshold", "RSV", "K", "D", "DIF", "MACD", "OSC", "ATR"
    ]
    df = df.dropna(subset=required_columns)

    if len(df) < 2:
        raise ValueError(f"{ticker} 技術指標計算後不足兩筆有效資料")

    return df


# ============================================================
# 週期與時間窗口量化分析模組（友達與 FCX 專屬）
# ============================================================

def calculate_cycle_metrics(df, ticker):
    """
    依據歷史峰谷特徵，計算當前波段運行天數、週期進度與時間轉折敏感窗。
    """
    if ticker not in CYCLE_SPECS:
        return None

    spec = CYCLE_SPECS[ticker]
    df_recent = df.tail(120).copy()

    window_pivot = 7
    df_recent["Local_Min"] = df_recent["Low"] == df_recent["Low"].rolling(window=window_pivot * 2 + 1, center=True).min()
    df_recent["Local_Max"] = df_recent["High"] == df_recent["High"].rolling(window=window_pivot * 2 + 1, center=True).max()

    troughs = df_recent[df_recent["Local_Min"]].index
    peaks = df_recent[df_recent["Local_Max"]].index

    last_trough = troughs[-1] if len(troughs) > 0 else df_recent.index[0]
    last_peak = peaks[-1] if len(peaks) > 0 else df_recent.index[0]

    latest_date = df_recent.index[-1]

    if last_trough >= last_peak:
        phase = "上升推升段 (Trough to Peak)"
        start_date = last_trough
        start_price = df_recent.loc[last_trough, "Low"]
        elapsed_days = (latest_date - last_trough).days
        mean_duration = spec["up_cycle_mean"]
        std_duration = spec["up_cycle_std"]
    else:
        phase = "回檔整理段 (Peak to Trough)"
        start_date = last_peak
        start_price = df_recent.loc[last_peak, "High"]
        elapsed_days = (latest_date - last_peak).days
        mean_duration = spec["down_cycle_mean"]
        std_duration = spec["down_cycle_std"]

    progress_pct = (elapsed_days / mean_duration) * 100 if mean_duration > 0 else 0
    days_to_mean = mean_duration - elapsed_days

    if elapsed_days >= (mean_duration - std_duration * 0.5):
        turn_alert = f"⚠️ 已進入轉折敏感時間窗！目前運行 {elapsed_days} 天（歷史平均 {mean_duration:.1f} ± {std_duration:.1f} 天），需防範波段轉折。"
    else:
        turn_alert = f"⏳ 處於常態運行期，距離歷史平均滿足點尚有約 {max(0, int(days_to_mean))} 天。"

    cycle_report = f"""🔄 【{spec['title']}】
* **當前波段所屬階段**：{phase}
* **本輪波段起點**：{start_date.strftime('%Y-%m-%d')}（起點價位：{start_price:.2f}）
* **累計運行天數**：第 {elapsed_days} 個交易日（平均時長：{mean_duration:.1f} 天，標準差：±{std_duration:.1f} 天）
* **週期滿足進度**：{progress_pct:.1f}%
* **時間轉折敏感窗**：{turn_alert}
* **產業與景氣座標**：{spec['industry_notes']}"""

    return cycle_report


# ============================================================
# 技術訊號字串判定
# ============================================================

def get_kd_status(latest):
    status = f"RSV {latest['RSV']:.1f}｜K {latest['K']:.1f}｜D {latest['D']:.1f}"
    if bool(latest.get("Golden_Cross", False)):
        status += "｜今日黃金交叉"
    elif bool(latest.get("Death_Cross", False)):
        status += "｜今日死亡交叉"
    elif latest["K"] > latest["D"]:
        status += "｜K高於D，多方排列延續"
    elif latest["K"] < latest["D"]:
        status += "｜K低於D，空方排列延續"
    else:
        status += "｜K與D接近"

    if latest["K"] >= 80 and latest["D"] >= 80:
        status += "｜KD位於超買區"
    elif latest["K"] <= 20 and latest["D"] <= 20:
        status += "｜KD位於超賣區"
    elif latest["K"] >= 80:
        status += "｜K進入超買區"
    elif latest["K"] <= 20:
        status += "｜K進入超賣區"
    return status


def get_macd_status(latest, previous):
    dif = latest["DIF"]
    macd = latest["MACD"]
    osc = latest["OSC"]
    prev_osc = previous["OSC"]

    status = f"DIF {dif:.2f}｜MACD {macd:.2f}｜OSC {osc:+.2f}"
    if dif >= 0 and macd >= 0:
        status += "｜雙線零軸之上"
    elif dif < 0 and macd < 0:
        status += "｜雙線零軸之下"
    else:
        status += "｜雙線多空交界"

    if osc > 0 and osc > prev_osc:
        status += "｜紅柱放大，多方動能增強"
    elif osc > 0 and osc <= prev_osc:
        status += "｜紅柱縮小，多方動能減弱"
    elif osc < 0 and osc < prev_osc:
        status += "｜綠柱放大，空方動能增強"
    elif osc < 0 and osc >= prev_osc:
        status += "｜綠柱收斂，空方動能減弱"
    return status


def get_bollinger_status(latest):
    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]

    if close > upper:
        return "收盤突破上軌，趨勢偏強，但短線乖離擴大"
    if close < lower:
        return "收盤跌破下軌，技術結構偏弱，同時留意超跌反彈"
    if close >= middle:
        return "收盤位於中軌之上，短線價格結構偏多"
    return "收盤位於中軌之下，短線價格結構仍偏弱"


def get_technical_conclusion(latest):
    close = latest["Close"]
    middle = latest["MA20"]
    upper = latest["Upper_Band"]
    lower = latest["Lower_Band"]

    k_val = latest["K"]
    d_val = latest["D"]
    risk_threshold = latest["Risk_Threshold"]
    ma_label = latest.get("MA_Long_Label", "200日均線")

    breakout_regime = bool(latest.get("Breakout_Regime", False))
    gap_up = bool(latest.get("Gap_Up", False))
    gap_down = bool(latest.get("Gap_Down", False))

    if breakout_regime:
        if gap_up:
            return (
                "🔥 【結構性突破】出現爆量與顯著向上跳空！\n"
                "市場定價邏輯可能已轉變，傳統震盪指標易出現高檔鈍化。\n"
                "技術面宜轉為趨勢跟隨，並以近期跳空缺口或 ATR 動態防守點作為評估參考。"
            )
        if gap_down:
            return (
                "⚠️ 【結構性斷裂】出現爆量與顯著向下跳空！\n"
                "底部支撐結構可能失效，需留意超賣指標失真風險，\n"
                "應優先審視基本面變數，並嚴格執行風險價值控管機制。"
            )

    if close < risk_threshold:
        return (
            f"🚨 收盤低於{ma_label}的95%參考值，"
            "長期價格結構偏弱，宜優先關注風險控管。"
        )

    if close > upper and k_val > d_val and k_val >= 80:
        return "🟠 價格突破布林上軌且KD偏多，但K值已進入超買區，短線乖離及震盪風險上升。"
    if close < lower and k_val < d_val and k_val <= 20:
        return "🔵 價格跌破布林下軌且KD偏空，但K值已進入超賣區，反映短線可能存在超跌現象，需留意技術性反彈契機。"
    if close >= middle and k_val > d_val:
        return "🟢 收盤位於布林中軌之上，且K高於D，短線多方動能占優。"
    if close < lower and k_val < d_val:
        return "🔴 收盤低於布林下軌，且K低於D，技術結構偏弱。"
    if close < middle and k_val > d_val:
        return "🟡 KD動能改善，但收盤仍低於布林中軌，較接近弱勢反彈，尚未確認趨勢轉強。"
    if close >= middle and k_val < d_val:
        return "🟡 價格仍在布林中軌之上，但KD動能轉弱，留意短線整理。"
    return "⚪ 布林通道與KD訊號分歧，目前以中性整理看待。"


# ============================================================
# 單一股票報告產出
# ============================================================

def build_stock_report(ticker, stock_info, df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    latest_date = df.index[-1].strftime("%Y-%m-%d")
    daily_change = latest["Close"] - previous["Close"]
    daily_change_pct = (latest["Close"] / previous["Close"] - 1) * 100

    volume_text = "資料缺漏" if pd.isna(latest["Volume"]) else f"{int(latest['Volume']):,} 股"
    currency_label = stock_info["currency_label"]
    ma_label = latest.get("MA_Long_Label", "200日均線")

    breakout_text = "觸發" if bool(latest.get("Breakout_Regime", False)) else "未觸發"

    report = f"""【{stock_info['name']}｜{ticker}】
📅 交易日期：{latest_date}
💰 收盤：{latest['Close']:.2f} {currency_label}
📉 漲跌：{daily_change:+.2f} {currency_label}（{daily_change_pct:+.2f}%）
📦 成交量：{volume_text}

📊 布林通道 (20, 2, ddof=0)
上軌：{latest['Upper_Band']:.2f}
中軌：{latest['MA20']:.2f}
下軌：{latest['Lower_Band']:.2f}
帶寬：{latest['BB_Width_Pct']:.2f}%
通道位置：{latest['BB_Position_Pct']:.2f}%
訊號：{get_bollinger_status(latest)}

📈 KD指標 (9, 3, 3)
{get_kd_status(latest)}

📉 MACD指標 (12, 26, 9)
{get_macd_status(latest, previous)}

🛡️ 長期風險與波動率參考
{ma_label}：{latest['MA_Long']:.2f}
長期風險95%參考值：{latest['Risk_Threshold']:.2f}
14日真實波幅 (ATR)：{latest['ATR']:.2f}
結構斷裂警示：{breakout_text}

💡 技術觀察結論
{get_technical_conclusion(latest)}"""

    cycle_metrics = calculate_cycle_metrics(df, ticker)
    if cycle_metrics:
        report += f"\n\n{cycle_metrics}"

    return report


# ============================================================
# Gemini 潤飾與 Telegram 自動分段推播
# ============================================================

def polish_combined_report_with_gemini(base_report):
    if not GEMINI_API_KEY:
        logger.warning("未設定 GEMINI_API_KEY，直接使用 Python 原始報告")
        return base_report

    prompt = f"""
你是一位繁體中文量化報告編輯。請整理下方已由 Python 完成計算的技術分析報告。
嚴格規則：
1. 不得修改任何日期或數值。
2. 不得重新計算指標。
3. 不得增加目標價、停損價或保證指令。
4. 必須保留各標的交易日期與週期分析專欄。
5. 使用台灣繁體中文，僅改善文字流暢度及排版。

原始報告：
{base_report}
"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip() if response.text else base_report
    except Exception as error:
        logger.warning("Gemini 整理失敗：%s，改用 Python 原始報告", repr(error))
        return base_report


def split_tg_message(text, max_len=MAX_TG_MESSAGE_LENGTH):
    """
    智能切分過長訊息（Telegram 單則上限 4096 字元），
    優先依據分隔線或換行符切分，避免在文字中間截斷。
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    delimiter = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    if delimiter in text:
        parts = text.split(delimiter)
        current_chunk = ""
        for i, part in enumerate(parts):
            piece = part if i == 0 else delimiter + part
            if len(current_chunk) + len(piece) <= max_len:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if all(len(c) <= max_len for c in chunks):
            return chunks

    lines = text.split("\n")
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_len:
            current_chunk += line + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def send_tg_message(message):
    """
    發送 Telegram 訊息，自動分段發送超過 4096 字元之超長報告。
    """
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.warning("未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，跳過 Telegram 發送")
        return

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    chunks = split_tg_message(message)
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": TG_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Telegram 發送失敗（第 {idx}/{total_chunks} 則），HTTP {response.status_code}：{response.text[:500]}"
            )
        logger.info("Telegram 報告第 %d/%d 則已成功發送", idx, total_chunks)
        if idx < total_chunks:
            time.sleep(0.5)


# ============================================================
# 主程式入口（支援 CLI --market [tw/eu/us/all] 與 --force）
# ============================================================

MARKET_MAP = {
    "tw": "台灣",
    "台灣": "台灣",
    "eu": "歐洲",
    "歐洲": "歐洲",
    "us": "美國",
    "美國": "美國",
    "all": None,
    "全部": None,
}


def run_analysis(market="all", force=False):
    """
    執行指定市場分析：
    market: 'tw' (台股), 'eu' (歐股), 'us' (美股), 'all' (全球市場)
    force: 是否略過假日檢核
    """
    check_dependencies()

    market_key = str(market).lower().strip()
    market_filter = MARKET_MAP.get(market_key, None)

    # 週末與假日快速過濾（若未啟用 --force）
    if not force:
        now_utc = pd.Timestamp.now(tz="UTC")
        if market_filter == "台灣":
            now_local = now_utc.tz_convert("Asia/Taipei")
        elif market_filter == "歐洲":
            now_local = now_utc.tz_convert("Europe/Berlin")
        else:
            now_local = now_utc.tz_convert("America/New_York")

        weekday = now_local.weekday()  # 5=Sat, 6=Sun
        if weekday in [5, 6]:
            logger.info(
                "今天是週末（%s，星期%d），依常規跳過排程分析。若需手動測試請帶入 --force 參數。",
                now_local.strftime("%Y-%m-%d"),
                weekday + 1,
            )
            return

    stock_reports = []
    failed_stocks = []

    target_stocks = {
        k: v for k, v in STOCKS.items()
        if market_filter is None or v.get("market") == market_filter
    }

    if market_filter == "台灣":
        market_label = "台股"
    elif market_filter == "歐洲":
        market_label = "歐股"
    elif market_filter == "美國":
        market_label = "美股"
    else:
        market_label = "多市場"

    logger.info("開始執行 %s 量化技術分析，目標標的共 %d 檔", market_label, len(target_stocks))

    for ticker, stock_info in target_stocks.items():
        try:
            logger.info("處理標的：%s (%s)", stock_info["name"], ticker)
            df = download_stock_data(ticker)
            df = calculate_indicators(df, ticker)
            report = build_stock_report(ticker, stock_info, df)
            stock_reports.append(report)
        except Exception as error:
            logger.exception("%s 分析失敗", ticker)
            failed_stocks.append(f"{ticker}：{type(error).__name__}，{str(error)[:200]}")

    if not stock_reports:
        raise RuntimeError(f"{market_label} 所有標的均分析失敗")

    combined_report = (
        f"📋 每日{market_label}量化技術與週期摘要報告\n\n"
        + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(stock_reports)
    )

    if failed_stocks:
        combined_report += "\n\n⚠️ 部分標的分析失敗：\n" + "\n".join(failed_stocks)

    combined_report += "\n\n📌 註：台股、歐股與美股時區不同，各標的均採用最新已結算完成交易日。本摘要為公開數據之量化模型計算，不構成投資建議。"

    final_report = polish_combined_report_with_gemini(combined_report)
    print("\n" + final_report + "\n", flush=True)

    try:
        send_tg_message(final_report)
    except Exception as e:
        logger.warning("Telegram 發送失敗：%s", repr(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="每日股票量化技術分析與週期監控")
    parser.add_argument(
        "--market",
        choices=["tw", "eu", "us", "all", "台灣", "歐洲", "美國", "全部"],
        default="all",
        help="選擇要分析的市場 (tw: 台股, eu: 歐股, us: 美股, all: 全部)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="是否強制執行（略過週末/休市檢核）",
    )

    args = parser.parse_args()

    try:
        run_analysis(market=args.market, force=args.force)
    except Exception as error:
        logger.exception("量化技術分析執行失敗：%s", repr(error))
        sys.exit(1)

