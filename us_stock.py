import os
import time
import yfinance as yf
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
import requests
import numpy as np

# 從 GitHub Secrets 安全保險箱自動讀取金鑰
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = "您的_TELEGRAM_CHAT_ID" # 這裡請保持填入您的純數字 Chat ID

genai.configure(api_key=GEMINI_API_KEY)

def ask_gemini_fcx_report():
    print("正在獲取美股 FCX (Freeport-McMoRan) 歷史數據並進行精準量化計算...")
    # 抓取美股 FCX 數據
    df = yf.Ticker("FCX").history(period="2y")
    if df.empty:
        print("⚠️ 美股數據抓取失敗。")
        return
        
    # 計算布林通道 (20, 2)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (2 * df['BB_STD'])
    df['Lower_Band'] = df['MA20'] - (2 * df['BB_STD'])
    
    # 計算長期年線與風險門檻
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Risk_Threshold'] = df['MA200'] * 0.95
    
    # 計算標準 KD 指標 (9, 3, 3)
    df['Min_Low_9'] = df['Low'].rolling(window=9).min()
    df['Max_High_9'] = df['High'].rolling(window=9).max()
    df['RSV'] = ((df['Close'] - df['Min_Low_9']) / (df['Max_High_9'] - df['Min_Low_9']).replace(0, np.nan)) * 100
    df['RSV'] = df['RSV'].fillna(50)

    k_list, d_list = [], []
    current_k, current_d = 50.0, 50.0
    for rsv in df['RSV']:
        if np.isnan(rsv):
            k_list.append(np.nan)
            d_list.append(np.nan)
        else:
            current_k = (2/3) * current_k + (1/3) * rsv
            current_d = (2/3) * current_d + (1/3) * current_k
            k_list.append(current_k)
            d_list.append(current_d)
    df['K'] = k_list
    df['D'] = d_list

    df.dropna(subset=['MA200', 'K'], inplace=True)
    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    
    kd_status = f"K={latest['K']:.1f} / D={latest['D']:.1f}"
    if latest['K'] > latest['D']:
        kd_status += " (K偏多黃金交叉)"
    else:
        kd_status += " (K偏空死亡交叉)"
    if latest['K'] < 20: kd_status += " [低檔超賣]"
    if latest['K'] > 80: kd_status += " [高檔超買]"

    prompt = f"""
    你是一位言簡意賅的美股量化交易助理。請依據以下已經由 Python 計算完畢的 FreePort-McMoRan（FCX）數據，**嚴格依照下方格式**輸出簡報，禁止任何延伸評論：
    
    📋 美股 FCX (自由港麥克莫蘭) 開市前診斷
    📅 數據日期：{latest_date} (美股前一交易日收盤)
    📊 布林通道：{latest['Close']:.2f}美元 (中軌:{latest['MA20']:.2f}, 下軌:{latest['Lower_Band']:.2f}, 上軌:{latest['Upper_Band']:.2f})
    📈 KD指標：{kd_status}
    💡 開市決策：[請綜合上述指標，從「買進」、「賣出」或「觀望」選擇一項，並給出一句話理由。]
    
    **關鍵安全閥限制**：
    昨日最新收盤價為 {latest['Close']:.2f} 美元，系統風險臨界值為 {latest['Risk_Threshold']:.2f} 美元。
    如果收盤價低於風險臨界值，不論短線指標如何，今日開市決策請強制顯示為：「🚨 注意系統性風險（暫停交易）」。
    """

    model = genai.GenerativeModel('gemini-2.5-flash')
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"正在請求美股極簡診斷 (第 {attempt + 1} 次嘗試)...")
            response = model.generate_content(prompt)
            send_tg_message(response.text)
            return
        except google_exceptions.ServiceUnavailable:
            print(f"⚠️ Google 伺服器忙碌 (503)。")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                send_tg_message("⚠️ 美股 FCX 量化分析失敗：Google 伺服器連線超時。")
        except Exception as e:
            print(f"❌ 發生未預期錯誤: {e}")
            break

def send_tg_message(message):
    if not TG_TOKEN or "您的" in TG_CHAT_ID:
        print("提示：未偵測到正確的 Telegram 設定。")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message})
    print("美股通知已發送。")

if __name__ == "__main__":
    ask_gemini_perfect_report()
