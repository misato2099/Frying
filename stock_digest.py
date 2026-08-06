from __future__ import annotations
import argparse, json, logging, math, os, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import yfinance as yf

LOG = logging.getLogger("stock_digest")
REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def normalize_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{ticker}: 行情下載結果為空")
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance 可能回傳 (Price,Ticker) 或 (Ticker,Price)
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0, drop_level=True)
    df.columns = [str(c).strip().title() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: 缺少欄位 {missing}; 實際欄位={list(df.columns)}")
    df = df[REQUIRED].copy()
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df[["Open","High","Low","Close"]] > 0).all(axis=1)]
    bad = (df["High"] < df[["Open","Close","Low"]].max(axis=1)) | (df["Low"] > df[["Open","Close","High"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{ticker}: 發現 {int(bad.sum())} 筆 OHLC 邏輯錯誤")
    if len(df) < 30:
        raise ValueError(f"{ticker}: 有效日線僅 {len(df)} 筆，至少需 30 筆")
    return df

def fetch_prices(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    last = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker, period=period, interval="1d", auto_adjust=False,
                actions=False, repair=True, keepna=False, progress=False,
                threads=False, timeout=30, multi_level_index=False
            )
            return normalize_download(raw, ticker)
        except Exception as exc:
            last = exc
            LOG.warning("%s 第 %s 次下載失敗: %s", ticker, attempt, exc)
    raise RuntimeError(f"{ticker}: 下載重試後仍失敗: {last}")

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    bw = int(cfg["bollinger_window"]); mult = float(cfg["bollinger_std"]); ddof = int(cfg.get("bollinger_ddof", 0))
    out["BB_Mid"] = out["Close"].rolling(bw, min_periods=bw).mean()
    sigma = out["Close"].rolling(bw, min_periods=bw).std(ddof=ddof)
    out["BB_Upper"] = out["BB_Mid"] + mult * sigma
    out["BB_Lower"] = out["BB_Mid"] - mult * sigma
    width = out["BB_Upper"] - out["BB_Lower"]
    out["BB_Width_Pct"] = width / out["BB_Mid"] * 100
    out["BB_Position_Pct"] = (out["Close"] - out["BB_Lower"]) / width.replace(0, pd.NA) * 100

    kw = int(cfg["kd_window"]); alpha = float(cfg["kd_alpha"]); initial = float(cfg["kd_initial"])
    low_n = out["Low"].rolling(kw, min_periods=kw).min()
    high_n = out["High"].rolling(kw, min_periods=kw).max()
    span = high_n - low_n
    out["RSV"] = (out["Close"] - low_n) / span.replace(0, pd.NA) * 100
    k_vals=[]; d_vals=[]; prev_k=initial; prev_d=initial
    for rsv in out["RSV"]:
        if pd.isna(rsv):
            k, d = prev_k, prev_d
        else:
            k = (1-alpha)*prev_k + alpha*float(rsv)
            d = (1-alpha)*prev_d + alpha*k
        k_vals.append(k); d_vals.append(d); prev_k,prev_d=k,d
    out["K"] = k_vals; out["D"] = d_vals
    out["KD_Signal"] = "無"
    gold = (out["K"] > out["D"]) & (out["K"].shift(1) <= out["D"].shift(1))
    death = (out["K"] < out["D"]) & (out["K"].shift(1) >= out["D"].shift(1))
    out.loc[gold, "KD_Signal"] = "黃金交叉"
    out.loc[death, "KD_Signal"] = "死亡交叉"
    return out

def finite_row(row: pd.Series, ticker: str):
    for c in ["BB_Mid","BB_Upper","BB_Lower","RSV","K","D"]:
        if pd.isna(row[c]) or not math.isfinite(float(row[c])):
            raise ValueError(f"{ticker}: 最新列 {c} 無法計算")

def interpretation(row: pd.Series, prev: pd.Series, cfg: dict) -> str:
    parts=[]
    if row.Close > row.BB_Upper: parts.append("收盤突破布林上軌，趨勢偏強但乖離擴大")
    elif row.Close < row.BB_Lower: parts.append("收盤跌破布林下軌，走勢偏弱但可能進入超跌")
    elif row.Close >= row.BB_Mid: parts.append("收盤位於布林中軌之上，結構偏多")
    else: parts.append("收盤位於布林中軌之下，結構尚偏弱")
    if row.KD_Signal != "無": parts.append(row.KD_Signal)
    elif row.K > row.D: parts.append("K 高於 D，多方動能延續")
    else: parts.append("K 低於 D，空方動能較強")
    if row.K >= cfg["overbought"] and row.D >= cfg["overbought"]: parts.append("KD 位於超買區")
    elif row.K <= cfg["oversold"] and row.D <= cfg["oversold"]: parts.append("KD 位於超賣區")
    elif row.K >= cfg["overbought"]: parts.append("K 已進入超買區")
    elif row.K <= cfg["oversold"]: parts.append("K 已進入超賣區")
    return "；".join(parts) + "。"

def fmt_block(ticker: str, meta: dict, df: pd.DataFrame, cfg: dict) -> str:
    row=df.iloc[-1]; prev=df.iloc[-2]; finite_row(row,ticker)
    day=pd.Timestamp(df.index[-1]).strftime("%Y/%m/%d")
    change=(row.Close/prev.Close-1)*100
    return (f"【{meta['name']} {ticker}｜{day}】\n"
            f"收盤：{row.Close:.2f} {meta['currency']}（{change:+.2f}%）｜量：{int(row.Volume):,}\n"
            f"布林：上 {row.BB_Upper:.2f}｜中 {row.BB_Mid:.2f}｜下 {row.BB_Lower:.2f}\n"
            f"帶寬：{row.BB_Width_Pct:.2f}%｜通道位置：{row.BB_Position_Pct:.2f}%\n"
            f"KD：RSV {row.RSV:.2f}｜K {row.K:.2f}｜D {row.D:.2f}｜{row.KD_Signal}\n"
            f"解讀：{interpretation(row,prev,cfg)}")

def send_telegram(text: str):
    token=os.environ["TELEGRAM_BOT_TOKEN"]; chat=os.environ["TELEGRAM_CHAT_ID"]
    r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id":chat,"text":text,"disable_web_page_preview":True}, timeout=30)
    r.raise_for_status()

def send_line(text: str):
    token=os.environ["LINE_CHANNEL_ACCESS_TOKEN"]; target=os.environ["LINE_TO"]
    r=requests.post("https://api.line.me/v2/bot/message/push", headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"}, json={"to":target,"messages":[{"type":"text","text":text[:5000]}]}, timeout=30)
    r.raise_for_status()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.json"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--out",default="output")
    a=p.parse_args(); cfg=load_config(a.config); logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True)
    blocks=[]; summary=[]
    for ticker,meta in cfg["tickers"].items():
        df=add_indicators(fetch_prices(ticker,cfg["history_period"]),cfg)
        df.to_csv(outdir/f"{ticker.replace('.','_')}_daily.csv",encoding="utf-8-sig",index_label="Date")
        blocks.append(fmt_block(ticker,meta,df,cfg))
        row=df.iloc[-1]; summary.append({"ticker":ticker,"date":str(df.index[-1].date()),"close":row.Close,"bb_mid":row.BB_Mid,"bb_upper":row.BB_Upper,"bb_lower":row.BB_Lower,"rsv":row.RSV,"k":row.K,"d":row.D,"signal":row.KD_Signal})
    now=datetime.now(ZoneInfo(cfg["timezone"])).strftime("%Y/%m/%d %H:%M")
    text=f"每日技術摘要｜產生時間 {now}\n\n"+"\n\n".join(blocks)+"\n\n資料為公開行情整理，不構成投資建議。"
    (outdir/"digest.txt").write_text(text,encoding="utf-8")
    (outdir/"latest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(text)
    if a.dry_run: return
    sent=False
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"): send_telegram(text); sent=True
    if os.getenv("LINE_CHANNEL_ACCESS_TOKEN") and os.getenv("LINE_TO"): send_line(text); sent=True
    if not sent: raise RuntimeError("未設定通知 Secrets；可用 Telegram 或 LINE")
if __name__=="__main__":
    try: main()
    except Exception as exc:
        LOG.exception("執行失敗: %s",exc); sys.exit(1)
