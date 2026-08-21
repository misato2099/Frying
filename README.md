# 每日技術摘要

此專案使用未調整 OHLC 計算 20 日布林通道與 9 日 KD，並透過 Telegram 或 LINE Messaging API 推播。

## 精準度設計

- 明確設定 `auto_adjust=False`，避免除息導致 OHLC 被自動回溯調整。
- `repair=True` 嘗試修復資料源的 100 倍幣別錯置。
- 每檔股票分開下載，避免跨市場時區與 MultiIndex 混淆。
- 布林標準差預設 `ddof=0`，與多數看盤軟體的母體標準差一致。
- KD 採 9 日 RSV，K/D 皆以 1/3 權重遞迴平滑，且用 1 年資料暖機。
- 檢查重複日期、NaN、負價格、OHLC 邏輯與最少資料量。
- 每次保留完整 CSV、摘要 TXT 與最新 JSON 作為 GitHub Actions artifact。

## 安裝方式

1. 將壓縮檔內容覆蓋或合併到既有 repository 根目錄。
2. 在 Repository Settings > Secrets and variables > Actions 建立以下其中一組 Secrets。

### Telegram

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### LINE Messaging API

- `LINE_CHANNEL_ACCESS_TOKEN`
- `LINE_TO`，可為 userId、groupId 或 roomId

若兩組都設定，程式會同時推播到兩個管道。

3. 到 Actions 頁面手動執行 `Daily stock technical digest` 測試。
4. 排程預設為台北時間週一至週五 08:10。GitHub cron 使用 UTC，且排程可能有平台延遲。

## 本機驗證

```bash
python -m pip install -r requirements.txt
python stock_digest.py --dry-run
python -m pytest -q
```

## 可調參數

編輯 `config.json`：布林期間、標準差倍數、`ddof`、KD 期間、平滑權重、超買超賣門檻與時區。

## 資料限制

Yahoo Finance 非交易所官方資料源，且 yfinance 僅適合依其條款允許的研究或個人用途。若你已有付費或交易所授權 API，可只替換 `fetch_prices()`，其餘計算與通知層不需改動。
