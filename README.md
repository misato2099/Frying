# Daily Stock Technical Digest

An automated stock market technical analysis tool that calculates Bollinger Bands, KD, MACD, and ATR indicators using unadjusted OHLC data, with results pushed to Telegram. Monitors both Taiwan Stock Exchange (TWSE) and US markets with automatic trading day validation.

## Overview

This project analyzes specified stocks and generates daily technical summaries using:
- **20-day Bollinger Bands** with position percentage and bandwidth
- **9-day KD (Stochastic)** indicator with 1/3 weight recursive smoothing
- **MACD** with 12/26/9 periods for momentum analysis
- **14-day ATR** (Average True Range) for volatility assessment
- **Volume Spike & Breakout Detection** for structural price movements
- **200-day MA Risk Threshold** for long-term trend reference
- Comprehensive data validation and holiday/market closure checks

## Key Features

- **Dual-Market Support**: Monitors Taiwan (TWSE) and US (NYSE/NASDAQ) stocks simultaneously
- **Precision Data Handling**
  - Uses unadjusted OHLC data (`auto_adjust=False`) to avoid dividend-related adjustments
  - Per-ticker downloads to eliminate cross-market timezone conflicts
  - Population standard deviation (ddof=0) aligned with charting software
  - 9-day RSV with 1/3 weight recursive smoothing, initialized at 50.0
  - Comprehensive validation: duplicate dates, NaN values, negative prices, OHLC logic, and minimum data volume

- **Advanced Signal Detection**
  - Golden/Death Cross detection (KD indicator)
  - MACD trend direction and momentum oscillator
  - Bollinger Band position analysis (breakouts vs mean reversion)
  - Volume spike detection (2.5x 20-day average)
  - Gap detection with ATR normalization
  - Structural breakout regime identification

- **Market Intelligence**
  - Automatic TWSE (Taiwan) holiday lookup via official API
  - US stock market holiday calendar (NYSE/NASDAQ)
  - Typhoon/natural disaster closure detection for TWSE
  - Timezone-aware trading day validation
  - Market-specific execution filtering

- **Output Artifacts**
  - Individual stock calculation CSVs (60 most recent days)
  - Combined technical digest report
  - Telegram push notifications
  - Gemini AI-polished report formatting (fallback to Python if API unavailable)

## Setup Instructions

### 1. Configure Stock Universe

Edit `stock_cycle.py` to customize the stock list and indicators:

```python
# Lines 31-95: Stock Configuration
STOCKS = {
    "2409.TW": {
        "name": "友達",                    # Display name
        "short_name": "2409.TW",
        "currency": "TWD",
        "currency_label": "元",
        "market": "台灣",                 # Market category: "台灣" or "美國"
    },
    "FCX": {
        "name": "自由港麥克莫蘭",
        "short_name": "FCX",
        "currency": "USD",
        "currency_label": "美元",
        "market": "美國",
    },
    # Add more stocks as needed
}

# Indicator Configuration (Lines 97-116)
HISTORY_PERIOD = "2y"              # Historical lookback period
BB_WINDOW = 20                      # Bollinger Bands period
BB_STD_MULTIPLIER = 2.0            # Bollinger Bands std dev multiplier
BB_DDOF = 0                         # 0=population, 1=sample
MA_LONG_WINDOW = 200               # Long-term moving average
RISK_THRESHOLD_RATIO = 0.95        # MA200 risk reference (95% of MA200)
KD_WINDOW = 9                       # KD period
KD_ALPHA = 1/3                      # KD smoothing weight
MACD_FAST = 12                      # MACD fast EMA
MACD_SLOW = 26                      # MACD slow EMA
MACD_SIGNAL = 9                     # MACD signal line
```

### 2. Add Repository Secrets

In **Repository Settings > Secrets and variables > Actions**, configure:

#### Telegram (Required for Notifications)
- `TELEGRAM_BOT_TOKEN`: Your bot token from BotFather (@BotFather on Telegram)
- `TELEGRAM_CHAT_ID`: Target chat numeric ID (use @userinfobot to find yours)

#### Gemini AI (Optional - for Report Polishing)
- `GEMINI_API_KEY`: Google Generative AI API key (from https://ai.google.dev/)

**Note**: If Gemini API is unavailable or misconfigured, the tool gracefully falls back to the raw Python-generated report.

### 3. Configure GitHub Actions Workflow

The workflow typically runs Monday–Friday at 08:10 Asia/Taipei time. Adjust the cron schedule in `.github/workflows/*.yml`:

```yaml
schedule:
  - cron: '10 0 * * 1-5'  # UTC: 00:10 on Mon-Fri (= 08:10 Taipei time)
```

### 4. Test Locally

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run analysis with forced execution (skip trading day checks)
python stock_cycle.py --force

# Run for Taiwan market only
python stock_cycle.py --market tw

# Run for US market only
python stock_cycle.py --market us

# Run for both markets (default)
python stock_cycle.py --market all
```

## Technical Indicator Reference

| Indicator | Period | Description |
|-----------|--------|-------------|
| **Bollinger Bands** | 20 days | Price envelope; breakouts signal trend strength |
| **BB Position %** | Dynamic | 0% = lower band, 100% = upper band |
| **BB Width %** | Dynamic | Band width as % of middle band; squeezes precede moves |
| **KD (RSV-based)** | 9 days | Stochastic oscillator; >80 overbought, <20 oversold |
| **MACD** | 12/26/9 | Exponential moving average momentum; OSC shows bars |
| **ATR** | 14 days | True Range volatility; used for breakout confirmation |
| **MA200** | 200 days | Long-term trend; risk threshold = 95% of MA200 |
| **Volume Spike** | 20-day MA | Triggered when volume > 2.5x average |
| **Breakout Regime** | ATR-normalized | True when volume spike + significant gap occur simultaneously |

## Interpretation Guide

### Bollinger Band Signals
- **Price > Upper Band**: Trend strength or overbought condition
- **Price < Lower Band**: Support test or oversold condition
- **Narrow Bands**: Low volatility; consolidation phase
- **Wide Bands**: High volatility; strong directional move

### KD Status
- **K > D + K > 50**: Bullish momentum
- **K < D + K < 50**: Bearish momentum
- **K > 80 or D > 80**: Overbought region
- **K < 20 or D < 20**: Oversold region
- **Golden Cross**: K crosses above D (bullish reversal signal)
- **Death Cross**: K crosses below D (bearish reversal signal)

### MACD Interpretation
- **DIF > MACD > 0**: Strong uptrend
- **DIF < MACD < 0**: Strong downtrend
- **OSC expanding**: Momentum accelerating
- **OSC contracting**: Momentum weakening
- **Golden Cross**: DIF crosses above MACD (buy signal)
- **Death Cross**: DIF crosses below MACD (sell signal)

### Structural Breakout
- **Breakout Regime = True**: Volume spike + gap detected; defines new market regime
- **Gap Up + Breakout**: Potential continuation or momentum spike
- **Gap Down + Breakout**: Potential breakdown or panic selling

## Data Source & Limitations

- Uses **Yahoo Finance** via `yfinance` library (NOT official exchange data)
- Suitable for personal research and educational use only
- Subject to Yahoo Finance terms of service and data availability
- For production trading, replace `download_stock_data()` with your authorized exchange API

## Command-Line Options

```bash
python stock_cycle.py --market {tw|us|all} [--force]
```

| Option | Description |
|--------|-------------|
| `--market tw` | Taiwan stocks only |
| `--market us` | US stocks only |
| `--market all` | Both markets (default) |
| `--force` | Skip trading day checks; run regardless of holidays |

## Dependencies

- Python 3.8+
- `yfinance`: Stock data download
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scipy`: Scientific computing (required for yfinance data repair)
- `requests`: HTTP library (Telegram API, TWSE API)
- `beautifulsoup4`: HTML parsing (typhoon closure check)
- `pytz`: Timezone handling
- `google-genai`: Gemini AI (optional, for report polishing)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Telegram messages not sending** | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set correctly |
| **Data validation errors** | Verify ticker symbols; check Yahoo Finance availability for that ticker |
| **"SciPy not installed"** | Run `pip install scipy` |
| **Gemini report fails** | API falls back to Python report; check `GEMINI_API_KEY` if polishing is needed |
| **Timezone mismatches** | Ensure `Asia/Taipei` or target timezone is correctly configured in `pytz` |

## License & Disclaimer

This project is for personal and research use only. It does not constitute investment advice. Always conduct your own due diligence and consult a licensed financial advisor before making investment decisions. The authors assume no liability for losses incurred through use of this analysis tool.
