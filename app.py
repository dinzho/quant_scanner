import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 頁面配置 (針對手機優化)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="量化策略掃描器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 動態量化策略掃描器")
st.caption("多週期道氏 + FIB 黃金口袋區 (手機響應版)")

# ------------------------------------------------------------------------------
# 1. 核心數據模組
# ------------------------------------------------------------------------------
@st.cache_data(ttl=60) # 快取 60 秒，避免頻繁請求被 yfinance 封鎖
def get_true_latest_price(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
        state = info.get("marketState", "").upper()
        if state == "PRE" and info.get("preMarketPrice"):
            return float(info["preMarketPrice"]), "盤前"
        elif state == "POST" and info.get("postMarketPrice"):
            return float(info["postMarketPrice"]), "盤後"
        elif info.get("regularMarketPrice"):
            return float(info["regularMarketPrice"]), "常規"
    except Exception:
        pass

    try:
        df_1m = ticker.history(period="1d", interval="1m", prepost=True)
        if not df_1m.empty and not df_1m['Close'].dropna().empty:
            return float(df_1m['Close'].dropna().iloc[-1]), "1分K"
    except Exception:
        pass

    return None, "失敗"

def analyze_stock(ticker):
    curr_price, source = get_true_latest_price(ticker)
    if curr_price is None:
        return None

    ticker_obj = yf.Ticker(ticker)
    df_daily = ticker_obj.history(period="1y", interval="1d")
    if len(df_daily) < 60:
        return None

    daily_recent = df_daily.tail(60)
    daily_high = max(daily_recent['High'].max(), curr_price)
    daily_low = min(daily_recent['Low'].min(), curr_price)

    fib_500 = daily_high - 0.500 * (daily_high - daily_low)
    fib_618 = daily_high - 0.618 * (daily_high - daily_low)
    fib_786 = daily_high - 0.786 * (daily_high - daily_low)

    in_fib_zone = (curr_price >= fib_786 * 0.99) and (curr_price <= fib_500 * 1.01)

    # 小時線訊號
    try:
        df_hourly = ticker_obj.history(period="1mo", interval="1h", prepost=True)
        df_hourly['EMA20'] = df_hourly['Close'].ewm(span=20, adjust=False).mean()
        df_hourly['Vol_MA'] = df_hourly['Volume'].rolling(20).mean()
        h_breakout = (df_hourly['Close'].iloc[-2] <= df_hourly['EMA20'].iloc[-2]) and (curr_price > df_hourly['EMA20'].iloc[-1])
        h_vol_spike = max(df_hourly['Volume'].iloc[-1], df_hourly['Volume'].iloc[-2]) > (df_hourly['Vol_MA'].iloc[-1] * 1.2)
        hourly_triggered = h_breakout and h_vol_spike
        hourly_stop_loss = round(min(df_hourly['Low'].tail(15).min(), fib_786) * 0.99, 2)
    except Exception:
        hourly_triggered = False
        hourly_stop_loss = round(fib_786 * 0.98, 2)

    # 策略判斷
    strategy = "👀 觀察中 (等待點位)"
    position = "0%"
    
    if in_fib_zone and hourly_triggered:
        strategy = "🚀 大主升浪起場點"
        position = "70% - 100%"
    elif in_fib_zone and (curr_price >= fib_618 * 0.99):
        strategy = "📈 中線波段建倉"
        position = "30% - 50%"
    elif hourly_triggered:
        strategy = "⚡ 短線衝刺/反彈"
        position = "15% - 25%"

    dist_618 = round(((curr_price - fib_618) / fib_618) * 100, 2)

    return {
        "代碼": ticker,
        "現價": round(curr_price, 2),
        "來源": source,
        "建議策略": strategy,
        "建議倉位": position,
        "Fib 0.618": round(fib_618, 2),
        "距0.618(%)": f"{dist_618}%",
        "止損價": hourly_stop_loss,
        "_d_h": daily_high,
        "_d_l": daily_low,
        "_f618": fib_618,
        "_f786": fib_786
    }

# ------------------------------------------------------------------------------
# 2. 介面互動
# ------------------------------------------------------------------------------
input_tickers = st.text_input("輸入股票代碼 (多個請用空格隔開):", value="BA NVDA TSLA")

if st.button("🔍 開始掃描分析", use_container_width=True):
    tickers = [t.strip().upper() for t in input_tickers.split() if t.strip()]
    
    if not tickers:
        st.warning("請輸入有效的股票代碼！")
    else:
        results = []
        with st.spinner("正即時連線抓取盤口數據..."):
            for t in tickers:
                res = analyze_stock(t)
                if res:
                    results.append(res)
        
        if results:
            st.subheader("📊 階段 1：實時盤口總覽")
            df_display = pd.DataFrame(results).drop(columns=["_d_h", "_d_l", "_f618", "_f786"])
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🧪 階段 2：情境壓力測試")

            for res in results:
                with st.expander(f"📌 {res['代碼']} 壓力測試明細", expanded=True):
                    f618 = res["_f618"]
                    f786 = res["_f786"]
                    d_h = res["_d_h"]

                    scenarios = [
                        {
                            "情境": "1. 回踩黃金口袋區 (理想低吸)",
                            "模擬價": round(f618 * 1.002, 2),
                            "建議策略": "🚀 長/中線重倉",
                            "建議倉位": "70% - 100%",
                            "止損價": round(f786 * 0.98, 2)
                        },
                        {
                            "情境": "2. 跌破 FIB 0.786 (破位避險)",
                            "模擬價": round(f786 * 0.97, 2),
                            "建議策略": "👀 離場觀望",
                            "建議倉位": "0%",
                            "止損價": "N/A"
                        },
                        {
                            "情境": "3. 突破前高 (右側追漲)",
                            "模擬價": round(d_h * 1.01, 2),
                            "建議策略": "⚡ 短線輕倉快進快出",
                            "建議倉位": "15% - 25%",
                            "止損價": round(d_h * 0.97, 2)
                        }
                    ]
                    st.table(pd.DataFrame(scenarios))
        else:
            st.error("無法取得相關股票數據，請檢查輸入代碼。")