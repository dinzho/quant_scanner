import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. 頁面配置 (針對手機極致優化)
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
# 2. 高效快取與數據抓取模組
# ------------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_data(ticker_symbol):
    """一站式抓取現價與歷史 K 線，並處理快取避免被封 IP"""
    ticker = yf.Ticker(ticker_symbol)
    curr_price = None
    source = "失敗"

    # A. 取得最新實時/盤前盤後價格
    try:
        info = ticker.info or {}
        state = info.get("marketState", "").upper()
        if state == "PRE" and info.get("preMarketPrice"):
            curr_price, source = float(info["preMarketPrice"]), "盤前"
        elif state == "POST" and info.get("postMarketPrice"):
            curr_price, source = float(info["postMarketPrice"]), "盤後"
        elif info.get("regularMarketPrice"):
            curr_price, source = float(info["regularMarketPrice"]), "常規"
    except Exception:
        pass

    if curr_price is None:
        try:
            df_1m = ticker.history(period="1d", interval="1m", prepost=True)
            if not df_1m.empty and not df_1m['Close'].dropna().empty:
                curr_price, source = float(df_1m['Close'].dropna().iloc[-1]), "1分K"
        except Exception:
            pass

    # B. 抓取日線與小時線
    try:
        df_daily = ticker.history(period="1y", interval="1d")
        df_hourly = ticker.history(period="1mo", interval="1h", prepost=True)
    except Exception:
        return None

    if curr_price is None or len(df_daily) < 60:
        return None

    return {
        "price": curr_price,
        "source": source,
        "df_daily": df_daily,
        "df_hourly": df_hourly
    }

def analyze_stock(ticker):
    data = fetch_stock_data(ticker)
    if not data:
        return None

    curr_price = data["price"]
    source = data["source"]
    df_daily = data["df_daily"]
    df_hourly = data["df_hourly"]

    # 日線 FIB 計算
    daily_recent = df_daily.tail(60)
    daily_high = max(daily_recent['High'].max(), curr_price)
    daily_low = min(daily_recent['Low'].min(), curr_price)

    fib_500 = daily_high - 0.500 * (daily_high - daily_low)
    fib_618 = daily_high - 0.618 * (daily_high - daily_low)
    fib_786 = daily_high - 0.786 * (daily_high - daily_low)

    in_fib_zone = (curr_price >= fib_786 * 0.99) and (curr_price <= fib_500 * 1.01)

    # 小時線訊號 (加入防禦性安全檢查)
    hourly_triggered = False
    hourly_stop_loss = round(fib_786 * 0.98, 2)

    if len(df_hourly) >= 20:
        try:
            df_hourly = df_hourly.copy()
            df_hourly['EMA20'] = df_hourly['Close'].ewm(span=20, adjust=False).mean()
            df_hourly['Vol_MA'] = df_hourly['Volume'].rolling(20).mean()
            
            # 確保最後兩筆數據非空
            c_prev = df_hourly['Close'].iloc[-2]
            ema_prev = df_hourly['EMA20'].iloc[-2]
            ema_curr = df_hourly['EMA20'].iloc[-1]
            vol_curr = df_hourly['Volume'].iloc[-1]
            vol_prev = df_hourly['Volume'].iloc[-2]
            vol_ma = df_hourly['Vol_MA'].iloc[-1]

            h_breakout = (c_prev <= ema_prev) and (curr_price > ema_curr)
            h_vol_spike = max(vol_curr, vol_prev) > (vol_ma * 1.2)
            
            hourly_triggered = bool(h_breakout and h_vol_spike)
            hourly_stop_loss = round(min(df_hourly['Low'].tail(15).min(), fib_786) * 0.99, 2)
        except Exception:
            pass

    # 策略判斷
    if in_fib_zone and hourly_triggered:
        strategy = "🚀 大主升浪起場點"
        position = "70% - 100%"
        status_color = "green"
    elif in_fib_zone and (curr_price >= fib_618 * 0.99):
        strategy = "📈 中線波段建倉"
        position = "30% - 50%"
        status_color = "blue"
    elif hourly_triggered:
        strategy = "⚡ 短線衝刺/反彈"
        position = "15% - 25%"
        status_color = "orange"
    else:
        strategy = "👀 觀察中 (等待點位)"
        position = "0%"
        status_color = "gray"

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
        "color": status_color,
        "_d_h": daily_high,
        "_d_l": daily_low,
        "_f618": fib_618,
        "_f786": fib_786
    }

# ------------------------------------------------------------------------------
# 3. 介面互動與佈局 (針對手機卡片化)
# ------------------------------------------------------------------------------
# 容錯處理：自動將逗號、頓號替換為空格
raw_input = st.text_input("輸入股票代碼 (支援空格/逗號分隔):", value="BA, NVDA, TSLA")

if st.button("🔍 開始掃描分析", use_container_width=True):
    # 使用正規化拆分代碼
    tickers = [t.upper() for t in re.split(r'[\s,]+', raw_input) if t.strip()]
    
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
            st.subheader("📊 實時盤口總覽")
            
            # 1. 簡潔表格 (適合手機快速覽閱)
            df_display = pd.DataFrame(results)[
                ["代碼", "現價", "來源", "建議策略", "建議倉位", "Fib 0.618", "距0.618(%)", "止損價"]
            ]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("📱 手機卡片與壓力測試")

            # 2. 卡片化排版 (手機檢視體驗最佳)
            for res in results:
                with st.expander(f"📌 **{res['代碼']}** - {res['建議策略']} (現價: ${res['現價']})", expanded=True):
                    # 使用 3 欄指標卡片展示關鍵數據
                    m1, m2, m3 = st.columns(3)
                    m1.metric("現價", f"${res['現價']}", delta=res['來源'])
                    m2.metric("Fib 0.618", f"${res['Fib 0.618']}")
                    m3.metric("建議止損", f"${res['止損價']}")

                    st.caption(f"**建議倉位**：{res['建議倉位']}｜**距離 0.618**：{res['距0.618(%)']}")

                    # 壓力測試明細
                    f618, f786, d_h = res["_f618"], res["_f786"], res["_d_h"]
                    scenarios = [
                        {
                            "情境": "1. 低吸黃金區",
                            "模擬價": f"${round(f618 * 1.002, 2)}",
                            "策略": "🚀 長/中線重倉",
                            "倉位": "70%-100%",
                            "止損": f"${round(f786 * 0.98, 2)}"
                        },
                        {
                            "情境": "2. 跌破 FIB 0.786",
                            "模擬價": f"${round(f786 * 0.97, 2)}",
                            "策略": "👀 離場觀望",
                            "倉位": "0%",
                            "止損": "N/A"
                        },
                        {
                            "情境": "3. 突破前高",
                            "模擬價": f"${round(d_h * 1.01, 2)}",
                            "策略": "⚡ 短線追漲",
                            "倉位": "15%-25%",
                            "止損": f"${round(d_h * 0.97, 2)}"
                        }
                    ]
                    st.table(pd.DataFrame(scenarios))
        else:
            st.error("無法取得相關股票數據，請檢查輸入代碼或網路連線。")
