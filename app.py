import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
import requests
import time
from dateutil import parser as date_parser

# ------------------------------------------------------------------------------
# 配置與常量定義
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
MAX_TICKERS = 50
REQUEST_TIMEOUT = 30
CACHE_TTL_HISTORY = 600
CACHE_TTL_PRICE = 60
CACHE_TTL_NEWS = 300
MIN_DATA_POINTS = 60
VOL_SPIKE_THRESHOLD = 1.5
EMA_PERIOD = 20

FIB_LEVELS = {'0': 0.0, '382': 0.382, '500': 0.500, '618': 0.618, '786': 0.786, '1000': 1.0}
NEWS_CACHE = {}

st.set_page_config(page_title="多週期量化策略掃描器", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 多週期波浪形態量化策略掃描器")
st.caption("Finnhub (歷史) + Yahoo (實時) | 自動降級備援機制")

# ------------------------------------------------------------------------------
# 側邊欄：API Key 配置與驗證
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ API 配置")
    
    # 自動去除首尾空格
    finnhub_key_input = st.text_input(
        "Finnhub API Key",
        type="password",
        help="請確保已驗證電子郵件。若報 403，請檢查信箱或重新生成 Key。",
        placeholder="cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    
    if finnhub_key_input:
        finnhub_key = finnhub_key_input.strip() # 關鍵修復：去除空格
        st.session_state['FINNHUB_KEY'] = finnhub_key
        
        # 測試連線按鈕
        if st.button("🧪 測試連線", use_container_width=True):
            with st.spinner("測試中..."):
                test_url = "https://finnhub.io/api/v1/stock/candle"
                test_params = {'symbol': 'AAPL', 'resolution': 'D', 'from': int(time.time())-86400, 'to': int(time.time()), 'token': finnhub_key}
                try:
                    resp = requests.get(test_url, params=test_params, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('s') == 'ok':
                            st.success("✅ 連線成功！API Key 有效。")
                        else:
                            st.error(f"❌ API 返回異常：{data.get('s')}")
                    elif resp.status_code == 401:
                        st.error("❌ 401 錯誤：Key 無效或已註銷。")
                    elif resp.status_code == 403:
                        st.error("❌ 403 錯誤：常見原因\n1. 未驗證電子郵件\n2. Key 複製錯誤\n3. 賬戶被凍結")
                        st.info("💡 解決方法：請登入 Finnhub 檢查郵箱驗證狀態，或在 Dashboard 重新生成 Key。")
                    elif resp.status_code == 429:
                        st.warning("⚠️ 429 錯誤：請求過於頻繁，請等待 1 分鐘後重試。")
                    else:
                        st.error(f"❌ 未知錯誤：{resp.status_code}")
                except Exception as e:
                    st.error(f"網路錯誤：{str(e)}")
        
        st.success("✅ API Key 已載入")
    else:
        st.warning("⚠️ 請輸入 Finnhub API Key")
        if 'FINNHUB_KEY' in st.session_state:
            del st.session_state['FINNHUB_KEY']

    st.markdown("---")
    st.info("💡 **架構說明**:\n- **優先**: Finnhub (日/周/月線)\n- **備選**: Yahoo Finance (當 Finnhub 失敗時)\n- **實時**: Yahoo Finance")

# ------------------------------------------------------------------------------
# 輸入驗證
# ------------------------------------------------------------------------------
def validate_ticker(ticker: str) -> bool:
    if not ticker or not isinstance(ticker, str): return False
    return bool(re.match(r'^[A-Z0-9.\-]{1,10}$', ticker.strip().upper()))

def parse_tickers(raw_input: str) -> List[str]:
    if not raw_input: return []
    raw_tickers = re.split(r'[,\s,]+', raw_input)
    valid = [t.strip().upper() for t in raw_tickers if validate_ticker(t)]
    if len(valid) > MAX_TICKERS:
        st.warning(f"已截斷至 {MAX_TICKERS} 個代碼")
        valid = valid[:MAX_TICKERS]
    return list(dict.fromkeys(valid)) # 去重

# ------------------------------------------------------------------------------
# 數據抓取模組 (混合架構 + 降級機制)
# ------------------------------------------------------------------------------
def fetch_finnhub_history(ticker: str, resolution: str, days: int) -> Optional[pd.DataFrame]:
    api_key = st.session_state.get('FINNHUB_KEY')
    if not api_key: return None
    
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 60 * 60)
    
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {'symbol': ticker, 'resolution': resolution, 'from': start_time, 'to': end_time, 'token': api_key}
    
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('s') == 'ok':
                    df = pd.DataFrame({'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']})
                    df.index = pd.to_datetime(data['t'], unit='s')
                    return df
                else:
                    return None # 數據為空
            elif resp.status_code in [401, 403]:
                logger.error(f"{ticker}: API Key 無效 (401/403)")
                return None # Key 錯誤，不再重試
            elif resp.status_code == 429:
                wait = (attempt + 1) * 2
                time.sleep(wait)
            else:
                return None
        except Exception as e:
            logger.error(f"{ticker}: 請求異常 - {str(e)}")
            if attempt < 2: time.sleep(2)
    return None

def fetch_yf_backup_history(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Yahoo Finance 備用方案"""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, timeout=15)
        if df is not None and not df.empty:
            # 重整欄位名稱以匹配 Finnhub 格式
            df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
            return df
    except Exception as e:
        logger.warning(f"{ticker}: YF 備用抓取失敗 - {str(e)}")
    return None

@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_multi_period_data(ticker_symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    if 'FINNHUB_KEY' not in st.session_state:
        # 若無 Key，直接嘗試全用 YF
        pass
    
    has_key = 'FINNHUB_KEY' in st.session_state
    
    # 1. 日線 (優先 Finnhub 365 天)
    df_daily = fetch_finnhub_history(ticker_symbol, 'D', 365) if has_key else None
    if df_daily is None:
        logger.info(f"{ticker_symbol}: Finnhub 日線失敗，切換至 Yahoo Finance")
        df_daily = fetch_yf_backup_history(ticker_symbol, "1y", "1d")
    
    if df_daily is None or len(df_daily) < MIN_DATA_POINTS:
        return None
    
    # 2. 周線 (優先 Finnhub 2 年)
    df_weekly = fetch_finnhub_history(ticker_symbol, 'W', 730) if has_key else None
    if df_weekly is None or df_weekly.empty:
        df_weekly = fetch_yf_backup_history(ticker_symbol, "2y", "1wk")
    
    # 3. 月線 (優先 Finnhub 5 年)
    df_monthly = fetch_finnhub_history(ticker_symbol, 'M', 1825) if has_key else None
    if df_monthly is None or df_monthly.empty:
        df_monthly = fetch_yf_backup_history(ticker_symbol, "5y", "3mo")
    
    # 4. 小時線 (優先 Finnhub 90 天)
    df_hourly = fetch_finnhub_history(ticker_symbol, '60', 90) if has_key else None
    if df_hourly is None or df_hourly.empty:
        # YF 小時線只能抓最近 30 天
        df_hourly = fetch_yf_backup_history(ticker_symbol, "1mo", "1h")

    return {
        "monthly": df_monthly if df_monthly is not None else pd.DataFrame(),
        "weekly": df_weekly if df_weekly is not None else pd.DataFrame(),
        "daily": df_daily,
        "hourly": df_hourly if df_hourly is not None else pd.DataFrame()
    }

@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_yf_price(ticker_symbol: str) -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
        curr_price = None
        prev_close = info.get("previousClose")
        source = "失敗"
        
        state = info.get("marketState", "").upper()
        if state == "PRE" and info.get("preMarketPrice"):
            curr_price = float(info["preMarketPrice"]); source = "盤前"
        elif state == "POST" and info.get("postMarketPrice"):
            curr_price = float(info["postMarketPrice"]); source = "盤後"
        elif info.get("regularMarketPrice"):
            curr_price = float(info["regularMarketPrice"]); source = "常規"
        
        if curr_price is None:
            df_1m = ticker.history(period="1d", interval="1m", prepost=True, timeout=10)
            if df_1m is not None and not df_1m.empty:
                curr_price = float(df_1m['Close'].iloc[-1])
                source = "1 分 K"
                if prev_close is None:
                    df_1d = ticker.history(period="5d", interval="1d", timeout=10)
                    if df_1d is not None and len(df_1d) >= 2: prev_close = df_1d['Close'].iloc[-2]

        if curr_price is None: return None
        
        change_amt = round(curr_price - prev_close, 2) if prev_close else 0
        change_pct = round((change_amt / prev_close) * 100, 2) if prev_close else 0
        
        return {"price": curr_price, "prev_close": prev_close, "change_amount": change_amt, "change_percent": change_pct, "source": source}
    except Exception as e:
        logger.error(f"{ticker_symbol}: 價格抓取失敗 - {str(e)}")
        return None

# ------------------------------------------------------------------------------
# 技術分析模組 (保持不變)
# ------------------------------------------------------------------------------
def identify_trend_direction(df: pd.DataFrame, period_name: str) -> str:
    if df is None or len(df) < 20: return "數據不足"
    try:
        h, l = df['High'].values, df['Low'].values
        rh, rl = h[-10:], l[-10:]
        hh = sum(1 for i in range(1, len(rh)) if rh[i] > rh[i-1])
        hl = sum(1 for i in range(1, len(rl)) if rl[i] > rl[i-1])
        lh = sum(1 for i in range(1, len(rh)) if rh[i] < rh[i-1])
        ll = sum(1 for i in range(1, len(rl)) if rl[i] < rl[i-1])
        if hh >= 6 and hl >= 6: return "上漲"
        if lh >= 6 and ll >= 6: return "下跌"
        return "盤整"
    except: return "無法判斷"

def identify_wave_stage(df: pd.DataFrame, trend: str, period_name: str) -> str:
    if df is None or len(df) < 20: return "數據不足"
    try:
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
        cp = df['Close'].iloc[-1]
        if trend == "上漲":
            if cp > ma20 > ma50: return "推動浪 (主升)"
            if ma50 < cp < ma20: return "回調浪 (次級)"
            return "盤整延伸"
        if trend == "下跌":
            if cp < ma20 < ma50: return "推動浪 (主跌)"
            if ma50 > cp > ma20: return "回調浪 (反彈)"
            return "盤整延伸"
        return "橫向盤整"
    except: return "無法判斷"

def calculate_fib_zones(df_daily: pd.DataFrame, curr_price: float) -> Dict[str, float]:
    if df_daily is None or len(df_daily) < 60: return {}
    recent = df_daily.tail(60)
    high = max(recent['High'].max(), curr_price)
    low = min(recent['Low'].min(), curr_price)
    rng = high - low
    zones = {f'fib_{k}': high - v * rng for k, v in FIB_LEVELS.items()}
    zones.update({'daily_high': high, 'daily_low': low})
    return zones

def check_hourly_signal(df_hourly: pd.DataFrame, curr_price: float, fib_786: float) -> Tuple[bool, float]:
    if df_hourly is None or len(df_hourly) < 20: return False, round(fib_786 * 0.98, 2)
    try:
        df_h = df_hourly.copy()
        df_h['EMA20'] = df_h['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        df_h['Vol_MA'] = df_h['Volume'].rolling(20).mean()
        if len(df_h) < 2: return False, round(fib_786 * 0.98, 2)
        
        c_prev, ema_prev = df_h['Close'].iloc[-2], df_h['EMA20'].iloc[-2]
        ema_curr, vol_curr = df_h['EMA20'].iloc[-1], df_h['Volume'].iloc[-1]
        vol_ma = df_h['Vol_MA'].iloc[-1]
        
        if any(pd.isna([c_prev, ema_prev, ema_curr, vol_curr, vol_ma])): return False, round(fib_786 * 0.98, 2)
        
        breakout = (c_prev <= ema_prev) and (curr_price > ema_curr)
        vol_spike = vol_curr > (vol_ma * VOL_SPIKE_THRESHOLD)
        
        triggered = bool(breakout and vol_spike)
        stop = round(min(df_h['Low'].tail(15).min(), fib_786) * 0.99, 2)
        return triggered, stop
    except: return False, round(fib_786 * 0.98, 2)

def calculate_confidence_score(trends: Dict[str, str], in_fib_zone: bool, hourly_triggered: bool) -> str:
    score = 0
    vals = list(trends.values())
    if len(set(vals)) == 1 and vals[0] == "上漲": score += 3
    elif vals.count("上漲") >= 2: score += 2
    if in_fib_zone: score += 2
    if hourly_triggered: score += 2
    return "高" if score >= 6 else ("中" if score >= 4 else "低")

def determine_strategy(trends: Dict[str, str], in_fib_zone: bool, hourly_triggered: bool, confidence: str) -> Dict[str, str]:
    m, w, d = trends.get("monthly", ""), trends.get("weekly", ""), trends.get("daily", "")
    if m == "上漲" and w == "上漲" and in_fib_zone and hourly_triggered:
        return {"strategy": "🚀 大主升浪起場點 (多週期共振)", "position": "70% - 100%", "color": "green"}
    if w == "上漲" and d == "上漲" and in_fib_zone:
        return {"strategy": "📈 中線波段建倉 (趨勢跟隨)", "position": "40% - 60%", "color": "blue"}
    if in_fib_zone and hourly_triggered:
        return {"strategy": "⚡ 短線衝刺/反彈 (技術訊號)", "position": "20% - 30%", "color": "orange"}
    if hourly_triggered and confidence == "高":
        return {"strategy": "⚡ 短線試單 (高信心)", "position": "15% - 25%", "color": "orange"}
    return {"strategy": "👀 觀察中 (等待點位)", "position": "0%", "color": "gray"}

def fetch_news(ticker: str) -> List[Dict[str, Any]]:
    try:
        key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d_%H')}"
        if key in NEWS_CACHE:
            t, news = NEWS_CACHE[key]
            if (datetime.now() - t).total_seconds() < CACHE_TTL_NEWS: return news
        
        items = yf.Ticker(ticker).news
        news = []
        if items:
            for item in items[:5]:
                title = item.get("title", "")
                txt = (title + " " + item.get("summary", "")).lower()
                typ = "中性"
                if any(w in txt for w in ['beat', 'surge', 'gain', 'up', 'rise', 'buy', 'upgrade', 'record', 'profit']): typ = "利好"
                elif any(w in txt for w in ['miss', 'drop', 'fall', 'down', 'sell', 'downgrade', 'lawsuit', 'loss', 'warning']): typ = "利空"
                
                pt = item.get("providerPublishTime")
                ts = datetime.fromtimestamp(pt).strftime("%Y-%m-%d %H:%M") if pt else "未知"
                news.append({"title": title, "publisher": item.get("publisher", "?"), "link": item.get("link", ""), "time": ts, "type": typ})
        
        if news: NEWS_CACHE[key] = (datetime.now(), news)
        return news
    except: return []

def analyze_single_stock(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        hist = fetch_multi_period_data(ticker)
        price = fetch_yf_price(ticker)
        
        if hist is None or price is None:
            logger.warning(f"{ticker}: 數據不完整")
            return None
        
        curr = price["price"]
        df_hourly = hist["hourly"] if not hist["hourly"].empty else pd.DataFrame()
        
        trends = {p: identify_trend_direction(hist[p], n) for p, n in [("monthly", "月線"), ("weekly", "周線"), ("daily", "日線"), ("hourly", "小時線")]}
        stages = {p: identify_wave_stage(hist[p], trends[p], n) for p, n in [("monthly", "月線"), ("weekly", "周線"), ("daily", "日線"), ("hourly", "小時線")]}
        
        fib = calculate_fib_zones(hist["daily"], curr)
        f618, f786, d_high = fib.get('fib_618', 0), fib.get('fib_786', 0), fib.get('daily_high', curr)
        
        in_zone = (curr >= f786 * 0.99) and (curr <= f618 * 1.01)
        triggered, stop = check_hourly_signal(df_hourly, curr, f786)
        conf = calculate_confidence_score(trends, in_zone, triggered)
        strat = determine_strategy(trends, in_zone, triggered, conf)
        news = fetch_news(ticker)
        dist = round(((curr - f618) / f618) * 100, 2) if f618 > 0 else 0
        
        return {
            "代碼": ticker, "現價": round(curr, 2), "漲跌額": price.get("change_amount"),
            "漲跌幅": price.get("change_percent"), "來源": price["source"],
            "建議策略": strat["strategy"], "建議倉位": strat["position"],
            "Fib 0.618": round(f618, 2), "距 0.618(%)": f"{dist}%", "止損價": stop,
            "color": strat["color"], "confidence": conf, "news": news,
            "trends": trends, "wave_stages": stages, "_d_h": d_high,
            "_d_l": fib.get('daily_low', 0), "_f618": f618, "_f786": f786
        }
    except Exception as e:
        logger.error(f"{ticker}: 分析失敗 - {str(e)}")
        return None

def analyze_stocks_parallel(tickers: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
    res = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyze_single_stock, t): t for t in tickers}
        for f in as_completed(futs):
            try:
                r = f.result(timeout=45)
                if r: res.append(r)
            except Exception as e:
                logger.error(f"{futs[f]}: 超時 - {str(e)}")
    return res

def generate_scenarios(f618, f786, d_h, trends):
    m, w, d = trends.get("monthly", ""), trends.get("weekly", ""), trends.get("daily", "")
    s1_cond = (m == "上漲" or w == "上漲")
    s3_cond = (d == "上漲" and w == "上漲")
    
    return pd.DataFrame([
        {"情境": "1. 回測 FIB 0.618", "模擬價": f"${round(f618*1.002, 2)}", "策略": "🚀 重倉" if s1_cond else "⚖️ 輕倉", "倉位": "60-80%" if s1_cond else "0-20%", "推導依據": "月/周線趨勢", "止損": f"${round(f786*0.98, 2)}"},
        {"情境": "2. 跌破 FIB 0.786", "模擬價": f"${round(f786*0.97, 2)}", "策略": "👀 離場", "倉位": "0%", "推導依據": "支撐失效", "止損": "N/A"},
        {"情境": "3. 突破前高", "模擬價": f"${round(d_h*1.01, 2)}", "策略": "⚡ 追漲" if s3_cond else "⚠️ 謹慎", "倉位": "20-30%" if s3_cond else "10-15%", "推導依據": "日/周線趨勢", "止損": f"${round(d_h*0.97, 2)}"}
    ])

def render_results(results):
    if not results:
        st.error("無法獲取數據。請檢查 API Key 或網路連線。")
        return
    
    st.subheader("📊 實時盤口總覽")
    df = pd.DataFrame(results)[["代碼", "現價", "漲跌額", "漲跌幅", "來源", "建議策略", "建議倉位", "Fib 0.618", "距 0.618(%)", "止損價", "confidence"]]
    
    def fmt_chg(r):
        if r['漲跌幅'] is None: return "N/A"
        s = "🔺" if r['漲跌幅'] > 0 else "🔻" if r['漲跌幅'] < 0 else "➖"
        a = f"{r['漲跌額']:+.2f}" if r['漲跌額'] else "0.00"
        return f"{s} {a} ({r['漲跌幅']:+.2f}%)"
    
    def fmt_conf(c): return "🟢 高" if c=="高" else ("🟡 中" if c=="中" else "⚪ 低")
    
    df['漲跌'] = df.apply(fmt_chg, axis=1)
    df['信心'] = df['confidence'].apply(fmt_conf)
    
    cols = ["代碼", "現價", "漲跌", "來源", "建議策略", "建議倉位", "Fib 0.618", "距 0.618(%)", "止損價", "信心"]
    cfg = {c: st.column_config.TextColumn(align="center") for c in cols}
    cfg["建議策略"] = st.column_config.TextColumn(width="large")
    
    st.dataframe(df[cols], use_container_width=True, hide_index=True, column_config=cfg)
    
    st.divider()
    for r in results:
        with st.expander(f"📌 **{r['代碼']}** - {r['建議策略']} (${r['現價']})", expanded=True):
            c1, c2, c3 = st.columns(3)
            delta = f"{r['漲跌額']:+.2f} ({r['漲跌幅']:+.2f}%)" if r['漲跌幅'] else "N/A"
            c1.metric("現價", f"${r['現價']}", delta=delta)
            c2.metric("Fib 0.618", f"${r['Fib 0.618']}")
            c3.metric("止損", f"${r['止損價']}")
            
            st.markdown("#### 🌊 多週期趨勢")
            cs = st.columns(4)
            for i, (p, n) in enumerate(zip(["monthly", "weekly", "daily", "hourly"], ["月線", "周線", "日線", "小時線"])):
                t = r['trends'].get(p, "?")
                em = "🔺" if t=="上漲" else "🔻" if t=="下跌" else "➖"
                cs[i].metric(n, f"{em} {t}")
                cs[i].caption(f"階段：{r['wave_stages'].get(p, '?')}")
            
            if r.get('news'):
                st.markdown("### 📰 新聞")
                for n in r['news'][:3]:
                    em = "🟢" if n['type']=="利好" else "🔴" if n['type']=="利空" else "⚪"
                    st.markdown(f"{em} **[{n['type']}]** {n['title']}")
                    st.caption(f"🕒 {n['time']} | 📰 {n['publisher']}")
            
            st.table(generate_scenarios(r["_f618"], r["_f786"], r["_d_h"], r["trends"]))

def main():
    raw = st.text_input("輸入股票代碼", value="AAPL NVDA TSLA")
    if st.button("🔍 開始掃描", use_container_width=True):
        tickers = parse_tickers(raw)
        if not tickers:
            st.warning("請輸入有效代碼")
            return
        if 'FINNHUB_KEY' not in st.session_state:
            st.error("請在側邊欄輸入 Finnhub API Key")
            return
        
        with st.spinner("數據抓取中 (自動降級備援)..."):
            res = analyze_stocks_parallel(tickers)
        render_results(res)

if __name__ == "__main__":
    main()
