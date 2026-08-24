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

# 日誌配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定義
MAX_TICKERS = 50
REQUEST_TIMEOUT = 30
CACHE_TTL_HISTORY = 600  # Finnhub 歷史數據快取 10 分鐘
CACHE_TTL_PRICE = 60     # YF 即時價格快取 1 分鐘
CACHE_TTL_NEWS = 300     # 新聞快取 5 分鐘
MIN_DATA_POINTS = 60
VOL_SPIKE_THRESHOLD = 1.5
EMA_PERIOD = 20

# FIB 黃金分割位
FIB_LEVELS = {
    '0': 0.0, '382': 0.382, '500': 0.500,
    '618': 0.618, '786': 0.786, '1000': 1.0
}

# 新聞快取
NEWS_CACHE = {}

# 頁面配置
st.set_page_config(
    page_title="多週期量化策略掃描器 (Finnhub + YF)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 多週期波浪形態量化策略掃描器")
st.caption("Finnhub 歷史數據 + Yahoo 實時價格 | 道氏趨勢 + FIB 黃金口袋區")


# ------------------------------------------------------------------------------
# 側邊欄：API Key 配置
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ API 配置")
    finnhub_key = st.text_input(
        "Finnhub API Key",
        type="password",
        help="請在 finnhub.io 免費註冊獲取 API Key",
        placeholder="輸入您的 Finnhub API Key"
    )
    
    if finnhub_key:
        st.success("✅ API Key 已設置")
        # 將 Key 存入 session state 供全局使用
        st.session_state['FINNHUB_KEY'] = finnhub_key
    else:
        st.warning("⚠️ 請輸入 Finnhub API Key 以獲取歷史數據")
        if 'FINNHUB_KEY' in st.session_state:
            del st.session_state['FINNHUB_KEY']
    
    st.markdown("---")
    st.info("💡 **說明**:\n- **歷史數據**: 使用 Finnhub (穩定/無限制)\n- **即時價格**: 使用 Yahoo Finance\n- **新聞**: 使用 Yahoo Finance")


# ------------------------------------------------------------------------------
# 輸入驗證
# ------------------------------------------------------------------------------
def validate_ticker(ticker: str) -> bool:
    if not ticker or not isinstance(ticker, str):
        return False
    ticker = ticker.strip().upper()
    pattern = r'^[A-Z0-9.\-]{1,10}$'
    return bool(re.match(pattern, ticker))

def parse_tickers(raw_input: str) -> List[str]:
    if not raw_input or not isinstance(raw_input, str):
        return []
    raw_tickers = re.split(r'[,\s,]+', raw_input)
    valid_tickers = []
    for t in raw_tickers:
        t_clean = t.strip().upper()
        if t_clean and validate_ticker(t_clean):
            if t_clean not in valid_tickers:
                valid_tickers.append(t_clean)
    if len(valid_tickers) > MAX_TICKERS:
        st.warning(f"最多只支持 {MAX_TICKERS} 個股票代碼。")
        valid_tickers = valid_tickers[:MAX_TICKERS]
    return valid_tickers


# ------------------------------------------------------------------------------
# 數據抓取模組 (混合架構)
# ------------------------------------------------------------------------------
def fetch_finnhub_history(ticker: str, resolution: str, count: int) -> Optional[pd.DataFrame]:
    """
    使用 Finnhub 抓取歷史 K 線
    resolution: D (日), W (周), M (月), 60 (小時)
    """
    api_key = st.session_state.get('FINNHUB_KEY')
    if not api_key:
        return None
    
    end_time = int(time.time())
    start_time = end_time - (count * 24 * 60 * 60) # 簡化計算天數
    
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': ticker,
        'resolution': resolution,
        'from': start_time,
        'to': end_time,
        'token': api_key
    }
    
    try:
        # 重試機制
        for attempt in range(3):
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if data['s'] == 'ok':
                    df = pd.DataFrame({
                        'Open': data['o'],
                        'High': data['h'],
                        'Low': data['l'],
                        'Close': data['c'],
                        'Volume': data['v']
                    })
                    df.index = pd.to_datetime(data['t'], unit='s')
                    return df
                else:
                    logger.warning(f"{ticker}: Finnhub 返回狀態 {data['s']}")
                    return None
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 2
                logger.warning(f"{ticker}: 觸發速率限制，等待 {wait_time}秒...")
                time.sleep(wait_time)
            else:
                logger.error(f"{ticker}: Finnhub 請求失敗 {response.status_code}")
                return None
        return None
    except Exception as e:
        logger.error(f"{ticker}: Finnhub 異常 - {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_multi_period_data(ticker_symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    """抓取多週期歷史數據 (使用 Finnhub)"""
    if 'FINNHUB_KEY' not in st.session_state:
        return None

    try:
        # 抓取不同週期
        df_monthly = fetch_finnhub_history(ticker_symbol, 'M', 1500) # 約 5 年
        df_weekly = fetch_finnhub_history(ticker_symbol, 'W', 1500)
        df_daily = fetch_finnhub_history(ticker_symbol, 'D', 1000)
        df_hourly = fetch_finnhub_history(ticker_symbol, '60', 90) # 約 3-4 個月
        
        if df_daily is None or len(df_daily) < MIN_DATA_POINTS:
            return None
        
        return {
            "monthly": df_monthly if df_monthly is not None and not df_monthly.empty else pd.DataFrame(),
            "weekly": df_weekly if df_weekly is not None and not df_weekly.empty else pd.DataFrame(),
            "daily": df_daily,
            "hourly": df_hourly if df_hourly is not None and not df_hourly.empty else pd.DataFrame()
        }
    except Exception as e:
        logger.error(f"{ticker_symbol}: 歷史數據整合失敗 - {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_yf_hourly_and_price(ticker_symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]:
    """
    使用 Yahoo Finance 抓取即時價格和備用小時線
    修復語法錯誤：添加缺失的閉合括號
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        curr_price = None
        prev_close = None
        source = "失敗"
        df_hourly_backup = pd.DataFrame()

        # 1. 嘗試獲取即時價格
        try:
            info = ticker.info or {}
            state = info.get("marketState", "").upper()
            prev_close = info.get("previousClose")
            
            if state == "PRE" and info.get("preMarketPrice"):
                curr_price = float(info["preMarketPrice"])
                source = "盤前"
            elif state == "POST" and info.get("postMarketPrice"):
                curr_price = float(info["postMarketPrice"])
                source = "盤後"
            elif info.get("regularMarketPrice"):
                curr_price = float(info["regularMarketPrice"])
                source = "常規"
        except Exception:
            pass

        # 2. 備用方案：1 分鐘 K 線
        if curr_price is None:
            try:
                df_1m = ticker.history(period="1d", interval="1m", prepost=True, timeout=10)
                if df_1m is not None and not df_1m.empty:
                    curr_price = float(df_1m['Close'].iloc[-1])
                    source = "1 分 K"
                    # 嘗試從日線獲取昨收
                    df_1d = ticker.history(period="5d", interval="1d", timeout=10)
                    if df_1d is not None and len(df_1d) >= 2:
                        prev_close = df_1d['Close'].iloc[-2]
            except Exception:
                pass
        
        # 3. 獲取備用小時線 (如果 Finnhub 失敗或需要補充)
        try:
            df_hourly_backup = ticker.history(period="1mo", interval="1h", prepost=True, timeout=10)
        except Exception:
            pass

        if curr_price is None:
            return None, None

        change_percent = None
        change_amount = None
        if prev_close and prev_close > 0:
            change_amount = round(curr_price - prev_close, 2)
            change_percent = round((change_amount / prev_close) * 100, 2)

        price_data = {
            "price": curr_price,
            "prev_close": prev_close,
            "change_amount": change_amount,
            "change_percent": change_percent,
            "source": source
        }
        
        return df_hourly_backup, price_data

    except Exception as e:
        logger.error(f"{ticker_symbol}: YF 價格抓取失敗 - {str(e)}")
        return None, None


# ------------------------------------------------------------------------------
# 技術分析模組
# ------------------------------------------------------------------------------
def identify_trend_direction(df: pd.DataFrame, period_name: str) -> str:
    if df is None or len(df) < 20:
        return "數據不足"
    try:
        highs = df['High'].values
        lows = df['Low'].values
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
        lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
        
        if hh_count >= 6 and hl_count >= 6: return "上漲"
        elif lh_count >= 6 and ll_count >= 6: return "下跌"
        else: return "盤整"
    except Exception as e:
        logger.warning(f"{period_name} 趨勢判斷失敗: {str(e)}")
        return "無法判斷"

def identify_wave_stage(df: pd.DataFrame, trend: str, period_name: str) -> str:
    if df is None or len(df) < 20: return "數據不足"
    try:
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
        current_price = df['Close'].iloc[-1]
        
        if trend == "上漲":
            if current_price > ma20 > ma50: return "推動浪 (主升)"
            elif ma50 < current_price < ma20: return "回調浪 (次級)"
            else: return "盤整延伸"
        elif trend == "下跌":
            if current_price < ma20 < ma50: return "推動浪 (主跌)"
            elif ma50 > current_price > ma20: return "回調浪 (反彈)"
            else: return "盤整延伸"
        else: return "橫向盤整"
    except Exception as e:
        logger.warning(f"{period_name} 波浪階段判斷失敗: {str(e)}")
        return "無法判斷"

def calculate_fib_zones(df_daily: pd.DataFrame, curr_price: float) -> Dict[str, float]:
    if df_daily is None or len(df_daily) < 60: return {}
    daily_recent = df_daily.tail(60)
    daily_high = max(daily_recent['High'].max(), curr_price)
    daily_low = min(daily_recent['Low'].min(), curr_price)
    price_range = daily_high - daily_low
    
    fib_zones = {}
    for name, level in FIB_LEVELS.items():
        fib_zones[f'fib_{name}'] = daily_high - level * price_range
    fib_zones['daily_high'] = daily_high
    fib_zones['daily_low'] = daily_low
    return fib_zones

def check_hourly_signal(df_hourly: pd.DataFrame, curr_price: float, fib_786: float) -> Tuple[bool, float]:
    if df_hourly is None or len(df_hourly) < 20:
        return False, round(fib_786 * 0.98, 2)
    try:
        df_h = df_hourly.copy()
        df_h['EMA20'] = df_h['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        df_h['Vol_MA'] = df_h['Volume'].rolling(20).mean()
        
        if len(df_h) < 2: return False, round(fib_786 * 0.98, 2)
        
        c_prev = df_h['Close'].iloc[-2]
        ema_prev = df_h['EMA20'].iloc[-2]
        ema_curr = df_h['EMA20'].iloc[-1]
        vol_curr = df_h['Volume'].iloc[-1]
        vol_ma = df_h['Vol_MA'].iloc[-1]
        
        if any(pd.isna([c_prev, ema_prev, ema_curr, vol_curr, vol_ma])):
            return False, round(fib_786 * 0.98, 2)
        
        h_breakout = (c_prev <= ema_prev) and (curr_price > ema_curr)
        h_vol_spike = vol_curr > (vol_ma * VOL_SPIKE_THRESHOLD)
        
        hourly_triggered = bool(h_breakout and h_vol_spike)
        hourly_low_min = df_h['Low'].tail(15).min()
        stop_loss = round(min(hourly_low_min, fib_786) * 0.99, 2)
        
        return hourly_triggered, stop_loss
    except Exception as e:
        logger.warning(f"小時線訊號計算失敗：{str(e)}")
        return False, round(fib_786 * 0.98, 2)

def calculate_confidence_score(trends: Dict[str, str], in_fib_zone: bool, hourly_triggered: bool) -> str:
    score = 0
    trend_values = list(trends.values())
    if len(set(trend_values)) == 1 and trend_values[0] == "上漲": score += 3
    elif trend_values.count("上漲") >= 2: score += 2
    if in_fib_zone: score += 2
    if hourly_triggered: score += 2
    return "高" if score >= 6 else ("中" if score >= 4 else "低")

def determine_strategy(trends: Dict[str, str], in_fib_zone: bool, hourly_triggered: bool, confidence: str) -> Dict[str, str]:
    m, w, d = trends.get("monthly", ""), trends.get("weekly", ""), trends.get("daily", "")
    
    if m == "上漲" and w == "上漲" and in_fib_zone and hourly_triggered:
        return {"strategy": "🚀 大主升浪起場點 (多週期共振)", "position": "70% - 100%", "color": "green"}
    elif w == "上漲" and d == "上漲" and in_fib_zone:
        return {"strategy": "📈 中線波段建倉 (趨勢跟隨)", "position": "40% - 60%", "color": "blue"}
    elif in_fib_zone and hourly_triggered:
        return {"strategy": "⚡ 短線衝刺/反彈 (技術訊號)", "position": "20% - 30%", "color": "orange"}
    elif hourly_triggered and confidence == "高":
        return {"strategy": "⚡ 短線試單 (高信心)", "position": "15% - 25%", "color": "orange"}
    else:
        return {"strategy": "👀 觀察中 (等待點位)", "position": "0%", "color": "gray"}

def fetch_news(ticker_symbol: str) -> List[Dict[str, Any]]:
    try:
        cache_key = f"{ticker_symbol}_{datetime.now().strftime('%Y-%m-%d_%H')}"
        if cache_key in NEWS_CACHE:
            cache_time, cached_news = NEWS_CACHE[cache_key]
            if (datetime.now() - cache_time).total_seconds() < CACHE_TTL_NEWS:
                return cached_news
        
        ticker = yf.Ticker(ticker_symbol)
        news_list = []
        try:
            items = ticker.news
            if items and isinstance(items, list):
                for item in items[:5]:
                    if not isinstance(item, dict): continue
                    title = item.get("title", "無標題")
                    text_content = (title + " " + item.get("summary", "")).lower()
                    news_type = "中性"
                    if any(w in text_content for w in ['beat', 'surge', 'gain', 'up', 'rise', 'buy', 'upgrade', 'record', 'profit', 'strong', 'growth']):
                        news_type = "利好"
                    elif any(w in text_content for w in ['miss', 'drop', 'fall', 'down', 'sell', 'downgrade', 'lawsuit', 'loss', 'warning', 'weak', 'decline']):
                        news_type = "利空"
                    
                    pub_time = item.get("providerPublishTime")
                    time_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M") if pub_time else "未知時間"
                    
                    news_list.append({
                        "title": title,
                        "publisher": item.get("publisher", "未知"),
                        "link": item.get("link", ""),
                        "time": time_str,
                        "type": news_type
                    })
        except Exception: pass
        
        if news_list: NEWS_CACHE[cache_key] = (datetime.now(), news_list)
        return news_list
    except Exception as e:
        logger.error(f"{ticker_symbol}: 新聞抓取異常 - {str(e)}")
        return []

def analyze_single_stock(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        # 1. 抓取 Finnhub 歷史數據
        historical_data = fetch_multi_period_data(ticker)
        
        # 2. 抓取 YF 即時價格
        df_hourly_yf, price_data = fetch_yf_hourly_and_price(ticker)
        
        if historical_data is None or price_data is None:
            logger.warning(f"{ticker}: 數據不完整 (Finnhub Key 缺失或 YF 限制)")
            return None
        
        curr_price = price_data["price"]
        change_percent = price_data.get("change_percent")
        change_amount = price_data.get("change_amount")
        source = price_data["source"]
        
        # 優先使用 Finnhub 小時線，若無則用 YF 備用
        df_hourly = historical_data["hourly"] if not historical_data["hourly"].empty else df_hourly_yf
        
        # 分析流程
        trends = {
            "monthly": identify_trend_direction(historical_data["monthly"], "月線"),
            "weekly": identify_trend_direction(historical_data["weekly"], "周線"),
            "daily": identify_trend_direction(historical_data["daily"], "日線"),
            "hourly": identify_trend_direction(df_hourly, "小時線")
        }
        
        wave_stages = {
            "monthly": identify_wave_stage(historical_data["monthly"], trends["monthly"], "月線"),
            "weekly": identify_wave_stage(historical_data["weekly"], trends["weekly"], "周線"),
            "daily": identify_wave_stage(historical_data["daily"], trends["daily"], "日線"),
            "hourly": identify_wave_stage(df_hourly, trends["hourly"], "小時線")
        }
        
        fib_data = calculate_fib_zones(historical_data["daily"], curr_price)
        fib_618 = fib_data.get('fib_618', 0)
        fib_786 = fib_data.get('fib_786', 0)
        daily_high = fib_data.get('daily_high', curr_price)
        
        in_fib_zone = (curr_price >= fib_786 * 0.99) and (curr_price <= fib_618 * 1.01)
        hourly_triggered, stop_loss = check_hourly_signal(df_hourly, curr_price, fib_786)
        confidence = calculate_confidence_score(trends, in_fib_zone, hourly_triggered)
        strategy_info = determine_strategy(trends, in_fib_zone, hourly_triggered, confidence)
        news_list = fetch_news(ticker)
        dist_618 = round(((curr_price - fib_618) / fib_618) * 100, 2) if fib_618 > 0 else 0
        
        return {
            "代碼": ticker, "現價": round(curr_price, 2), "漲跌額": change_amount,
            "漲跌幅": change_percent, "來源": source, "建議策略": strategy_info["strategy"],
            "建議倉位": strategy_info["position"], "Fib 0.618": round(fib_618, 2),
            "距 0.618(%)": f"{dist_618}%", "止損價": stop_loss, "color": strategy_info["color"],
            "confidence": confidence, "news": news_list, "trends": trends,
            "wave_stages": wave_stages, "_d_h": daily_high, "_d_l": fib_data.get('daily_low', 0),
            "_f618": fib_618, "_f786": fib_786
        }
    except Exception as e:
        logger.error(f"{ticker}: 分析過程失敗 - {str(e)}")
        return None

def analyze_stocks_parallel(tickers: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
    # 降低並發數至 3，進一步避免 YF 限制
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(analyze_single_stock, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                res = future.result(timeout=REQUEST_TIMEOUT * 2)
                if res: results.append(res)
            except Exception as e:
                logger.error(f"{t}: 執行超時 - {str(e)}")
    return results

def generate_scenarios(f618: float, f786: float, d_h: float, trends: Dict[str, str]) -> pd.DataFrame:
    m, w, d = trends.get("monthly", "盤整"), trends.get("weekly", "盤整"), trends.get("daily", "盤整")
    
    s1_basis = "月線 + 周線趨勢"
    s1_strat = "🚀 長/中線重倉" if (m=="上漲" or w=="上漲") else "⚖️ 觀望/輕倉"
    s1_pos = "60%-80%" if (m=="上漲" or w=="上漲") else "0%-20%"
    
    s2_basis = "FIB 支撐失效原理"
    
    s3_basis = "日線 + 周線趨勢"
    s3_strat = "⚡ 順勢追漲" if (d=="上漲" and w=="上漲") else "⚠️ 謹慎追高"
    s3_pos = "20%-30%" if (d=="上漲" and w=="上漲") else "10%-15%"
    
    return pd.DataFrame([
        {"情境": "1. 回測 FIB 0.618 黃金區", "模擬價": f"${round(f618*1.002, 2)}", "策略": s1_strat, "倉位": s1_pos, "推導依據": s1_basis, "止損": f"${round(f786*0.98, 2)}"},
        {"情境": "2. 跌破 FIB 0.786 支撐", "模擬價": f"${round(f786*0.97, 2)}", "策略": "👀 離場觀望", "倉位": "0%", "推導依據": s2_basis, "止損": "N/A"},
        {"情境": "3. 突破前高阻力", "模擬價": f"${round(d_h*1.01, 2)}", "策略": s3_strat, "倉位": s3_pos, "推導依據": s3_basis, "止損": f"${round(d_h*0.97, 2)}"}
    ])

def render_results(results: List[Dict[str, Any]]):
    if not results:
        st.error("無法取得數據。請檢查 Finnhub API Key 是否正確，或稍後再試 (Yahoo 可能暫時限制)。")
        return
    
    st.subheader("📊 實時盤口總覽")
    df_display = pd.DataFrame(results)[["代碼", "現價", "漲跌額", "漲跌幅", "來源", "建議策略", "建議倉位", "Fib 0.618", "距 0.618(%)", "止損價", "confidence"]]
    
    def fmt_change(row):
        if row['漲跌幅'] is None: return "N/A"
        sign = "🔺" if row['漲跌幅'] > 0 else "🔻" if row['漲跌幅'] < 0 else "➖"
        amt = f"{row['漲跌額']:+.2f}" if row['漲跌額'] else "0.00"
        return f"{sign} {amt} ({row['漲跌幅']:+.2f}%)"
    
    def fmt_conf(c): return "🟢 高" if c=="高" else ("🟡 中" if c=="中" else "⚪ 低")
    
    df_display['漲跌'] = df_display.apply(fmt_change, axis=1)
    df_display['信心'] = df_display['confidence'].apply(fmt_conf)
    
    cols = ["代碼", "現價", "漲跌", "來源", "建議策略", "建議倉位", "Fib 0.618", "距 0.618(%)", "止損價", "信心"]
    df_final = df_display[cols]
    
    col_config = {c: st.column_config.TextColumn(width="medium", align="center") for c in cols}
    col_config["建議策略"] = st.column_config.TextColumn(width="large", align="center")
    col_config["代碼"] = st.column_config.TextColumn(width="small", align="center")
    
    st.dataframe(df_final, use_container_width=True, hide_index=True, column_config=col_config)
    
    st.divider()
    st.subheader("📱 多週期詳情與壓力測試")
    
    for res in results:
        with st.expander(f"📌 **{res['代碼']}** - {res['建議策略']} (現價: ${res['現價']})", expanded=True):
            cp = res.get('漲跌幅')
            ca = res.get('漲跌額')
            delta_txt = f"{ca:+.2f} ({cp:+.2f}%)" if cp is not None else "N/A"
            
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"${res['現價']}", delta=delta_txt)
            c2.metric("Fib 0.618", f"${res['Fib 0.618']}")
            c3.metric("止損價", f"${res['止損價']}")
            
            st.markdown("#### 🌊 多週期趨勢")
            cols = st.columns(4)
            for i, (p, name) in enumerate(zip(["monthly", "weekly", "daily", "hourly"], ["月線", "周線", "日線", "小時線"])):
                t = res['trends'].get(p, "?")
                s = res['wave_stages'].get(p, "?")
                emoji = "🔺" if t=="上漲" else "🔻" if t=="下跌" else "➖"
                with cols[i]:
                    st.metric(f"{name}", f"{emoji} {t}")
                    st.caption(f"階段：{s}")
            
            if res.get('news'):
                st.markdown("### 📰 新聞")
                for n in res['news'][:3]:
                    em = "🟢" if n['type']=="利好" else "🔴" if n['type']=="利空" else "⚪"
                    st.markdown(f"{em} **[{n['type']}]** {n['title']}")
                    st.caption(f"🕒 {n['time']} | 📰 {n['publisher']}")
            
            st.table(generate_scenarios(res["_f618"], res["_f786"], res["_d_h"], res["trends"]))

def main():
    raw = st.text_input("輸入股票代碼 (空格/逗號分隔):", value="AAPL NVDA TSLA")
    if st.button("🔍 開始混合架構掃描", use_container_width=True):
        tickers = parse_tickers(raw)
        if not tickers:
            st.warning("請輸入有效代碼")
            return
        
        if 'FINNHUB_KEY' not in st.session_state:
            st.error("❌ 請在左側側邊欄輸入 Finnhub API Key！")
            return
            
        with st.spinner(f"正在抓取數據 (Finnhub 歷史 + YF 實時)..."):
            results = analyze_stocks_parallel(tickers, max_workers=3)
        render_results(results)

if __name__ == "__main__":
    main()
