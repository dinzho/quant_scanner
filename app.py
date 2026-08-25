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
from bs4 import BeautifulSoup
import requests
import time

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
REQUEST_TIMEOUT = 45
CACHE_TTL_HISTORY = 300
CACHE_TTL_PRICE = 60
MIN_DATA_POINTS = 60
VOL_SPIKE_THRESHOLD = 1.5
EMA_PERIOD = 20

# FIB 黃金分割位
FIB_LEVELS = {
    '0': 0.0,
    '382': 0.382,
    '500': 0.500,
    '618': 0.618,
    '786': 0.786,
    '1000': 1.0
}

# 頁面配置
st.set_page_config(
    page_title="多週期量化策略掃描器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 多週期波浪形態量化策略掃描器")
st.caption("道氏趨勢 + FIB 黃金口袋區 + 多週期共振 (月/周/日/小時)")


# ------------------------------------------------------------------------------
# 輸入驗證與安全防護
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
        st.warning(f"最多只支持 {MAX_TICKERS} 個股票代碼，已自動截斷。")
        valid_tickers = valid_tickers[:MAX_TICKERS]
    
    return valid_tickers


# ------------------------------------------------------------------------------
# 快取優化：分離歷史數據和實時價格
# ------------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_multi_period_data(ticker_symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        df_monthly = ticker.history(period="5y", interval="3mo")
        df_weekly = ticker.history(period="2y", interval="1wk")
        df_daily = ticker.history(period="1y", interval="1d")
        df_hourly = ticker.history(period="1mo", interval="1h", prepost=True)
        
        if df_daily is None or len(df_daily) < MIN_DATA_POINTS:
            logger.warning(f"{ticker_symbol}: 日線數據不足")
            return None
        
        return {
            "monthly": df_monthly if df_monthly is not None and not df_monthly.empty else pd.DataFrame(),
            "weekly": df_weekly if df_weekly is not None and not df_weekly.empty else pd.DataFrame(),
            "daily": df_daily,
            "hourly": df_hourly if df_hourly is not None and not df_hourly.empty else pd.DataFrame()
        }
    
    except Exception as e:
        logger.error(f"{ticker_symbol}: 歷史數據抓取失敗 - {str(e)}")
        return None


@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_current_price(ticker_symbol: str) -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        curr_price = None
        prev_close = None
        source = "失敗"

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
        except (ValueError, TypeError, KeyError):
            pass

        if curr_price is None:
            try:
                df_1m = ticker.history(period="1d", interval="1m", prepost=True)
                if df_1m is not None and not df_1m.empty:
                    close_series = df_1m['Close'].dropna()
                    if not close_series.empty:
                        curr_price = float(close_series.iloc[-1])
                        source = "1 分 K"
                        df_1d = ticker.history(period="5d", interval="1d")
                        if df_1d is not None and len(df_1d) >= 2:
                            prev_close = df_1d['Close'].iloc[-2]
            except Exception:
                pass

        if curr_price is None:
            return None

        change_percent = None
        change_amount = None
        if prev_close and prev_close > 0:
            change_amount = round(curr_price - prev_close, 2)
            change_percent = round((change_amount / prev_close) * 100, 2)

        return {
            "price": curr_price,
            "prev_close": prev_close,
            "change_amount": change_amount,
            "change_percent": change_percent,
            "source": source
        }

    except Exception as e:
        logger.error(f"{ticker_symbol}: 價格抓取失敗 - {str(e)}")
        return None


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
        
        if hh_count >= 6 and hl_count >= 6:
            return "上漲"
        elif lh_count >= 6 and ll_count >= 6:
            return "下跌"
        else:
            return "盤整"
            
    except Exception as e:
        logger.warning(f"{period_name} 趨勢判斷失敗: {str(e)}")
        return "無法判斷"


def identify_wave_stage(df: pd.DataFrame, trend: str, period_name: str) -> str:
    if df is None or len(df) < 20:
        return "數據不足"
    
    try:
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20
        current_price = df['Close'].iloc[-1]
        
        if trend == "上漲":
            if current_price > ma20 > ma50:
                return "推動浪 (主升)"
            elif ma50 < current_price < ma20:
                return "回調浪 (次級)"
            else:
                return "盤整延伸"
                
        elif trend == "下跌":
            if current_price < ma20 < ma50:
                return "推動浪 (主跌)"
            elif ma50 > current_price > ma20:
                return "回調浪 (反彈)"
            else:
                return "盤整延伸"
        else:
            return "橫向盤整"
            
    except Exception as e:
        logger.warning(f"{period_name} 波浪階段判斷失敗: {str(e)}")
        return "無法判斷"


def calculate_fib_zones(df_daily: pd.DataFrame, curr_price: float) -> Dict[str, float]:
    if df_daily is None or len(df_daily) < 60:
        return {}
    
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
        df_hourly = df_hourly.copy()
        
        df_hourly['EMA20'] = df_hourly['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        df_hourly['Vol_MA'] = df_hourly['Volume'].rolling(20).mean()
        
        if len(df_hourly) < 2:
            return False, round(fib_786 * 0.98, 2)
        
        c_prev = df_hourly['Close'].iloc[-2]
        ema_prev = df_hourly['EMA20'].iloc[-2]
        ema_curr = df_hourly['EMA20'].iloc[-1]
        vol_curr = df_hourly['Volume'].iloc[-1]
        vol_prev = df_hourly['Volume'].iloc[-2]
        vol_ma = df_hourly['Vol_MA'].iloc[-1]
        
        if any(pd.isna([c_prev, ema_prev, ema_curr, vol_curr, vol_prev, vol_ma])):
            return False, round(fib_786 * 0.98, 2)
        
        h_breakout = (c_prev <= ema_prev) and (curr_price > ema_curr)
        h_vol_spike = max(vol_curr, vol_prev) > (vol_ma * VOL_SPIKE_THRESHOLD)
        
        hourly_triggered = bool(h_breakout and h_vol_spike)
        
        hourly_low_min = df_hourly['Low'].tail(15).min()
        hourly_stop_loss = round(min(hourly_low_min, fib_786) * 0.99, 2)
        
        return hourly_triggered, hourly_stop_loss
    
    except Exception as e:
        logger.warning(f"小時線訊號計算失敗：{str(e)}")
        return False, round(fib_786 * 0.98, 2)


def calculate_confidence_score(trends: Dict[str, str], in_fib_zone: bool, hourly_triggered: bool) -> str:
    score = 0
    
    trend_values = list(trends.values())
    if len(set(trend_values)) == 1 and trend_values[0] == "上漲":
        score += 3
    elif trend_values.count("上漲") >= 2:
        score += 2
    
    if in_fib_zone:
        score += 2
    
    if hourly_triggered:
        score += 2
    
    if score >= 6:
        return "高"
    elif score >= 4:
        return "中"
    else:
        return "低"


def determine_strategy(trends: Dict[str, str], in_fib_zone: bool, 
                      hourly_triggered: bool, curr_price: float, 
                      fib_618: float, confidence: str) -> Dict[str, str]:
    
    monthly_trend = trends.get("monthly", "")
    weekly_trend = trends.get("weekly", "")
    daily_trend = trends.get("daily", "")
    
    if monthly_trend == "上漲" and weekly_trend == "上漲" and in_fib_zone and hourly_triggered:
        return {
            "strategy": "🚀 大主升浪起場點 (多週期共振)",
            "position": "70% - 100%",
            "color": "green"
        }
    elif weekly_trend == "上漲" and daily_trend == "上漲" and in_fib_zone:
        return {
            "strategy": "📈 中線波段建倉 (趨勢跟隨)",
            "position": "40% - 60%",
            "color": "blue"
        }
    elif in_fib_zone and hourly_triggered:
        return {
            "strategy": "⚡ 短線衝刺/反彈 (技術訊號)",
            "position": "20% - 30%",
            "color": "orange"
        }
    elif hourly_triggered and confidence == "高":
        return {
            "strategy": "⚡ 短線試單 (高信心)",
            "position": "15% - 25%",
            "color": "orange"
        }
    else:
        return {
            "strategy": "👀 觀察中 (等待點位)",
            "position": "0%",
            "color": "gray"
        }

# ------------------------------------------------------------------------------
# 核心分析函數
# ------------------------------------------------------------------------------
def analyze_single_stock(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        historical_data = fetch_multi_period_data(ticker)
        price_data = fetch_current_price(ticker)
        
        if historical_data is None or price_data is None:
            logger.warning(f"{ticker}: 數據不完整，跳過分析")
            return None
        
        curr_price = price_data["price"]
        change_percent = price_data.get("change_percent")
        change_amount = price_data.get("change_amount")
        source = price_data["source"]
        
        df_monthly = historical_data["monthly"]
        df_weekly = historical_data["weekly"]
        df_daily = historical_data["daily"]
        df_hourly = historical_data["hourly"]
        
        trends = {
            "monthly": identify_trend_direction(df_monthly, "月線"),
            "weekly": identify_trend_direction(df_weekly, "周線"),
            "daily": identify_trend_direction(df_daily, "日線"),
            "hourly": identify_trend_direction(df_hourly, "小時線")
        }
        
        wave_stages = {
            "monthly": identify_wave_stage(df_monthly, trends["monthly"], "月線"),
            "weekly": identify_wave_stage(df_weekly, trends["weekly"], "周線"),
            "daily": identify_wave_stage(df_daily, trends["daily"], "日線"),
            "hourly": identify_wave_stage(df_hourly, trends["hourly"], "小時線")
        }
        
        fib_data = calculate_fib_zones(df_daily, curr_price)
        fib_618 = fib_data.get('fib_618', 0)
        fib_786 = fib_data.get('fib_786', 0)
        daily_high = fib_data.get('daily_high', curr_price)
        
        in_fib_zone = (curr_price >= fib_786 * 0.99) and (curr_price <= fib_618 * 1.01)
        hourly_triggered, hourly_stop_loss = check_hourly_signal(df_hourly, curr_price, fib_786)
        confidence = calculate_confidence_score(trends, in_fib_zone, hourly_triggered)
        strategy_info = determine_strategy(trends, in_fib_zone, hourly_triggered, curr_price, fib_618, confidence)
        
        dist_618 = round(((curr_price - fib_618) / fib_618) * 100, 2) if fib_618 > 0 else 0
        
        # 新增：價格位置狀態標記 (供情境生成使用)
        price_position = "above_618"
        if curr_price < fib_786 * 0.99:
            price_position = "below_786" # 已跌破深層支撐
        elif curr_price < fib_618 * 0.99:
            price_position = "between_618_786" # 跌穿 0.618 但在 0.786 之上
        elif curr_price > daily_high * 0.99:
            price_position = "breakout_high" # 突破前高
            
        return {
            "代碼": ticker,
            "現價": round(curr_price, 2),
            "漲跌額": change_amount,
            "漲跌幅": change_percent,
            "來源": source,
            "建議策略": strategy_info["strategy"],
            "建議倉位": strategy_info["position"],
            "Fib 0.618": round(fib_618, 2),
            "距 0.618(%)": f"{dist_618}%",
            "止損價": hourly_stop_loss,
            "color": strategy_info["color"],
            "confidence": confidence,
            "trends": trends,
            "wave_stages": wave_stages,
            "_d_h": daily_high,
            "_d_l": fib_data.get('daily_low', 0),
            "_f618": fib_618,
            "_f786": fib_786,
            "_price_pos": price_position # 傳遞給情境生成函數
        }
    
    except Exception as e:
        logger.error(f"{ticker}: 分析過程失敗 - {str(e)}")
        return None


def analyze_stocks_parallel(tickers: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(analyze_single_stock, ticker): ticker 
            for ticker in tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=REQUEST_TIMEOUT)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"{ticker}: 執行超時或失敗 - {str(e)}")
    
    return results


# ------------------------------------------------------------------------------
# 壓力測試情景生成 (已修復：加入現價位置感知)
# ------------------------------------------------------------------------------
def generate_scenarios(f618: float, f786: float, d_h: float, trends: Dict[str, str], price_pos: str) -> pd.DataFrame:
    """
    根據現價相對 FIB 的位置，動態調整情境策略
    price_pos: 'above_618', 'between_618_786', 'below_786', 'breakout_high'
    """
    
    weekly_trend = trends.get("weekly", "盤整")
    daily_trend = trends.get("daily", "盤整")
    is_uptrend = weekly_trend == "上漲"
    
    scenarios = []
    
    # --- 情境 1: FIB 0.618 相關 ---
    s1_title = "1. 回測 FIB 0.618 黃金區"
    s1_price = f"${round(f618 * 1.002, 2)}"
    
    if price_pos == "above_618":
        # 正常情況：現價在上方，等待回測
        s1_strat = "🚀 長/中線重倉" if is_uptrend else "⚖️ 中線建倉"
        s1_pos = "70%-100%" if is_uptrend else "40%-60%"
        s1_basis = "上升趨勢中的回調支撐"
    elif price_pos in ["between_618_786", "below_786"]:
        # 異常情況：現價已跌破，0.618 從支撐變為阻力
        s1_title = "1. 反彈測試 FIB 0.618 (已轉阻力)"
        s1_strat = "📉 減倉/離場機會"
        s1_pos = "0%-20%"
        s1_basis = "支撐失守轉為阻力 (看跌)"
    else:
        # 其他情況
        s1_strat = "持有" if is_uptrend else "觀望"
        s1_pos = "50%"
        s1_basis = "常規支撐測試"

    scenarios.append({
        "情境": s1_title,
        "模擬價": s1_price,
        "策略": s1_strat,
        "倉位": s1_pos,
        "推導依據": s1_basis,
        "止損": f"${round(f786 * 0.98, 2)}" if price_pos == "above_618" else "N/A (阻力位)"
    })
    
    # --- 情境 2: FIB 0.786 相關 ---
    s2_title = "2. 跌破 FIB 0.786 支撐"
    s2_price = f"${round(f786 * 0.97, 2)}"
    
    if price_pos == "below_786":
        # 已經跌破了，這裡應該是確認續跌
        s2_title = "2. 確認跌破 FIB 0.786 (趨勢轉弱)"
        s2_strat = "🛑 嚴格止損/空頭佈局"
        s2_basis = "深度回調確認，趨勢可能反轉"
    else:
        s2_strat = "👀 離場觀望"
        s2_basis = "最後防線失守"

    scenarios.append({
        "情境": s2_title,
        "模擬價": s2_price,
        "策略": s2_strat,
        "倉位": "0%",
        "推導依據": s2_basis,
        "止損": "N/A"
    })
    
    # --- 情境 3: 突破前高 ---
    s3_title = "3. 突破前高阻力"
    s3_price = f"${round(d_h * 1.01, 2)}"
    
    if price_pos == "breakout_high":
        # 已經突破了
        s3_title = "3. 已突破前高 (趨勢延續)"
        s3_strat = "🏃 順勢加倉/移動止損"
        s3_basis = "新高確認，動能強勁"
    elif daily_trend == "上漲" and weekly_trend == "上漲":
        s3_strat = "⚡ 順勢追漲"
        s3_pos_val = "20%-30%"
        s3_basis = "多頭趨勢突破"
    else:
        s3_strat = "⚠️ 謹慎追高/假突破風險"
        s3_pos_val = "10%-15%"
        s3_basis = "震盪市突破風險"

    scenarios.append({
        "情境": s3_title,
        "模擬價": s3_price,
        "策略": s3_strat,
        "倉位": "0%" if price_pos == "below_786" else ("20%-30%" if is_uptrend else "10%-15%"),
        "推導依據": s3_basis if price_pos != "breakout_high" else "已實現突破",
        "止損": f"${round(d_h * 0.97, 2)}"
    })
    
    return pd.DataFrame(scenarios)


# ------------------------------------------------------------------------------
# 主界面邏輯
# ------------------------------------------------------------------------------
def render_results(results: List[Dict[str, Any]]):
    if not results:
        st.error("無法取得相關股票數據，請檢查輸入代碼或網路連線。")
        return
    
    st.subheader("📊 實時盤口總覽")
    
    df_display = pd.DataFrame(results)[
        ["代碼", "現價", "漲跌額", "漲跌幅", "來源", "建議策略", "建議倉位", "Fib 0.618", "距 0.618(%)", "止損價", "confidence"]
    ]
    
    def format_change(row):
        if row['漲跌幅'] is None:
            return "N/A"
        sign = "🔺" if row['漲跌幅'] > 0 else "🔻" if row['漲跌幅'] < 0 else "➖"
        change_amt = f"{row['漲跌額']:+.2f}" if row['漲跌額'] else "0.00"
        return f"{sign} {change_amt} ({row['漲跌幅']:+.2f}%)"
    
    def format_confidence(conf):
        if conf == "高":
            return "🟢 高"
        elif conf == "中":
            return "🟡 中"
        else:
            return "⚪ 低"
    
    df_display['漲跌'] = df_display.apply(format_change, axis=1)
    df_display['信心'] = df_display['confidence'].apply(format_confidence)
    df_display = df_display.drop(columns=['漲跌額', '漲跌幅', 'confidence'])
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("📱 手機卡片與多週期詳情")
    
    for res in results:
        # 動態警告提示
        pos = res.get("_price_pos", "above_618")
        if pos == "below_786":
            st.error(f"⚠️ **{res['代碼']} 警報**: 現價已跌破 FIB 0.786 深層支撐，趨勢轉弱風險高！")
        elif pos == "between_618_786":
            st.warning(f"⚠️ **{res['代碼']} 注意**: 現價已跌破 FIB 0.618 關鍵支撐，正在測試 0.786。")
        
        with st.expander(
            f"📌 **{res['代碼']}** - {res['建議策略']} (現價: ${res['現價']})",
            expanded=True
        ):
            change_percent = res.get('漲跌幅')
            change_amount = res.get('漲跌額')
            
            if change_percent is not None:
                delta_color = "normal"
                delta_text = f"{change_amount:+.2f} ({change_percent:+.2f}%)" if change_amount else "0.00 (0.00%)"
            else:
                delta_text = "N/A"
            
            m1, m2, m3 = st.columns(3)
            m1.metric("現價", f"${res['現價']}", delta=delta_text, delta_color=delta_color)
            m2.metric("Fib 0.618", f"${res['Fib 0.618']}")
            m3.metric("建議止損", f"${res['止損價']}")
            
            st.markdown("#### 🌊 多週期趨勢分析")
            cols = st.columns(4)
            periods = ["monthly", "weekly", "daily", "hourly"]
            period_names = ["月線", "周線", "日線", "小時線"]
            
            for i, (period, name) in enumerate(zip(periods, period_names)):
                trend = res['trends'].get(period, "未知")
                stage = res['wave_stages'].get(period, "未知")
                trend_emoji = "🔺" if trend == "上漲" else "🔻" if trend == "下跌" else "➖"
                with cols[i]:
                    st.metric(f"{name}趨勢", f"{trend_emoji} {trend}")
                    st.caption(f"階段：{stage}")
            
            st.caption(
                f"**建議倉位**：{res['建議倉位']}｜"
                f"**距離 0.618**：{res['距 0.618(%)']}｜"
                f"**信心等級**：{res['confidence']}｜"
                f"**價格來源**：{res['來源']}"
            )
            
            # 壓力測試情景 (傳入 price_pos)
            scenarios_df = generate_scenarios(
                res["_f618"],
                res["_f786"],
                res["_d_h"],
                res["trends"],
                res["_price_pos"]
            )
            st.table(scenarios_df)


def main():
    raw_input = st.text_input(
        "輸入股票代碼 (支援空格/逗號分隔):",
        value="TSLA, MU, NVDA",
        help="例如：AAPL, MSFT, GOOGL"
    )
    
    if st.button("🔍 開始多週期掃描分析", use_container_width=True):
        tickers = parse_tickers(raw_input)
        
        if not tickers:
            st.warning("請輸入有效的股票代碼！")
            return
        
        with st.spinner(f"正即時連線抓取 {len(tickers)} 支股票的多週期數據..."):
            results = analyze_stocks_parallel(tickers, max_workers=min(5, len(tickers)))
        
        render_results(results)


if __name__ == "__main__":
    main()
