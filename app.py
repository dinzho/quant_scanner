--- app.py (原始)
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


+++ app.py (修改后)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

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
MAX_TICKERS = 50  # 最大股票數量限制
REQUEST_TIMEOUT = 30  # API 請求超時時間 (秒)
CACHE_TTL_HISTORY = 300  # 歷史數據快取時間 (秒)
CACHE_TTL_PRICE = 60  # 實時價格快取時間 (秒)
MIN_DAILY_DATA = 60  # 最小日線數據要求
MIN_HOURLY_DATA = 20  # 最小小時線數據要求
VOL_SPIKE_THRESHOLD = 1.2  # 成交量放大倍數閾值
FIB_LEVELS = {
    '500': 0.500,
    '618': 0.618,
    '786': 0.786
}

# 頁面配置
st.set_page_config(
    page_title="量化策略掃描器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 動態量化策略掃描器")
st.caption("多週期道氏 + FIB 黃金口袋區 (手機響應版)")


# ------------------------------------------------------------------------------
# 輸入驗證與安全防護
# ------------------------------------------------------------------------------
def validate_ticker(ticker: str) -> bool:
    """驗證股票代碼格式"""
    if not ticker or not isinstance(ticker, str):
        return False
    ticker = ticker.strip().upper()
    # 基本格式檢查：字母、數字、點號、連字號，長度 1-10
    pattern = r'^[A-Z0-9.\-]{1,10}$'
    return bool(re.match(pattern, ticker))


def parse_tickers(raw_input: str) -> List[str]:
    """解析並驗證股票代碼列表"""
    if not raw_input or not isinstance(raw_input, str):
        return []

    # 分割代碼（支援空格、逗號、頓號）
    raw_tickers = re.split(r'[,\s,]+', raw_input)

    # 驗證並過濾
    valid_tickers = []
    for t in raw_tickers:
        t_clean = t.strip().upper()
        if t_clean and validate_ticker(t_clean):
            if t_clean not in valid_tickers:  # 去重
                valid_tickers.append(t_clean)

    # 數量限制
    if len(valid_tickers) > MAX_TICKERS:
        st.warning(f"最多只支持 {MAX_TICKERS} 個股票代碼，已自動截斷。")
        valid_tickers = valid_tickers[:MAX_TICKERS]

    return valid_tickers


# ------------------------------------------------------------------------------
# 快取優化：分離歷史數據和實時價格
# ------------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_historical_data(ticker_symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    """抓取歷史 K 線數據（日線和小時線）"""
    try:
        ticker = yf.Ticker(ticker_symbol)

        # 抓取日線和小時線
        df_daily = ticker.history(period="1y", interval="1d")
        df_hourly = ticker.history(period="1mo", interval="1h", prepost=True)

        # 數據質量檢查
        if df_daily is None or df_daily.empty or len(df_daily) < MIN_DAILY_DATA:
            logger.warning(f"{ticker_symbol}: 日線數據不足 ({len(df_daily) if df_daily is not None else 0}條)")
            return None

        if df_hourly is None or df_hourly.empty:
            df_hourly = pd.DataFrame()  # 允許小時線為空

        return {
            "df_daily": df_daily,
            "df_hourly": df_hourly
        }

    except Exception as e:
        logger.error(f"{ticker_symbol}: 歷史數據抓取失敗 - {str(e)}")
        return None


@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_current_price(ticker_symbol: str) -> Optional[Dict[str, Any]]:
    """抓取當前價格（獨立快取以提高實時性）"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        curr_price = None
        source = "失敗"

        # 嘗試從 info 獲取實時價格
        try:
            info = ticker.info or {}
            state = info.get("marketState", "").upper()

            if state == "PRE" and info.get("preMarketPrice"):
                curr_price = float(info["preMarketPrice"])
                source = "盤前"
            elif state == "POST" and info.get("postMarketPrice"):
                curr_price = float(info["postMarketPrice"])
                source = "盤後"
            elif info.get("regularMarketPrice"):
                curr_price = float(info["regularMarketPrice"])
                source = "常規"
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"{ticker_symbol}: info 價格獲取失敗 - {str(e)}")

        # 備用方案：從 1 分鐘 K 線獲取
        if curr_price is None:
            try:
                df_1m = ticker.history(period="1d", interval="1m", prepost=True)
                if df_1m is not None and not df_1m.empty:
                    close_series = df_1m['Close'].dropna()
                    if not close_series.empty:
                        curr_price = float(close_series.iloc[-1])
                        source = "1 分 K"
            except Exception as e:
                logger.debug(f"{ticker_symbol}: 1 分 K 價格獲取失敗 - {str(e)}")

        if curr_price is None:
            return None

        return {
            "price": curr_price,
            "source": source
        }

    except Exception as e:
        logger.error(f"{ticker_symbol}: 價格抓取失敗 - {str(e)}")
        return None


# ------------------------------------------------------------------------------
# 技術分析模組
# ------------------------------------------------------------------------------
def calculate_fib_levels(df_daily: pd.DataFrame, curr_price: float) -> Dict[str, float]:
    """計算 FIB 回撤位"""
    daily_recent = df_daily.tail(60)
    daily_high = max(daily_recent['High'].max(), curr_price)
    daily_low = min(daily_recent['Low'].min(), curr_price)

    price_range = daily_high - daily_low

    return {
        'fib_500': daily_high - FIB_LEVELS['500'] * price_range,
        'fib_618': daily_high - FIB_LEVELS['618'] * price_range,
        'fib_786': daily_high - FIB_LEVELS['786'] * price_range,
        'daily_high': daily_high,
        'daily_low': daily_low
    }


def check_hourly_signal(df_hourly: pd.DataFrame, curr_price: float, fib_786: float) -> tuple[bool, float]:
    """檢查小時線交易訊號"""
    if df_hourly is None or len(df_hourly) < MIN_HOURLY_DATA:
        return False, round(fib_786 * 0.98, 2)

    try:
        df_hourly = df_hourly.copy()

        # 計算技術指標
        df_hourly['EMA20'] = df_hourly['Close'].ewm(span=20, adjust=False).mean()
        df_hourly['Vol_MA'] = df_hourly['Volume'].rolling(20).mean()

        # 提取關鍵數據（增加空值檢查）
        if len(df_hourly) < 2:
            return False, round(fib_786 * 0.98, 2)

        c_prev = df_hourly['Close'].iloc[-2]
        ema_prev = df_hourly['EMA20'].iloc[-2]
        ema_curr = df_hourly['EMA20'].iloc[-1]
        vol_curr = df_hourly['Volume'].iloc[-1]
        vol_prev = df_hourly['Volume'].iloc[-2]
        vol_ma = df_hourly['Vol_MA'].iloc[-1]

        # 檢查 NaN 值
        if any(pd.isna([c_prev, ema_prev, ema_curr, vol_curr, vol_prev, vol_ma])):
            return False, round(fib_786 * 0.98, 2)

        # 突破條件：價格從 EMA 下方突破到上方
        h_breakout = (c_prev <= ema_prev) and (curr_price > ema_curr)

        # 成交量放大條件
        h_vol_spike = max(vol_curr, vol_prev) > (vol_ma * VOL_SPIKE_THRESHOLD)

        hourly_triggered = bool(h_breakout and h_vol_spike)

        # 動態止損計算
        hourly_low_min = df_hourly['Low'].tail(15).min()
        hourly_stop_loss = round(min(hourly_low_min, fib_786) * 0.99, 2)

        return hourly_triggered, hourly_stop_loss

    except Exception as e:
        logger.warning(f"小時線訊號計算失敗：{str(e)}")
        return False, round(fib_786 * 0.98, 2)


def determine_strategy(in_fib_zone: bool, hourly_triggered: bool,
                       curr_price: float, fib_618: float) -> Dict[str, str]:
    """根據條件決定交易策略"""
    if in_fib_zone and hourly_triggered:
        return {
            "strategy": "🚀 大主升浪起場點",
            "position": "70% - 100%",
            "color": "green"
        }
    elif in_fib_zone and (curr_price >= fib_618 * 0.99):
        return {
            "strategy": "📈 中線波段建倉",
            "position": "30% - 50%",
            "color": "blue"
        }
    elif hourly_triggered:
        return {
            "strategy": "⚡ 短線衝刺/反彈",
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
    """分析單支股票（完整流程）"""
    try:
        # 並行抓取歷史數據和當前價格
        historical_data = fetch_historical_data(ticker)
        price_data = fetch_current_price(ticker)

        # 數據完整性檢查
        if historical_data is None or price_data is None:
            logger.warning(f"{ticker}: 數據不完整，跳過分析")
            return None

        curr_price = price_data["price"]
        source = price_data["source"]
        df_daily = historical_data["df_daily"]
        df_hourly = historical_data["df_hourly"]

        # 計算 FIB 位
        fib_data = calculate_fib_levels(df_daily, curr_price)
        fib_500 = fib_data['fib_500']
        fib_618 = fib_data['fib_618']
        fib_786 = fib_data['fib_786']
        daily_high = fib_data['daily_high']

        # 判斷是否在 FIB 區間
        in_fib_zone = (curr_price >= fib_786 * 0.99) and (curr_price <= fib_500 * 1.01)

        # 檢查小時線訊號
        hourly_triggered, hourly_stop_loss = check_hourly_signal(df_hourly, curr_price, fib_786)

        # 確定策略
        strategy_info = determine_strategy(in_fib_zone, hourly_triggered, curr_price, fib_618)

        # 計算距離百分比
        dist_618 = round(((curr_price - fib_618) / fib_618) * 100, 2)

        return {
            "代碼": ticker,
            "現價": round(curr_price, 2),
            "來源": source,
            "建議策略": strategy_info["strategy"],
            "建議倉位": strategy_info["position"],
            "Fib 0.618": round(fib_618, 2),
            "距 0.618(%)": f"{dist_618}%",
            "止損價": hourly_stop_loss,
            "color": strategy_info["color"],
            "_d_h": daily_high,
            "_d_l": fib_data['daily_low'],
            "_f618": fib_618,
            "_f786": fib_786
        }

    except Exception as e:
        logger.error(f"{ticker}: 分析過程失敗 - {str(e)}")
        return None


def analyze_stocks_parallel(tickers: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
    """並行分析多支股票"""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        future_to_ticker = {
            executor.submit(analyze_single_stock, ticker): ticker
            for ticker in tickers
        }

        # 收集結果
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
# 壓力測試情景生成
# ------------------------------------------------------------------------------
def generate_scenarios(f618: float, f786: float, d_h: float) -> pd.DataFrame:
    """生成壓力測試情景"""
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
    return pd.DataFrame(scenarios)


# ------------------------------------------------------------------------------
# 主界面邏輯
# ------------------------------------------------------------------------------
def render_results(results: List[Dict[str, Any]]):
    """渲染分析結果"""
    if not results:
        st.error("無法取得相關股票數據，請檢查輸入代碼或網路連線。")
        return

    st.subheader("📊 實時盤口總覽")

    # 顯示簡潔表格
    df_display = pd.DataFrame(results)[
        ["代碼", "現價", "來源", "建議策略", "建議倉位", "Fib 0.618", "距 0.618(%)", "止損價"]
    ]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📱 手機卡片與壓力測試")

    # 卡片化展示
    for res in results:
        with st.expander(
            f"📌 **{res['代碼']}** - {res['建議策略']} (現價: ${res['現價']})",
            expanded=True
        ):
            # 三欄指標卡片
            m1, m2, m3 = st.columns(3)
            m1.metric("現價", f"${res['現價']}", delta=res['來源'])
            m2.metric("Fib 0.618", f"${res['Fib 0.618']}")
            m3.metric("建議止損", f"${res['止損價']}")

            st.caption(
                f"**建議倉位**：{res['建議倉位']}｜"
                f"**距離 0.618**：{res['距 0.618(%)']}"
            )

            # 壓力測試情景
            scenarios_df = generate_scenarios(
                res["_f618"],
                res["_f786"],
                res["_d_h"]
            )
            st.table(scenarios_df)


def main():
    """主函數"""
    # 輸入區域
    raw_input = st.text_input(
        "輸入股票代碼 (支援空格/逗號分隔):",
        value="BA, NVDA, TSLA",
        help="例如：AAPL, MSFT, GOOGL 或 AAPL MSFT GOOGL"
    )

    if st.button("🔍 開始掃描分析", use_container_width=True):
        # 解析並驗證代碼
        tickers = parse_tickers(raw_input)

        if not tickers:
            st.warning("請輸入有效的股票代碼！")
            return

        # 顯示進度提示
        with st.spinner(f"正即時連線抓取 {len(tickers)} 支股票數據..."):
            # 並行分析
            results = analyze_stocks_parallel(tickers, max_workers=min(5, len(tickers)))

        # 渲染結果
        render_results(results)


if __name__ == "__main__":
    main()
