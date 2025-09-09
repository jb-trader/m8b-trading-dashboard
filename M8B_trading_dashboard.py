#!/usr/bin/env python3
"""
Time Trends Dashboard (TTD)
Automatically updates when any setting changes
No optimization - direct analysis of historical data

Version 1.2.2
- Earnings E+1 handling: When "Exclude Earnings Days" is enabled, we now exclude
  BOTH the earnings date AND the following business day (E+1). FOMC exclusions
  remain unchanged (no +1).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import requests
from io import BytesIO
from pandas.tseries.offsets import BDay  # <-- for E+1 business day handling
import config as cfg

warnings.filterwarnings('ignore')

# Try to import optimizer module for data processing
try:
    from spx_butterfly_optimizer import TradingDataProcessor, Config
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    print("Warning: spx_butterfly_optimizer not found. Using sample data.")


# ============================================================================
# EXCLUSION CALENDAR HELPERS (E + E+1, FOMC unchanged)
# ============================================================================

def _to_date_index(dates_like):
    """Convert list-like of YYYY-MM-DD to normalized pandas DatetimeIndex."""
    if dates_like is None or len(dates_like) == 0:
        return pd.DatetimeIndex([])
    return pd.to_datetime(dates_like, errors="coerce").dropna().normalize()


def build_exclusion_index(include_fomc: bool = True,
                          include_earn: bool = True,
                          include_eplus1: bool = True):
    """
    Returns (fomc_index, earnings_union_index).
    - FOMC: exactly cfg.FOMC_DATES (no +1).
    - Earnings: cfg.EARNINGS_DATES; if include_eplus1=True, also adds next business day.
    """
    fomc_idx = _to_date_index(cfg.FOMC_DATES) if include_fomc else pd.DatetimeIndex([])
    earn_idx = _to_date_index(getattr(cfg, "get_earnings_dates", lambda: cfg.EARNINGS_DATES)())
    if not include_earn:
        return fomc_idx, pd.DatetimeIndex([])
    if include_eplus1 and len(earn_idx) > 0:
        eplus1 = (pd.Series(earn_idx) + BDay(1)).dt.normalize()
        earn_idx = earn_idx.union(pd.DatetimeIndex(eplus1))
    return fomc_idx, earn_idx


# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_sortino_ratio(returns, mar=0):
    """Calculate Sortino ratio (return/downside deviation)."""
    if len(returns) < 2:
        return np.nan
    mean_return = np.mean(returns)
    downside_returns = np.minimum(returns - mar, 0)
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    if downside_dev == 0:
        return np.nan
    return mean_return / downside_dev


def calculate_profit_factor(profits):
    """Calculate profit factor (sum of wins / sum of losses)."""
    wins = profits[profits > 0].sum()
    losses = abs(profits[profits < 0].sum())
    if losses == 0:
        return np.inf if wins > 0 else 0
    return wins / losses


def calculate_composite_score(metrics, weights):
    """Calculate weighted composite score on normalized scales."""
    score = 0
    total_weight = 0
    for metric, weight in weights.items():
        if metric in metrics and not pd.isna(metrics[metric]):
            if metric == 'sortino_ratio':
                normalized = (metrics[metric] + 2) / 5          # approx -2..3
            elif metric == 'avg_profit':
                normalized = (metrics[metric] + 1000) / 3000     # heuristic
            elif metric == 'win_rate':
                normalized = metrics[metric]                     # already 0..1
            elif metric == 'profit_factor':
                normalized = min(metrics[metric], 3) / 3         # cap at 3
            else:
                normalized = 0.5
            normalized = max(0, min(1, normalized))
            score += weight * normalized
            total_weight += weight
    if total_weight > 0:
        score = score / total_weight
    return score


def calculate_max_drawdown(cumulative_returns):
    """Calculate max drawdown from a cumulative returns series."""
    if len(cumulative_returns) < 2:
        return 0
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    return abs(drawdown.min())


def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """Per-trade Sharpe-like statistic (not annualized)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    sd = excess.std()
    if sd == 0:
        return 0.0
    return excess.mean() / sd


def get_score_color_style(score):
    """Return color style based on score value."""
    if score >= 0.70:
        return "background-color: #90EE90; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold;"
    elif score >= 0.50:
        return "background-color: #FFD700; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold;"
    else:
        return "background-color: #FF6B6B; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold;"


# ============================================================================
# DATA LOADING
# ============================================================================

def _download_drive_bytes() -> bytes:
    sess = requests.Session()
    url = cfg.get_data_url()  # direct download url from config
    r = sess.get(url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True, timeout=30)
    if "text/html" in r.headers.get("Content-Type", ""):
        import re as _re
        m = _re.search(r'href="([^"]+confirm[^"]+)"', r.text)
        if m:
            confirm_url = "https://drive.google.com" + m.group(1).replace("&amp;", "&")
            r = sess.get(confirm_url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.content


@st.cache_data(ttl=7200)
def load_historical_data(symbol: str, strategy: str, data_version: str):
    _ = data_version  # include version in cache key
    local_path = Path(r"D:/_Documents/Magic 8 Ball/data/dfe_table.parquet")
    df = pd.DataFrame()
    try:
        data_bytes = _download_drive_bytes()
        df = pd.read_parquet(BytesIO(data_bytes))
        source = "cloud"
    except Exception as e:
        st.warning(f"Cloud download issue ({e}). Using local file if available.")
        if local_path.exists():
            df = pd.read_parquet(local_path)
            source = "local"
        else:
            st.error("No local fallback file found.")
            return pd.DataFrame()

    df = df[(df['Symbol'] == symbol) & (df['Name'] == strategy)].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'].astype(str), errors='coerce').dt.time
    df = df.dropna(subset=['Date', 'Entry_Time'])

    latest_date = df['Date'].max()
    st.sidebar.success(f"✅ Data loaded from {source}")
    if pd.notna(latest_date):
        ver = getattr(cfg, "get_data_version", lambda: "NA")()
        st.sidebar.caption(f"Latest date: {latest_date:%m/%d/%Y} • v={ver}")
    return df


# ============================================================================
# FILTERING (WEEKS WINDOW) — now includes Earnings E+1 when exclude_earnings=True
# ============================================================================

def filter_data_by_weeks(df, weeks_back, exclude_fomc=True, exclude_earnings=True):
    """Filter for fixed weeks_back window and apply exclusions (E + E+1 if enabled)."""
    end_date = df['Date'].max()
    start_date = end_date - timedelta(weeks=weeks_back)
    filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    fomc_idx, earn_idx = build_exclusion_index(
        include_fomc=exclude_fomc,
        include_earn=exclude_earnings,
        include_eplus1=True  # <--- key: Earnings + next business day
    )

    if exclude_fomc and len(fomc_idx) > 0:
        filtered = filtered[~filtered['Date'].isin(fomc_idx)]
    if exclude_earnings and len(earn_idx) > 0:
        filtered = filtered[~filtered['Date'].isin(earn_idx)]

    cutoff = pd.to_datetime('15:30', format='%H:%M').time()
    filtered = filtered[filtered['Entry_Time'] <= cutoff]
    return filtered


# ============================================================================
# OPTIONAL: BACKFILL N DATES PER WEEKDAY (kept for future use; unchanged logic)
# ============================================================================

def select_recent_n_dates_per_weekday(df, n=20, exclude_fomc=True, exclude_earnings=True):
    """Pick most recent N valid dates per weekday (applies E+1 if exclude_earnings)."""
    fomc_idx, earn_idx = build_exclusion_index(
        include_fomc=exclude_fomc,
        include_earn=exclude_earnings,
        include_eplus1=True
    )
    cutoff = pd.to_datetime('15:30', format='%H:%M').time()

    base = df.copy()
    base = base[base['Entry_Time'] <= cutoff]
    if exclude_fomc and len(fomc_idx) > 0:
        base = base[~base['Date'].isin(fomc_idx)]
    if exclude_earnings and len(earn_idx) > 0:
        base = base[~base['Date'].isin(earn_idx)]

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    keep_dates = []

    for day in days_order:
        day_dates = (base.loc[base['Day_of_week'] == day, 'Date']
                        .drop_duplicates()
                        .sort_values(ascending=False))
        take = day_dates.head(n).tolist()
        keep_dates.extend(take)

    keep_idx = pd.DatetimeIndex(pd.unique(keep_dates))
    return base[base['Date'].isin(keep_idx)].copy()


# ============================================================================
# METRICS TABLES
# ============================================================================

def calculate_metrics_for_times(df, contracts=1):
    """Compute metrics for each Day_of_week × Entry_Time."""
    results = []
    df = df.copy()
    df['Profit'] = df['Profit'] * contracts
    for (dow, entry_time), group in df.groupby(['Day_of_week', 'Entry_Time']):
        if len(group) < 3:
            continue
        profits = group['Profit'].values
        metrics = {
            'Day_of_week': dow,
            'Entry_Time': entry_time,
            'trade_count': len(group),
            'sortino_ratio': calculate_sortino_ratio(profits),
            'avg_profit': np.mean(profits),
            'win_rate': np.mean(profits > 0),
            'profit_factor': calculate_profit_factor(profits)
        }
        results.append(metrics)
    return pd.DataFrame(results)


def get_top_times_by_day(metrics_df, weights):
    """Top-3 times per weekday by composite score."""
    if metrics_df.empty:
        return pd.DataFrame()
    metrics_df = metrics_df.copy()
    metrics_df['composite_score'] = metrics_df.apply(
        lambda row: calculate_composite_score(row.to_dict(), weights),
        axis=1
    )
    top_times = []
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in days_order:
        day_data = metrics_df[metrics_df['Day_of_week'] == day].copy()
        if len(day_data) > 0:
            day_data = day_data.nlargest(3, 'composite_score')
            day_data['rank'] = range(1, len(day_data) + 1)
            top_times.append(day_data)
    return pd.concat(top_times) if top_times else pd.DataFrame()


# ============================================================================
# OPTIMAL WEEKS (unchanged behavior)
# ============================================================================

@st.cache_data
def find_optimal_weeks(symbol, strategy, min_weeks=4, max_weeks=52,
                       exclude_fomc=True, exclude_earnings=True, contracts=1):
    if symbol == "SPX" and strategy == "Butterfly":
        return 20
    try:
        df = load_historical_data(symbol, strategy, data_version=cfg.get_data_version())
        if df.empty:
            return 30
        results = []
        for weeks in range(min_weeks, min(max_weeks + 1, 53)):
            filtered = filter_data_by_weeks(df, weeks, exclude_fomc, exclude_earnings)
            if len(filtered) > 0:
                avg_profit = filtered['Profit'].mean() * contracts
                total_profit = filtered['Profit'].sum() * contracts
                trade_count = len(filtered)
                results.append({
                    'weeks': weeks,
                    'avg_profit': avg_profit,
                    'total_profit': total_profit,
                    'trade_count': trade_count
                })
        if results:
            results_df = pd.DataFrame(results)
            optimal_weeks = results_df.loc[results_df['avg_profit'].idxmax(), 'weeks']
            return int(optimal_weeks)
        return 30
    except Exception as e:
        st.error(f"Error finding optimal weeks: {e}")
        return 30


# ============================================================================
# FORWARD TESTING — uses E + E+1 for earnings when exclusions enabled
# ============================================================================

def run_forward_test(df, training_weeks, trading_weeks, rank, day_filter,
                     active_weights, contracts, starting_balance,
                     exclude_fomc=True, exclude_earnings=True):
    """Walk-forward simulation with consistent exclusions in train and trade windows."""
    df = df.sort_values('Date').copy()

    # Build indices ONCE with E+1 for earnings
    fomc_idx, earn_idx = build_exclusion_index(
        include_fomc=exclude_fomc,
        include_earn=exclude_earnings,
        include_eplus1=True
    )

    min_date = df['Date'].min()
    max_date = df['Date'].max()

    equity_curve, trades_log, retraining_log = [], [], []
    current_balance = starting_balance

    current_start = min_date
    while current_start < max_date:
        # rolling windows
        training_end = current_start + timedelta(weeks=training_weeks)
        trading_start = training_end
        trading_end = min(trading_start + timedelta(weeks=trading_weeks), max_date)

        if trading_start >= max_date:
            break

        # --- TRAINING slice (current_start <= Date < training_end)
        training_data = df[(df['Date'] >= current_start) & (df['Date'] < training_end)].copy()
        if exclude_fomc and len(fomc_idx) > 0:
            training_data = training_data[~training_data['Date'].isin(fomc_idx)]
        if exclude_earnings and len(earn_idx) > 0:
            training_data = training_data[~training_data['Date'].isin(earn_idx)]

        if len(training_data) == 0:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        metrics_df = calculate_metrics_for_times(training_data, contracts)
        if metrics_df.empty:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        top_times_df = get_top_times_by_day(metrics_df, active_weights)
        if top_times_df.empty:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        selected_times = top_times_df[top_times_df['rank'] == rank][['Day_of_week', 'Entry_Time']].copy()
        retraining_log.append({
            'Training Start': current_start,
            'Training End': training_end,
            'Trading Start': trading_start,
            'Trading End': trading_end,
            'Selected Times': len(selected_times)
        })

        # --- TRADING slice (trading_start <= Date < trading_end)
        trading_data = df[(df['Date'] >= trading_start) & (df['Date'] < trading_end)].copy()
        if exclude_fomc and len(fomc_idx) > 0:
            trading_data = trading_data[~trading_data['Date'].isin(fomc_idx)]
        if exclude_earnings and len(earn_idx) > 0:
            trading_data = trading_data[~trading_data['Date'].isin(earn_idx)]
        if day_filter != "All Days":
            trading_data = trading_data[trading_data['Day_of_week'] == day_filter]

        if trading_data.empty or selected_times.empty:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        # keep only rows matching the selected (Day, Time) pairs
        key_trd = trading_data.assign(_key=trading_data['Day_of_week'].astype(str) + '|' + trading_data['Entry_Time'].astype(str))
        key_sel = selected_times.assign(_key=selected_times['Day_of_week'].astype(str) + '|' + selected_times['Entry_Time'].astype(str))['_key']
        trading_data = trading_data[key_trd['_key'].isin(set(key_sel))]

        if trading_data.empty:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        # execute trades
        for _, trade in trading_data.sort_values('Date').iterrows():
            profit = trade['Profit'] * contracts
            current_balance += profit
            equity_curve.append({'Date': trade['Date'], 'Balance': current_balance, 'Profit': profit})
            trades_log.append({'Date': trade['Date'], 'Day': trade['Day_of_week'], 'Time': trade['Entry_Time'], 'Profit': profit})

        # advance the walk-forward window
        current_start = current_start + timedelta(weeks=trading_weeks)

    equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    retraining_df = pd.DataFrame(retraining_log) if retraining_log else pd.DataFrame()
    return equity_df, trades_df, retraining_df



# ============================================================================
# DEBUGGING EXPANDER — shows E vs E+1 explicitly
# ============================================================================

def debug_date_filtering(df, weeks_back, exclude_fomc=True, exclude_earnings=True):
    """
    Show which dates are included/excluded for each weekday within weeks_back window.
    Earnings exclusions include E and E+1; we also show the raw E-only for clarity.
    """
    end_date = df['Date'].max()
    start_date = end_date - timedelta(weeks=weeks_back)
    date_range_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    fomc_idx, earn_e1_idx = build_exclusion_index(
        include_fomc=exclude_fomc,
        include_earn=exclude_earnings,
        include_eplus1=True
    )
    earn_only_idx = _to_date_index(getattr(cfg, "get_earnings_dates", lambda: cfg.EARNINGS_DATES)())

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    debug_info = {}
    cutoff = pd.to_datetime('15:30', format='%H:%M').time()

    for day in days_order:
        day_dates = date_range_df[date_range_df['Day_of_week'] == day]['Date'].drop_duplicates().sort_values()
        excluded_dates = {
            'fomc': [],
            'earnings': [],
            'earnings_plus1': [],
            'late_time': [],
            'total_available': int(day_dates.size),
            'all_dates': []
        }

        for d in day_dates:
            date_info = {'date': f"{d:%Y-%m-%d}", 'excluded_by': []}
            if exclude_fomc and d in fomc_idx:
                excluded_dates['fomc'].append(f"{d:%Y-%m-%d}")
                date_info['excluded_by'].append('FOMC')
            if exclude_earnings:
                if d in earn_only_idx:
                    excluded_dates['earnings'].append(f"{d:%Y-%m-%d}")
                    date_info['excluded_by'].append('Earnings(E)')
                elif d in earn_e1_idx:
                    excluded_dates['earnings_plus1'].append(f"{d:%Y-%m-%d}")
                    date_info['excluded_by'].append('Earnings(E+1)')

            day_data = date_range_df[date_range_df['Date'] == d]
            if not day_data.empty:
                late_times = day_data[day_data['Entry_Time'] > cutoff]
                if len(late_times) == len(day_data):
                    excluded_dates['late_time'].append(f"{d:%Y-%m-%d}")
                    date_info['excluded_by'].append('Late Time')
            excluded_dates['all_dates'].append(date_info)

        final_df = date_range_df[
            (date_range_df['Day_of_week'] == day) &
            (~date_range_df['Date'].isin(fomc_idx)) &
            (~date_range_df['Date'].isin(earn_e1_idx)) &
            (date_range_df['Entry_Time'] <= cutoff)
        ]
        final_dates = final_df['Date'].drop_duplicates().sort_values()

        excluded_dates['final_count'] = int(final_dates.size)
        excluded_dates['dates_included'] = [f"{d:%Y-%m-%d}" for d in final_dates]
        debug_info[day] = excluded_dates

    return debug_info


def add_debug_section(df, weeks_history, exclude_fomc, exclude_earnings):
    with st.expander("🔍 Debug: Date Filtering Details", expanded=False):
        st.write("### Date Range Analysis")
        debug_data = debug_date_filtering(df, weeks_history, exclude_fomc, exclude_earnings)

        summary_data = []
        for day, info in debug_data.items():
            summary_data.append({
                'Day': day,
                'Total in Range': info['total_available'],
                'FOMC Excluded': len(info['fomc']),
                'Earnings (E) Excluded': len(info['earnings']),
                'Earnings (E+1) Excluded': len(info['earnings_plus1']),
                'Late Time Excluded': len(info['late_time']),
                'Final Count': info['final_count']
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        selected_day = st.selectbox("View details for:", list(debug_data.keys()))
        if selected_day:
            day_info = debug_data[selected_day]
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Excluded Dates:**")
                if day_info['fomc']:
                    st.write(f"FOMC: {', '.join(day_info['fomc'])}")
                if day_info['earnings']:
                    st.write(f"Earnings (E): {', '.join(day_info['earnings'])}")
                if day_info['earnings_plus1']:
                    st.write(f"Earnings (E+1): {', '.join(day_info['earnings_plus1'])}")
                if day_info['late_time']:
                    st.write(f"Late Time: {', '.join(day_info['late_time'])}")
            with col2:
                st.write("**Included Dates:**")
                included = day_info['dates_included']
                if len(included) > 10:
                    st.write(f"First 5: {', '.join(included[:5])}")
                    st.write(f"Last 5: {', '.join(included[-5:])}")
                    st.write(f"Total: {len(included)} dates")
                else:
                    st.write(', '.join(included))

        st.write("### Date Range Info")
        end_date = df['Date'].max()
        start_date = end_date - timedelta(weeks=weeks_history)
        st.write(f"- Start: {start_date:%Y-%m-%d}")
        st.write(f"- End: {end_date:%Y-%m-%d}")
        st.write(f"- Weeks: {weeks_history}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Time Trends Dashboard (TTD)",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    [data-testid="stSidebar"] .stSlider > div > div {
        background: linear-gradient(to right, #ff4444 0%, #ffff00 50%, #44ff44 100%);
    }
    .metric-value { font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.subheader("📊 Strategy")
        symbol = st.selectbox("Symbol", ["SPX", "XSP", "RUT", "NDX"], key="symbol")
        strategy = st.selectbox("Strategy Type", ["Butterfly", "Iron Condor", "Vertical", "Sonar"], key="strategy")

        st.subheader("📅 Data Range")

        if 'optimal_weeks' not in st.session_state or \
           st.session_state.get('last_symbol') != symbol or \
           st.session_state.get('last_strategy') != strategy:
            with st.spinner("Finding optimal period..."):
                optimal_weeks = find_optimal_weeks(
                    symbol, strategy,
                    exclude_fomc=st.session_state.get('exclude_fomc', True),
                    exclude_earnings=st.session_state.get('exclude_earnings', True),
                    contracts=st.session_state.get('contracts', 1)
                )
                st.session_state.optimal_weeks = optimal_weeks
                st.session_state.last_symbol = symbol
                st.session_state.last_strategy = strategy

        if 'reset_to_optimal' not in st.session_state:
            st.session_state.reset_to_optimal = False

        slider_value = st.session_state.optimal_weeks if st.session_state.reset_to_optimal \
            else st.session_state.get('weeks_history', st.session_state.optimal_weeks)

        col1, col2 = st.columns([3, 1])
        with col1:
            weeks_history = st.slider(
                "Weeks of History", min_value=4, max_value=52,
                value=slider_value, step=1,
                help=f"Auto-optimized to {st.session_state.optimal_weeks} weeks for best avg profit/trade",
                key="weeks_history"
            )
        with col2:
            if st.button("🎯", help="Reset to optimal weeks"):
                st.session_state.reset_to_optimal = True
                st.rerun()

        if weeks_history == st.session_state.optimal_weeks:
            st.success(f"✨ Using optimal period for Avg Profit ({weeks_history} weeks)")
        else:
            st.info(f"📊 Custom period (Optimal Avg Profit: {st.session_state.optimal_weeks} weeks)")

        st.caption(f"Analyzing {weeks_history} weeks of data")

        st.subheader("⚖️ Metric Weights")
        if 'metric_weights' not in st.session_state:
            st.session_state.metric_weights = {
                'sortino_ratio': {'enabled': True, 'weight': 30},
                'avg_profit': {'enabled': True, 'weight': 30},
                'win_rate': {'enabled': True, 'weight': 20},
                'profit_factor': {'enabled': True, 'weight': 20}
            }
        metrics_config = {
            'sortino_ratio': {'label': '📊 Sortino Ratio', 'desc': 'Risk-adjusted returns'},
            'avg_profit': {'label': '💰 Average Profit', 'desc': 'Profit per trade'},
            'win_rate': {'label': '🎯 Win Rate', 'desc': 'Win frequency'},
            'profit_factor': {'label': '📈 Profit Factor', 'desc': 'Win/Loss ratio'}
        }
        total_weight = 0
        active_weights = {}
        for metric, config_item in metrics_config.items():
            c1, c2 = st.columns([1, 3])
            with c1:
                enabled = st.checkbox(
                    config_item['label'],
                    value=st.session_state.metric_weights[metric]['enabled'],
                    key=f"check_{metric}"
                )
                st.session_state.metric_weights[metric]['enabled'] = enabled
            with c2:
                if enabled:
                    w = st.slider(config_item['desc'], 0, 100,
                                  value=st.session_state.metric_weights[metric]['weight'],
                                  key=f"weight_{metric}")
                    st.session_state.metric_weights[metric]['weight'] = w
                    active_weights[metric] = w
                    total_weight += w
        if total_weight > 0:
            if total_weight == 100:
                st.success("✅ Weights = 100%")
            else:
                st.warning(f"⚠️ Weights = {total_weight}%")
        else:
            st.error("❌ No weights set (0%)")



        st.subheader("🔧 Data Filters")
        exclude_fomc = st.checkbox("Exclude FOMC Days", value=True, key="exclude_fomc")
        exclude_earnings = st.checkbox("Exclude Earnings Days (includes E+1)", value=True, key="exclude_earnings")

        st.subheader("💼 Trade Settings")
        contracts = st.number_input("Number of Contracts", 1, 100, value=1, key="contracts")
        starting_balance = st.number_input("Starting Account Balance ($)", 1000, 1000000, value=10000, step=1000, format="%d", key="starting_balance")

        st.markdown("---")
        if st.button("🔄 Force refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("optimal_weeks", None)
            st.rerun()

    # MAIN CONTENT
    with st.spinner("Loading data..."):
        df = load_historical_data(symbol, strategy, data_version=cfg.get_data_version())
        if df.empty:
            st.error("No data available for selected symbol and strategy")
            return

        latest_data_date = df['Date'].max().strftime('%m/%d/%Y')

        filtered_df = filter_data_by_weeks(df, weeks_history, exclude_fomc, exclude_earnings)

        st.markdown(f"""
        <h1 style='text-align:left;margin-bottom:0'>
            📊 Time Trends Dashboard (TTD)
            <span style='font-size:0.5em;color:#1E90FF;font-weight:400;'>by jb-trader</span>
            <span style='font-size:0.4em;color:#888;font-weight:400;margin-left:20px;'>
                Version 1.2.2 | Source: M8B v1.37 | Data updated: {latest_data_date}
            </span>
        </h1>
        """, unsafe_allow_html=True)
        st.markdown(f"Analyzing {weeks_history} weeks of historical data | Live updates enabled")

        metrics_df = calculate_metrics_for_times(filtered_df, contracts)
        add_debug_section(df, weeks_history, exclude_fomc, exclude_earnings)

        active_weights = {k: v['weight'] for k, v in st.session_state.metric_weights.items() if v['enabled']}
        top_times_df = get_top_times_by_day(metrics_df, active_weights)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Best Times", "📈 Performance", "🔄 Forward Testing", "📊 Analysis", "📋 All Times", "✅ Validation"
    ])

    # TAB 1: BEST TIMES
    with tab1:
        st.subheader("🎯 Top 3 Trading Times by Day")
        if not top_times_df.empty:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            today_et = datetime.now(et_tz)
            if (today_et.weekday() == 4 and today_et.hour >= 16) or today_et.weekday() in [5, 6]:
                monday = today_et - timedelta(days=today_et.weekday()) + timedelta(days=7)
            else:
                monday = today_et - timedelta(days=today_et.weekday())
            monday = monday.replace(tzinfo=None)

            cols = st.columns([1, 2, 2, 2, 2, 2])
            cols[0].markdown("**Rank**")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for i, day in enumerate(days):
                date = monday + timedelta(days=i)
                cols[i+1].markdown(f"**{day}**")
                cols[i+1].caption(f"{date:%m/%d/%Y}")

            for rank in range(1, 4):
                cols = st.columns([1, 2, 2, 2, 2, 2])
                cols[0].markdown("**1st** 🥇" if rank == 1 else "**2nd** 🥈" if rank == 2 else "**3rd** 🥉")
                for i, day in enumerate(days):
                    day_data = top_times_df[(top_times_df['Day_of_week'] == day) & (top_times_df['rank'] == rank)]
                    if not day_data.empty:
                        row = day_data.iloc[0]
                        time_str = row['Entry_Time'].strftime('%H:%M')
                        score = row['composite_score']
                        cols[i+1].markdown(f"**{time_str}**")
                        cols[i+1].markdown(f'<span style="{get_score_color_style(score)}">Score: {score:.3f}</span>', unsafe_allow_html=True)
                    else:
                        cols[i+1].write("-")

            with st.expander("📊 Detailed Metrics for Top Times"):
                display_df = top_times_df[['Day_of_week', 'rank', 'Entry_Time',
                                           'composite_score', 'sortino_ratio',
                                           'avg_profit', 'win_rate', 'profit_factor',
                                           'trade_count']].copy()
                display_df.columns = ['Day', 'Rank', 'Time', 'Score', 'Sortino',
                                      'Avg Profit', 'Win Rate', 'Profit Factor', 'Trades']
                display_df['Win Rate'] = (display_df['Win Rate'] * 100).round(1).astype(str) + '%'
                display_df['Avg Profit'] = '$' + display_df['Avg Profit'].round(0).astype(str)
                display_df['Score'] = display_df['Score'].round(3)
                display_df['Sortino'] = display_df['Sortino'].round(2)
                display_df['Profit Factor'] = display_df['Profit Factor'].round(2)
                def style_scores(df_to_style):
                    def color_score(val):
                        if isinstance(val, (int, float)):
                            if val >= 0.70:
                                return 'background-color: #90EE90; color: black; font-weight: bold'
                            elif val >= 0.50:
                                return 'background-color: #FFD700; color: black; font-weight: bold'
                            else:
                                return 'background-color: #FF6B6B; color: black; font-weight: bold'
                        return ''
                    return df_to_style.style.map(color_score, subset=['Score'])
                styled_df = style_scores(display_df)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough data to display top times")

    # TAB 2: PERFORMANCE
    with tab2:
        st.subheader("📈 Historical Performance - Maybe This Trend Will Continue")

        if not filtered_df.empty and not top_times_df.empty:
            c1, c2, c3, c4, c5 = st.columns([2, 2, 3, 1, 1])
            with c1:
                selected_rank = st.selectbox("Select Rank", options=[1, 2, 3],
                                             format_func=lambda x: f"{'1st' if x==1 else '2nd' if x==2 else '3rd'} Rank Times",
                                             key="perf_rank_selector")
            with c2:
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                selected_day = st.selectbox("Day of Week", options=["All Days"] + days_order, key="perf_day_selector")
            with c3:
                perf_weeks = st.slider("Date Range (weeks)", 4, 52, value=weeks_history, key="perf_weeks_slider")
            with c4:
                perf_exclude_fomc = st.checkbox("Exclude FOMC", value=exclude_fomc, key="perf_exclude_fomc")
            with c5:
                perf_exclude_earnings = st.checkbox("Exclude Earnings (E+1)", value=exclude_earnings, key="perf_exclude_earnings")

            perf_filtered_df = filter_data_by_weeks(df, perf_weeks, perf_exclude_fomc, perf_exclude_earnings)
            if perf_weeks != weeks_history or perf_exclude_fomc != exclude_fomc or perf_exclude_earnings != exclude_earnings:
                perf_metrics_df = calculate_metrics_for_times(perf_filtered_df, contracts)
                perf_top_times_df = get_top_times_by_day(perf_metrics_df, active_weights)
            else:
                perf_top_times_df = top_times_df
                perf_filtered_df = filtered_df

            rank_times = perf_top_times_df[perf_top_times_df['rank'] == selected_rank][['Day_of_week', 'Entry_Time']]
            if selected_day != "All Days":
                rank_times = rank_times[rank_times['Day_of_week'] == selected_day]

            rank_filtered_df = perf_filtered_df.merge(rank_times, on=['Day_of_week', 'Entry_Time'], how='inner').copy()
            if selected_day != "All Days":
                rank_filtered_df = rank_filtered_df[rank_filtered_df['Day_of_week'] == selected_day].copy()

            if not rank_filtered_df.empty:
                rank_filtered_df = rank_filtered_df.sort_values('Date')
                rank_filtered_df['Cumulative_PL'] = (rank_filtered_df['Profit'] * contracts).cumsum()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rank_filtered_df['Date'],
                    y=starting_balance + rank_filtered_df['Cumulative_PL'],
                    mode='lines',
                    name=f'Equity (Rank {selected_rank}, {selected_day})',
                    line=dict(color='green', width=4)
                ))
                fig.add_hline(y=starting_balance, line_dash="dash", line_color="gray", annotation_text="Starting Balance")
                fig.update_layout(
                    title=f"Equity Curve - {'1st' if selected_rank==1 else '2nd' if selected_rank==2 else '3rd'} Rank Times - {selected_day} ({perf_weeks} weeks)",
                    xaxis_title="Date",
                    yaxis_title="Account Value ($)",
                    yaxis_tickformat='$,.0f',
                    height=700,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                total_trades = len(rank_filtered_df)
                total_profit = rank_filtered_df['Profit'].sum() * contracts
                win_rate = (rank_filtered_df['Profit'] > 0).mean() * 100
                avg_profit = rank_filtered_df['Profit'].mean() * contracts
                c1.metric("Total Trades", f"{total_trades:,}")
                c2.metric("Total Profit", f"${total_profit:,.0f}")
                c3.metric("Win Rate", f"{win_rate:.1f}%")
                c4.metric("Avg Profit/Trade", f"${avg_profit:.0f}")

                c1, c2, c3, c4 = st.columns(4)
                max_drawdown = calculate_max_drawdown(rank_filtered_df['Cumulative_PL'])
                sharpe = calculate_sharpe_ratio(rank_filtered_df['Profit'] * contracts)
                best_trade = (rank_filtered_df['Profit'] * contracts).max()
                worst_trade = (rank_filtered_df['Profit'] * contracts).min()
                c1.metric("Max Drawdown", f"${max_drawdown:,.0f}")
                c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                c3.metric("Best Trade", f"${best_trade:,.0f}")
                c4.metric("Worst Trade", f"${worst_trade:,.0f}")

                with st.expander(f"📅 Trading Times Used (Rank {selected_rank}, {selected_day})"):
                    times_display = rank_times.copy()
                    if selected_day != "All Days":
                        times_display = times_display[times_display['Day_of_week'] == selected_day]
                    times_display['Entry_Time'] = times_display['Entry_Time'].apply(lambda x: x.strftime('%H:%M'))
                    times_display = times_display.sort_values(['Day_of_week', 'Entry_Time'])
                    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                    times_by_day = []
                    for day in days_order:
                        day_time = times_display[times_display['Day_of_week'] == day]
                        if not day_time.empty:
                            times_by_day.append(f"**{day}**: {day_time['Entry_Time'].iloc[0]}")
                    if times_by_day:
                        st.markdown(" | ".join(times_by_day))
                    else:
                        st.markdown(f"**{selected_day}**: {times_display['Entry_Time'].iloc[0] if not times_display.empty else 'N/A'}")
            else:
                st.warning(f"No trades found for Rank {selected_rank} times on {selected_day}")
        else:
            st.warning("Not enough data to display performance metrics")

    # TAB 3: FORWARD TESTING
    with tab3:
        st.subheader("🔄 Forward Testing - Realistic Walk-Forward Analysis")

        col1, col2, col3, col4 = st.columns([1.5, 1.5, 2, 2])
        with col1:
            training_weeks = st.slider("Training Window", 4, 26, value=8, key="fwd_training_weeks")
        with col2:
            trading_weeks = st.slider("Trading Window", 1, 8, value=2, key="fwd_trading_weeks")
        with col3:
            fwd_rank = st.selectbox("Select Rank", [1, 2, 3], format_func=lambda x: f"{'1st' if x==1 else '2nd' if x==2 else '3rd'} Rank", key="fwd_rank_selector")
        with col4:
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            fwd_day = st.selectbox("Day of Week", ["All Days"] + days_order, key="fwd_day_selector")

        col1, col2, col3 = st.columns([3, 1.5, 1.5])
        with col1:
            only_after_aug_upgrade = st.checkbox("Only After Aug 2024 1.37 M8B upgrade", value=False, key="fwd_only_after_aug_2024")
        with col2:
            fwd_exclude_fomc = st.checkbox("Exclude FOMC", value=exclude_fomc, key="fwd_exclude_fomc")
        with col3:
            fwd_exclude_earnings = st.checkbox("Exclude Earnings (E+1)", value=exclude_earnings, key="fwd_exclude_earnings")

        with st.spinner("Running forward test simulation..."):
            df_for_fwd = df.copy()
            if only_after_aug_upgrade:
                cutoff = pd.Timestamp("2024-08-01")
                df_for_fwd = df_for_fwd[df_for_fwd["Date"] >= cutoff]

            equity_df, trades_df, retraining_df = run_forward_test(
                df_for_fwd, training_weeks, trading_weeks, fwd_rank, fwd_day,
                active_weights, contracts, starting_balance,
                fwd_exclude_fomc, fwd_exclude_earnings
            )

            if not equity_df.empty:
                # Hindsight comparison uses same E+1 rule for earnings
                fomc_idx, earn_idx = build_exclusion_index(
                    include_fomc=fwd_exclude_fomc,
                    include_earn=fwd_exclude_earnings,
                    include_eplus1=True
                )
                # 🔍 DEBUG: show excluded counts + sample dates
                print("Excluded earnings count:", len(earn_idx))
                if len(earn_idx) > 0:
                    print("First few excluded earnings dates:",
                        [d.strftime("%Y-%m-%d") for d in sorted(list(earn_idx))[:5]])
                hindsight_filtered = df_for_fwd.copy()
                if fwd_exclude_fomc and len(fomc_idx) > 0:
                    hindsight_filtered = hindsight_filtered[~hindsight_filtered["Date"].isin(fomc_idx)]
                if fwd_exclude_earnings and len(earn_idx) > 0:
                    hindsight_filtered = hindsight_filtered[~hindsight_filtered["Date"].isin(earn_idx)]

                hindsight_metrics = calculate_metrics_for_times(hindsight_filtered, contracts)
                hindsight_top_times = get_top_times_by_day(hindsight_metrics, active_weights)
                hindsight_times = hindsight_top_times[hindsight_top_times["rank"] == fwd_rank][["Day_of_week", "Entry_Time"]]

                hindsight_trades = hindsight_filtered.merge(hindsight_times, on=["Day_of_week", "Entry_Time"], how="inner")
                if fwd_day != "All Days":
                    hindsight_trades = hindsight_trades[hindsight_trades["Day_of_week"] == fwd_day]
                hindsight_trades = hindsight_trades.sort_values("Date")
                hindsight_trades["Cumulative_PL"] = (hindsight_trades["Profit"] * contracts).cumsum()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df["Date"], y=equity_df["Balance"],
                    mode="lines", name="Forward Test (Realistic)", line=dict(color="blue", width=3)
                ))
                fig.add_hline(y=starting_balance, line_dash="dash", line_color="red", annotation_text="Starting Balance")
                fig.update_layout(
                    title=f"Forward Test Results - Training: {training_weeks}w, Trading: {trading_weeks}w",
                    xaxis_title="Date", yaxis_title="Account Value ($)", yaxis_tickformat="$,.0f",
                    height=600, hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width= True)

                if not trades_df.empty:
                    st.subheader("Trade Analysis")
                    fwd_final_balance = equity_df["Balance"].iloc[-1]
                    fwd_total_return = fwd_final_balance - starting_balance
                    fwd_return_pct = (fwd_total_return / starting_balance) * 100
                    avg_profit = trades_df["Profit"].mean()
                    win_rate = (trades_df["Profit"] > 0).mean() * 100
                    profit_factor = calculate_profit_factor(trades_df["Profit"])
                    max_dd = calculate_max_drawdown(equity_df["Balance"] - starting_balance)
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("Avg Profit", f"${avg_profit:.0f}")
                    c2.metric("Win Rate", f"{win_rate:.1f}%")
                    c3.metric("Profit Factor", f"{profit_factor:.2f}")
                    c4.metric("Forward Test Return", f"${fwd_total_return:,.0f}", f"{fwd_return_pct:.1f}%")
                    c5.metric("Max Drawdown", f"${max_dd:,.0f}")
                    c6.metric("Total Trades", len(trades_df))

                st.markdown("### 📍 Current Trading Times")
                most_recent_date = df_for_fwd['Date'].max()

                training_start = most_recent_date - timedelta(weeks=training_weeks)
                recent_training = df_for_fwd[(df_for_fwd["Date"] >= training_start) & (df_for_fwd["Date"] <= most_recent_date)].copy()


                if fwd_exclude_fomc and len(fomc_idx) > 0:
                    recent_training = recent_training[~recent_training['Date'].isin(fomc_idx)]
                if fwd_exclude_earnings and len(earn_idx) > 0:
                    recent_training = recent_training[~recent_training['Date'].isin(earn_idx)]
                recent_metrics = calculate_metrics_for_times(recent_training, contracts)
                recent_top_times = get_top_times_by_day(recent_metrics, active_weights)
                current_times = recent_top_times[recent_top_times['rank'] == fwd_rank].copy()
                if fwd_day != "All Days":
                    current_times = current_times[current_times['Day_of_week'] == fwd_day]
                if not current_times.empty:
                    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                    cols = st.columns(5)
                    for i, day in enumerate(days_order):
                        day_time = current_times[current_times['Day_of_week'] == day]
                        with cols[i]:
                            st.markdown(f"### {day}")
                            if not day_time.empty:
                                time_str = day_time.iloc[0]['Entry_Time'].strftime('%H:%M')
                                score = day_time.iloc[0]['composite_score']
                                st.markdown(f"### 🕐 {time_str}")
                                st.markdown(f"**Score: {score:.3f}**")
                            else:
                                st.markdown("—")
                    st.caption(f"*Based on last {training_weeks} weeks of data through {most_recent_date:%m/%d/%Y}*")
                else:
                    st.warning("No trading times found for current parameters")
            else:
                st.warning("Not enough trades to plot an equity curve for the selected parameters.")

    # TAB 4: ANALYSIS
    with tab4:
        st.subheader("📊 Trading Analysis")
        if not metrics_df.empty:
            st.markdown("### Entry Time Performance Heatmap")
            metrics_df_display = metrics_df.copy()
            metrics_df_display["composite_score"] = metrics_df_display.apply(
                lambda row: calculate_composite_score(row.to_dict(), active_weights), axis=1
            )
            metrics_df_display["Entry_Time"] = metrics_df_display["Entry_Time"].apply(lambda x: x.strftime("%H:%M"))
            heatmap_data = metrics_df_display.pivot_table(
                index="Entry_Time", columns="Day_of_week", values="composite_score", aggfunc="mean"
            )
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns])
            if not heatmap_data.empty:
                heatmap_data = heatmap_data.sort_index(key=lambda s: pd.to_datetime(s, format="%H:%M"))
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Day of Week", y="Entry Time", color="Score"),
                    color_continuous_scale="RdYlGn", aspect="auto"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No heatmap data available with the current filters.")
        else:
            st.info("No metrics available to plot yet.")

    # TAB 5: ALL TIMES
    with tab5:
        st.subheader("📋 All Entry Times Ranked")
        if not metrics_df.empty:
            all_times = metrics_df.copy()
            all_times['composite_score'] = all_times.apply(
                lambda row: calculate_composite_score(row.to_dict(), active_weights), axis=1
            )
            all_times = all_times.sort_values('composite_score', ascending=False)
            display_all = all_times[['Day_of_week', 'Entry_Time', 'composite_score',
                                     'sortino_ratio', 'avg_profit', 'win_rate',
                                     'profit_factor', 'trade_count']].copy()
            display_all.columns = ['Day', 'Time', 'Score', 'Sortino',
                                   'Avg Profit', 'Win Rate', 'Profit Factor', 'Trades']
            display_all['Time'] = display_all['Time'].apply(lambda x: x.strftime('%H:%M'))
            display_all['Score'] = display_all['Score'].round(3)
            display_all['Sortino'] = display_all['Sortino'].round(2)
            display_all['Avg Profit'] = display_all['Avg Profit'].round(0)
            display_all['Win Rate'] = (display_all['Win Rate'] * 100).round(1)
            display_all['Profit Factor'] = display_all['Profit Factor'].round(2)
            display_all.insert(0, 'Rank', range(1, len(display_all) + 1))
            st.dataframe(display_all.head(50), use_container_width =True, hide_index=True)
            st.caption(f"Showing top 50 of {len(display_all)} total entry times")

    # TAB 6: VALIDATION (unchanged wiring, module optional)
    with tab6:
        st.subheader("✅ Statistical Validation")
        st.info("""
        This tab runs optional validation modules if available:
        1) Walk-Forward Analysis
        2) Monte Carlo Significance
        3) Pattern Stability
        """)
        if st.button("Run Complete Validation", type="primary"):
            processing_container = st.empty()
            processing_container.warning("🔄 VALIDATION IN PROGRESS...")
            try:
                from robust_validation_module import WalkForwardValidator, MonteCarloTester, PatternStabilityAnalyzer
                processing_container.info("Validation modules loaded. Run your routines here as before.")
            except ImportError:
                processing_container.empty()
                st.error("robust_validation_module.py not found. Install or place it in the same folder and retry.")

    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray; font-size: 12px;'>
        Last updated: {datetime.now():%H:%M:%S} |
        Data range: {weeks_history} weeks |
        Active metrics: {len(active_weights)}
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
