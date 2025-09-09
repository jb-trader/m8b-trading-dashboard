#!/usr/bin/env python3
"""
Real-Time Trading Dashboard
Automatically updates when any setting changes
No optimization - direct analysis of historical data
Version 1.0.1 - Added Forward Testing capability
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
import config  as cfg

warnings.filterwarnings('ignore')

# Try to import optimizer module for data processing
try:
    from spx_butterfly_optimizer import TradingDataProcessor, Config
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    print("Warning: spx_butterfly_optimizer not found. Using sample data.")
def debug_date_filtering(df, weeks_back, exclude_fomc=True, exclude_earnings=True):
    """
    Debug function to show exactly which dates are included/excluded for each day of week
    """
    import pandas as pd
    from datetime import timedelta
    import config as cfg
    
    # Get date range
    end_date = df['Date'].max()
    start_date = end_date - timedelta(weeks=weeks_back)
    
    # Get all dates in range
    date_range_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    # Convert exclusion dates to datetime
    fomc_dates = pd.to_datetime(cfg.FOMC_DATES) if exclude_fomc else pd.DatetimeIndex([])
    earnings_dates = pd.to_datetime(cfg.EARNINGS_DATES) if exclude_earnings else pd.DatetimeIndex([])
    
    # Days of week
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    debug_info = {}
    
    for day in days_order:
        # Get all dates for this day of week
        day_dates = date_range_df[date_range_df['Day_of_week'] == day]['Date'].unique()
        day_dates = pd.Series(day_dates).sort_values()
        
        excluded_dates = {
            'fomc': [],
            'earnings': [],
            'late_time': [],
            'total_available': len(day_dates),
            'all_dates': []
        }
        
        # Track each date
        for date in day_dates:
            date_info = {'date': date.strftime('%Y-%m-%d'), 'excluded_by': []}
            
            # Check exclusions
            if exclude_fomc and date in fomc_dates:
                excluded_dates['fomc'].append(date.strftime('%Y-%m-%d'))
                date_info['excluded_by'].append('FOMC')
            
            if exclude_earnings and date in earnings_dates:
                excluded_dates['earnings'].append(date.strftime('%Y-%m-%d'))
                date_info['excluded_by'].append('Earnings')
            
            # Check for late times (after 15:30)
            day_data = date_range_df[date_range_df['Date'] == date]
            if not day_data.empty:
                # Check if any entries are after 15:30
                late_times = day_data[day_data['Entry_Time'] > pd.to_datetime('15:30', format='%H:%M').time()]
                if len(late_times) == len(day_data):  # All entries are late
                    excluded_dates['late_time'].append(date.strftime('%Y-%m-%d'))
                    date_info['excluded_by'].append('Late Time')
            
            excluded_dates['all_dates'].append(date_info)
        
        # Calculate final count
        final_dates = day_dates.copy()
        if exclude_fomc:
            final_dates = final_dates[~final_dates.isin(fomc_dates)]
        if exclude_earnings:
            final_dates = final_dates[~final_dates.isin(earnings_dates)]
        
        # Also need to filter by time
        final_df = date_range_df[
            (date_range_df['Day_of_week'] == day) & 
            (date_range_df['Date'].isin(final_dates)) &
            (date_range_df['Entry_Time'] <= pd.to_datetime('15:30', format='%H:%M').time())
        ]
        
        excluded_dates['final_count'] = len(final_df['Date'].unique())
        excluded_dates['dates_included'] = sorted([d.strftime('%Y-%m-%d') for d in final_df['Date'].unique()])
        
        debug_info[day] = excluded_dates
    
    return debug_info

# Add this to your main dashboard in a debug expander
def add_debug_section(df, weeks_history, exclude_fomc, exclude_earnings):
    """
    Add this to your main() function after loading data
    """
    with st.expander("🔍 Debug: Date Filtering Details", expanded=False):
        st.write("### Date Range Analysis")
        
        debug_data = debug_date_filtering(df, weeks_history, exclude_fomc, exclude_earnings)
        
        # Summary table
        summary_data = []
        for day, info in debug_data.items():
            summary_data.append({
                'Day': day,
                'Total in Range': info['total_available'],
                'FOMC Excluded': len(info['fomc']),
                'Earnings Excluded': len(info['earnings']),
                'Late Time Excluded': len(info['late_time']),
                'Final Count': info['final_count']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Detailed view for each day
        selected_day = st.selectbox("View details for:", list(debug_data.keys()))
        
        if selected_day:
            day_info = debug_data[selected_day]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Excluded Dates:**")
                if day_info['fomc']:
                    st.write(f"FOMC: {', '.join(day_info['fomc'])}")
                if day_info['earnings']:
                    st.write(f"Earnings: {', '.join(day_info['earnings'])}")
                if day_info['late_time']:
                    st.write(f"Late Time: {', '.join(day_info['late_time'])}")
            
            with col2:
                st.write("**Included Dates:**")
                # Show first and last 5 dates
                included = day_info['dates_included']
                if len(included) > 10:
                    st.write(f"First 5: {', '.join(included[:5])}")
                    st.write(f"Last 5: {', '.join(included[-5:])}")
                    st.write(f"Total: {len(included)} dates")
                else:
                    st.write(', '.join(included))
        
        # Check for potential issues
        st.write("### Potential Issues Detected:")
        issues = []
        
        for day, info in debug_data.items():
            expected_weeks = weeks_history
            expected_dates = expected_weeks  # Roughly 1 date per week for each day
            
            if info['final_count'] < expected_dates * 0.8:  # Less than 80% of expected
                issues.append(f"**{day}**: Only {info['final_count']} dates found (expected ~{expected_dates})")
        
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("No obvious issues detected")
        
        # Date range info
        st.write("### Date Range Info")
        end_date = df['Date'].max()
        start_date = end_date - timedelta(weeks=weeks_history)
        st.write(f"- Start: {start_date.strftime('%Y-%m-%d')}")
        st.write(f"- End: {end_date.strftime('%Y-%m-%d')}")
        st.write(f"- Weeks: {weeks_history}")
        
        # Check for holidays in the period
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=start_date, end=end_date)
        
        if len(holidays) > 0:
            st.write(f"### US Market Holidays in Period ({len(holidays)} total):")
            for holiday in holidays:
                day_of_week = holiday.strftime('%A')
                if day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    st.write(f"- {holiday.strftime('%Y-%m-%d')} ({day_of_week})")
# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_sortino_ratio(returns, mar=0):
    """Calculate Sortino ratio (return/downside deviation)"""
    if len(returns) < 2:
        return np.nan
    mean_return = np.mean(returns)
    downside_returns = np.minimum(returns - mar, 0)
    downside_dev = np.sqrt(np.mean(downside_returns ** 2))
    if downside_dev == 0:
        return np.nan
    return mean_return / downside_dev

def calculate_profit_factor(profits):
    """Calculate profit factor (sum of wins / sum of losses)"""
    wins = profits[profits > 0].sum()
    losses = abs(profits[profits < 0].sum())
    if losses == 0:
        return np.inf if wins > 0 else 0
    return wins / losses

def calculate_composite_score(metrics, weights):
    """Calculate weighted composite score"""
    score = 0
    total_weight = 0
    
    for metric, weight in weights.items():
        if metric in metrics and not pd.isna(metrics[metric]):
            # Normalize each metric to 0-1 scale
            if metric == 'sortino_ratio':
                # Sortino typically ranges from -2 to 3
                normalized = (metrics[metric] + 2) / 5
            elif metric == 'avg_profit':
                # Normalize based on typical profit range
                normalized = (metrics[metric] + 1000) / 3000
            elif metric == 'win_rate':
                # Already 0-1
                normalized = metrics[metric]
            elif metric == 'profit_factor':
                # Profit factor typically 0-3
                normalized = min(metrics[metric], 3) / 3
            else:
                normalized = 0.5
            
            normalized = max(0, min(1, normalized))  # Clamp to 0-1
            score += weight * normalized
            total_weight += weight
    
    if total_weight > 0:
        score = score / total_weight
    
    return score

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns"""
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
    """Return color style based on score value"""
    if score >= 0.70:
        return "background-color: #90EE90; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold;"  # Light green
    elif score >= 0.50:
        return "background-color: #FFD700; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold;"  # Gold
    else:
        return "background-color: #FF6B6B; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold;"  # Red

# ============================================================================
# DATA PROCESSING
# ============================================================================

def _download_drive_bytes() -> bytes:
    sess = requests.Session()
    url = cfg.get_data_url()  # includes ?v=<date-based version> to bust caches
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
        # If you updated the downloader to use cfg.get_data_url():
        data_bytes = _download_drive_bytes()  
        # Otherwise keep the old call:
        # data_bytes = _download_drive_bytes(cfg.GOOGLE_DRIVE_FILE_ID)

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

    # Filter & format
    df = df[(df['Symbol'] == symbol) & (df['Name'] == strategy)].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'].astype(str), errors='coerce').dt.time
    df = df.dropna(subset=['Date', 'Entry_Time'])

    # Sidebar status
    latest_date = df['Date'].max()
    st.sidebar.success(f"✅ Data loaded from {source}")
    if pd.notna(latest_date):
        st.sidebar.caption(f"Latest date: {latest_date:%m/%d/%Y} • v={cfg.get_data_version()}")

    return df

def filter_data_by_weeks(df, weeks_back, exclude_fomc=True, exclude_earnings=True):
    """Filter data for the specified number of weeks"""
    end_date = df['Date'].max()
    start_date = end_date - timedelta(weeks=weeks_back)
    
    filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    
    if exclude_fomc:
        fomc_dates = pd.to_datetime(cfg.FOMC_DATES)
        filtered = filtered[~filtered['Date'].isin(fomc_dates)]
    
    if exclude_earnings:
        earnings_dates = pd.to_datetime(cfg.EARNINGS_DATES)
        filtered = filtered[~filtered['Date'].isin(earnings_dates)]
    
    # Filter out late times
    filtered = filtered[filtered['Entry_Time'] <= pd.to_datetime('15:30', format='%H:%M').time()]
    
    return filtered

def calculate_metrics_for_times(df, contracts=1):
    """Calculate all metrics for each day/time combination"""
    results = []
    
    # Apply contracts multiplier
    df = df.copy()
    df['Profit'] = df['Profit'] * contracts
    
    # Group by day of week and entry time
    for (dow, entry_time), group in df.groupby(['Day_of_week', 'Entry_Time']):
        if len(group) < 3:  # Skip if too few trades
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
    """Get top 3 times for each day based on composite score"""
    # Calculate composite scores
    metrics_df['composite_score'] = metrics_df.apply(
        lambda row: calculate_composite_score(row.to_dict(), weights), 
        axis=1
    )
    
    # Get top 3 for each day
    top_times = []
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    for day in days_order:
        day_data = metrics_df[metrics_df['Day_of_week'] == day].copy()
        
        if len(day_data) > 0:
            day_data = day_data.nlargest(3, 'composite_score')
            day_data['rank'] = range(1, len(day_data) + 1)
            top_times.append(day_data)
    
    if top_times:
        return pd.concat(top_times)
    return pd.DataFrame()

@st.cache_data
def find_optimal_weeks(symbol, strategy, min_weeks=4, max_weeks=52, 
                       exclude_fomc=True, exclude_earnings=True, contracts=1):
    """Find the week count with highest average profit per trade"""
    
    # Set specific defaults for SPX Butterfly
    if symbol == "SPX" and strategy == "Butterfly":
        return 20
    
    try:
        # Load historical data once
        df = load_historical_data(symbol, strategy, data_version=cfg.get_data_version())

        if df.empty:
            return 30  # Default fallback for other combinations
        
        results = []
        
        # Test different week ranges
        for weeks in range(min_weeks, min(max_weeks + 1, 53)):
            # Filter data for this week count
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
            # Find the week count with highest average profit
            results_df = pd.DataFrame(results)
            optimal_weeks = results_df.loc[results_df['avg_profit'].idxmax(), 'weeks']
            return int(optimal_weeks)
        
        return 30  # Default fallback
        
    except Exception as e:
        st.error(f"Error finding optimal weeks: {e}")
        return 30  # Default fallback

# ============================================================================
# FORWARD TESTING FUNCTIONS
# ============================================================================

def run_forward_test(df, training_weeks, trading_weeks, rank, day_filter, 
                     active_weights, contracts, starting_balance,
                     exclude_fomc=True, exclude_earnings=True):
    """Run walk-forward testing simulation with consistent filters in BOTH
    training and trading windows."""
    df = df.sort_values('Date').copy()

    # Precompute exclusion calendars once
    fomc_dates = pd.to_datetime(cfg.FOMC_DATES) if exclude_fomc else pd.DatetimeIndex([])
    earnings_dates = pd.to_datetime(cfg.EARNINGS_DATES) if exclude_earnings else pd.DatetimeIndex([])

    min_date = df['Date'].min()
    max_date = df['Date'].max()

    equity_curve, trades_log, retraining_log = [], [], []
    current_balance = starting_balance

    current_start = min_date
    while current_start < max_date:
        training_end = current_start + timedelta(weeks=training_weeks)
        trading_start = training_end
        trading_end = min(trading_start + timedelta(weeks=trading_weeks), max_date)

        if trading_start >= max_date:
            break

        # --- TRAINING DATA (apply exclusions) ---
        training_data = df[(df['Date'] >= current_start) & (df['Date'] < training_end)].copy()
        if exclude_fomc:
            training_data = training_data[~training_data['Date'].isin(fomc_dates)]
        if exclude_earnings:
            training_data = training_data[~training_data['Date'].isin(earnings_dates)]

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

        # --- TRADING DATA (apply SAME exclusions) ---
        trading_data = df[(df['Date'] >= trading_start) & (df['Date'] < trading_end)].copy()
        if exclude_fomc:
            trading_data = trading_data[~trading_data['Date'].isin(fomc_dates)]
        if exclude_earnings:
            trading_data = trading_data[~trading_data['Date'].isin(earnings_dates)]
        if day_filter != "All Days":
            trading_data = trading_data[trading_data['Day_of_week'] == day_filter]

        if trading_data.empty or selected_times.empty:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        # Vectorized selection instead of row-wise lambda
        key_trd = trading_data.assign(_key=trading_data['Day_of_week'].astype(str) + '|' + trading_data['Entry_Time'].astype(str))
        key_sel = selected_times.assign(_key=selected_times['Day_of_week'].astype(str) + '|' + selected_times['Entry_Time'].astype(str))['_key']
        trading_data = trading_data[key_trd['_key'].isin(set(key_sel))]

        if trading_data.empty:
            current_start = current_start + timedelta(weeks=trading_weeks)
            continue

        # Execute trades
        for _, trade in trading_data.sort_values('Date').iterrows():
            profit = trade['Profit'] * contracts
            current_balance += profit
            equity_curve.append({'Date': trade['Date'], 'Balance': current_balance, 'Profit': profit})
            trades_log.append({'Date': trade['Date'], 'Day': trade['Day_of_week'], 'Time': trade['Entry_Time'], 'Profit': profit})

        current_start = current_start + timedelta(weeks=trading_weeks)

    equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    retraining_df = pd.DataFrame(retraining_log) if retraining_log else pd.DataFrame()
    return equity_df, trades_df, retraining_df


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Time Trends Dashboard (TTD)",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS - only apply gradient to sidebar slider
    st.markdown("""
    <style>
    [data-testid="stSidebar"] .stSlider > div > div {
        background: linear-gradient(to right, #ff4444 0%, #ffff00 50%, #44ff44 100%);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ========== SIDEBAR CONFIGURATION ==========
    with st.sidebar:
        #st.title("⚙️ Settings")     


        # Strategy Selection
        st.subheader("📊 Strategy")
        
        symbol = st.selectbox(
            "Symbol",
            ["SPX", "XSP", "RUT", "NDX"],
            key="symbol"
        )
        
        strategy = st.selectbox(
            "Strategy Type",
            ["Butterfly", "Iron Condor", "Vertical", "Sonar"],
            key="strategy"
        )
        
        # Weeks of History Slider with Auto-Optimization
        st.subheader("📅 Data Range")
        
        # Initialize optimal weeks in session state if not present
        if 'optimal_weeks' not in st.session_state or \
           st.session_state.get('last_symbol') != symbol or \
           st.session_state.get('last_strategy') != strategy:
            
            with st.spinner("Finding optimal period..."):
                optimal_weeks = find_optimal_weeks(
                    symbol, 
                    strategy,
                    exclude_fomc=st.session_state.get('exclude_fomc', True),
                    exclude_earnings=st.session_state.get('exclude_earnings', True),
                    contracts=st.session_state.get('contracts', 1)
                )
                st.session_state.optimal_weeks = optimal_weeks
                st.session_state.last_symbol = symbol
                st.session_state.last_strategy = strategy
        
        # Check if reset was clicked
        if 'reset_to_optimal' not in st.session_state:
            st.session_state.reset_to_optimal = False
        
        # Determine the value to use for the slider
        if st.session_state.reset_to_optimal:
            slider_value = st.session_state.optimal_weeks
            st.session_state.reset_to_optimal = False  # Reset the flag
        else:
            slider_value = st.session_state.get('weeks_history', st.session_state.optimal_weeks)
        
        # Display optimization info
        col1, col2 = st.columns([3, 1])
        with col1:
            weeks_history = st.slider(
                "Weeks of History",
                min_value=4,
                max_value=52,
                value=slider_value, 
                step=1,
                help=f"Auto-optimized to {st.session_state.optimal_weeks} weeks for best avg profit/trade",
                key="weeks_history"
            )
        
        with col2:
                if st.button("🎯", help="Reset to optimal weeks"):
                    st.session_state.reset_to_optimal = True
                    st.rerun()
        
        # Show optimization details
        if weeks_history == st.session_state.optimal_weeks:
            st.success(f"✨ Using optimal period for Avg Profit ({weeks_history} weeks)")
        else:
            st.info(f"📊 Custom period (Optimal Avg Profit: {st.session_state.optimal_weeks} weeks)")
        
        with st.expander("ℹ️ What does 'Optimal' mean?", expanded=False):
            st.markdown(f"""
            **Understanding the Optimal Period:**
            
            🎯 **Optimal: {st.session_state.optimal_weeks} weeks** = The historical period that produced the highest average profit per trade
            
            **How it's calculated:**
            - We tested all periods from 4 to 52 weeks
            - Found that {st.session_state.optimal_weeks} weeks gave the best avg profit/trade
            - This is based on your current filter settings (FOMC, earnings, etc.)
            
            **Current Selection: {weeks_history} weeks**
            
            **Important Notes:**
            - "Optimal" is based on historical data only
            - Past performance doesn't guarantee future results
            - Different periods may reveal different patterns
            - Consider testing multiple timeframes for robustness
            
            💡 **Tip:** Click the 🎯 button to reset to the optimal period
            """)

        st.caption(f"Analyzing {weeks_history} weeks of data")
        
        # Quick analysis button
        with st.expander("📈 Week Range Analysis"):
            if st.button("Analyze All Ranges"):
                with st.spinner("Analyzing..."):
                    df = load_historical_data(symbol, strategy, data_version=cfg.get_data_version())
                    if not df.empty:
                        analysis_results = []
                        for w in [4, 8, 12, 16, 20, 26, 30, 39, 52]:
                            filtered = filter_data_by_weeks(
                                df, w, 
                                st.session_state.get('exclude_fomc', True),
                                st.session_state.get('exclude_earnings', True)
                            )
                            if len(filtered) > 0:
                                analysis_results.append({
                                    'Weeks': w,
                                    'Avg Profit': f"${filtered['Profit'].mean():.0f}",
                                    'Total': f"${filtered['Profit'].sum():.0f}",
                                    'Trades': len(filtered),
                                    'Win Rate': f"{(filtered['Profit'] > 0).mean()*100:.1f}%"
                                })
                        
                        if analysis_results:
                            analysis_df = pd.DataFrame(analysis_results)
                            st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                            
                            # Highlight the optimal row
                            optimal_row = analysis_df[analysis_df['Weeks'] == st.session_state.optimal_weeks]
                            if not optimal_row.empty:
                                st.success(f"🎯 Optimal: {st.session_state.optimal_weeks} weeks")
        
        # Metric Weights
        st.subheader("⚖️ Metric Weights")
        st.caption("Adjust importance of each metric")
        
        # Initialize metric settings
        if 'metric_weights' not in st.session_state:
            st.session_state.metric_weights = {
                'sortino_ratio': {'enabled': True, 'weight': 30},
                'avg_profit': {'enabled': True, 'weight': 30},
                'win_rate': {'enabled': True, 'weight': 20},
                'profit_factor': {'enabled': True, 'weight': 20}
            }
        
        # Metric configuration
        metrics_config = {
            'sortino_ratio': {'label': '📊 Sortino Ratio', 'desc': 'Risk-adjusted returns'},
            'avg_profit': {'label': '💰 Average Profit', 'desc': 'Profit per trade'},
            'win_rate': {'label': '🎯 Win Rate', 'desc': 'Win frequency'},
            'profit_factor': {'label': '📈 Profit Factor', 'desc': 'Win/Loss ratio'}
        }
        
        total_weight = 0
        active_weights = {}
        
        for metric, config in metrics_config.items():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                enabled = st.checkbox(
                    config['label'],
                    value=st.session_state.metric_weights[metric]['enabled'],
                    key=f"check_{metric}"
                )
                st.session_state.metric_weights[metric]['enabled'] = enabled
            
            with col2:
                if enabled:
                    weight = st.slider(
                        config['desc'],
                        min_value=0,
                        max_value=100,
                        value=st.session_state.metric_weights[metric]['weight'],
                        key=f"weight_{metric}"
                    )
                    st.session_state.metric_weights[metric]['weight'] = weight
                    active_weights[metric] = weight
                    total_weight += weight
        
        # Weight summary
        if total_weight > 0:
            if total_weight == 100:
                st.success(f"✅ Weights = 100%")
            else:
                st.warning(f"⚠️ Weights = {total_weight}%")
        
        # Quick Presets
        st.subheader("🎯 Quick Presets")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Conservative", use_container_width=True):
                st.session_state.metric_weights = {
                    'sortino_ratio': {'enabled': True, 'weight': 40},
                    'avg_profit': {'enabled': True, 'weight': 25},
                    'win_rate': {'enabled': True, 'weight': 25},
                    'profit_factor': {'enabled': True, 'weight': 10}
                }
                st.rerun()
            
            if st.button("Balanced", use_container_width=True):
                st.session_state.metric_weights = {
                    'sortino_ratio': {'enabled': True, 'weight': 25},
                    'avg_profit': {'enabled': True, 'weight': 25},
                    'win_rate': {'enabled': True, 'weight': 25},
                    'profit_factor': {'enabled': True, 'weight': 25}
                }
                st.rerun()
        
        with col2:
            if st.button("Aggressive", use_container_width=True):
                st.session_state.metric_weights = {
                    'sortino_ratio': {'enabled': True, 'weight': 15},
                    'avg_profit': {'enabled': True, 'weight': 40},
                    'win_rate': {'enabled': True, 'weight': 20},
                    'profit_factor': {'enabled': True, 'weight': 25}
                }
                st.rerun()
            
            if st.button("Risk-Averse", use_container_width=True):
                st.session_state.metric_weights = {
                    'sortino_ratio': {'enabled': True, 'weight': 35},
                    'avg_profit': {'enabled': True, 'weight': 20},
                    'win_rate': {'enabled': True, 'weight': 30},
                    'profit_factor': {'enabled': True, 'weight': 15}
                }
                st.rerun()
        
        # Data Filters
        st.subheader("🔧 Data Filters")
        
        exclude_fomc = st.checkbox("Exclude FOMC Days", value=True, key="exclude_fomc")
        exclude_earnings = st.checkbox("Exclude Earnings Days", value=True, key="exclude_earnings")
        
        # Trade Settings
        st.subheader("💼 Trade Settings")
        
        contracts = st.number_input(
            "Number of Contracts",
            min_value=1,
            max_value=100,
            value=1,
            key="contracts"
        )
        
        starting_balance = st.number_input(
            "Starting Account Balance ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            format="%d",
            key="starting_balance"
        )
        st.markdown("---")

        if st.button("🔄 Force refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("optimal_weeks", None)  # optional: force re-optimize
            st.rerun()   
    # ========== MAIN CONTENT AREA ==========
   
    # Load data first to get the latest date
    with st.spinner("Loading data..."):
        # Load historical data
        df = load_historical_data(symbol, strategy, data_version=cfg.get_data_version())
        
        if df.empty:
            st.error("No data available for selected symbol and strategy")
            return
        
        # Get the latest date from the data for display
        latest_data_date = df['Date'].max().strftime('%m/%d/%Y')
        
        # Filter data
        filtered_df = filter_data_by_weeks(
            df, 
            weeks_history, 
            exclude_fomc, 
            exclude_earnings
        )
    
        # Now display the title with version and data date
        st.markdown(f"""
        <h1 style='text-align:left;margin-bottom:0'>
            📊 Time Trends Dashboard (TTD)
            <span style='font-size:0.5em;color:#1E90FF;font-weight:400;'>by jb-trader</span>
            <span style='font-size:0.4em;color:#888;font-weight:400;margin-left:20px;'>
                Version 1.2.0 | Source: M8B v1.37 | Data updated: {latest_data_date}
            </span>
        </h1>
        """, unsafe_allow_html=True)

        st.markdown(f"Analyzing {weeks_history} weeks of historical data | Live updates enabled")
        
        # Calculate metrics
        metrics_df = calculate_metrics_for_times(filtered_df, contracts)
        # Add debug section for troubleshooting
        add_debug_section(df, weeks_history, exclude_fomc, exclude_earnings)
        # Get active weights
        active_weights = {
            k: v['weight'] for k, v in st.session_state.metric_weights.items() 
            if v['enabled']
        }
        
        # Get top times
        top_times_df = get_top_times_by_day(metrics_df, active_weights)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Best Times",
        "📈 Performance",
        "🔄 Forward Testing",
        "📊 Analysis",
        "📋 All Times",
        "✅ Validation"
    ])
    
    # ========== TAB 1: BEST TIMES ==========
    with tab1:
        st.subheader("🎯 Top 3 Trading Times by Day")
        
        # Add info box explaining scores
        with st.expander("ℹ️ What does the Score mean?", expanded=False):
            st.info("""
            **Composite Score (0 to 1 scale)**
            
            The score combines multiple performance metrics into a single ranking value:
            
            **Components:**
            - **Sortino Ratio** - Measures risk-adjusted returns (downside risk focus)
            - **Average Profit** - Mean profit per trade in dollars
            - **Win Rate** - Percentage of profitable trades
            - **Profit Factor** - Ratio of total wins to total losses
            
            **Score Interpretation:**
            - 🟢 **0.70+** = Excellent historical performance
            - 🟡 **0.50-0.70** = Good historical performance  
            - 🔴 **Below 0.50** = Below average historical performance
                        
            **Customization:**
            You can adjust how much each metric contributes to the score using the 
            Metric Weights sliders in the sidebar. This lets you prioritize what matters 
            most to your trading style (e.g., consistency vs. profit potential).
            
            **Note:** Scores are relative to other time slots in your selected date range 
            and represent historical performance only - not predictions of future results.
            """)
        
        if not top_times_df.empty:
            # Get current week dates
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            today_et = datetime.now(et_tz)


            # If it's Friday after 4pm ET or any time on weekend, show next week
            if (today_et.weekday() == 4 and today_et.hour >= 16) or today_et.weekday() in [5, 6]:
                # Show next week
                monday = today_et - timedelta(days=today_et.weekday()) + timedelta(days=7)
            else:
                # Show current week
                monday = today_et - timedelta(days=today_et.weekday())

            # Remove timezone info for display
            monday = monday.replace(tzinfo=None)
            
            # Create header
            cols = st.columns([1, 2, 2, 2, 2, 2])
            cols[0].markdown("**Rank**")
            
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for i, day in enumerate(days):
                date = monday + timedelta(days=i)
                cols[i+1].markdown(f"**{day}**")
                cols[i+1].caption(f"{date:%m/%d/%Y}")
            
            # Display top 3 times
            for rank in range(1, 4):
                cols = st.columns([1, 2, 2, 2, 2, 2])
                
                # Rank column
                if rank == 1:
                    cols[0].markdown("**1st** 🥇")
                elif rank == 2:
                    cols[0].markdown("**2nd** 🥈")
                else:
                    cols[0].markdown("**3rd** 🥉")
                
                # Times for each day
                for i, day in enumerate(days):
                    day_data = top_times_df[
                        (top_times_df['Day_of_week'] == day) & 
                        (top_times_df['rank'] == rank)
                    ]
                    
                    if not day_data.empty:
                        row = day_data.iloc[0]
                        time_str = row['Entry_Time'].strftime('%H:%M')
                        score = row['composite_score']
                        
                        # Display the time
                        cols[i+1].markdown(f"**{time_str}**")
                        
                        # Display the colored score
                        score_html = f'<span style="{get_score_color_style(score)}">Score: {score:.3f}</span>'
                        cols[i+1].markdown(score_html, unsafe_allow_html=True)
                    else:
                        cols[i+1].write("-")
            
            # Detailed metrics table
            with st.expander("📊 Detailed Metrics for Top Times"):
                display_df = top_times_df[['Day_of_week', 'rank', 'Entry_Time', 
                                          'composite_score', 'sortino_ratio', 
                                          'avg_profit', 'win_rate', 'profit_factor',
                                          'trade_count']].copy()
                
                display_df.columns = ['Day', 'Rank', 'Time', 'Score', 'Sortino', 
                                      'Avg Profit', 'Win Rate', 'Profit Factor', 'Trades']
                
                # Format columns
                display_df['Win Rate'] = (display_df['Win Rate'] * 100).round(1).astype(str) + '%'
                display_df['Avg Profit'] = '$' + display_df['Avg Profit'].round(0).astype(str)
                display_df['Score'] = display_df['Score'].round(3)
                display_df['Sortino'] = display_df['Sortino'].round(2)
                display_df['Profit Factor'] = display_df['Profit Factor'].round(2)
                
                # Define the styling function
                def style_scores(df_to_style):
                    def color_score(val):
                        if isinstance(val, (int, float)):
                            if val >= 0.70:
                                return 'background-color: #90EE90; color: black; font-weight: bold'   # green
                            elif val >= 0.50:
                                return 'background-color: #FFD700; color: black; font-weight: bold'   # gold
                            else:
                                return 'background-color: #FF6B6B; color: black; font-weight: bold'   # red
                        return ''
                    return df_to_style.style.map(color_score, subset=['Score'])

                
                # Display with styling
                styled_df = style_scores(display_df)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough data to display top times")
    
    # ========== TAB 2: PERFORMANCE ==========
    with tab2:
        st.subheader("📈 Historical Performance - Maybe This Trend Will Continue")
        
        # Add important disclaimer
        with st.expander("⚠️ **IMPORTANT: Understanding These Results**", expanded=False):
            st.warning("""
            **Critical Assumptions & Limitations:**
            
            **What This Shows:**
            - Performance if you had traded ONLY at the identified "best times" for each day
            - These "best times" were determined by analyzing ALL the historical data shown
            - This is essentially "perfect hindsight" - knowing in advance which times would work best
            
            **Why This Overstates Expected Returns:**
            - In real trading, you cannot know in advance which time will be "best" for any given day
            - The best times were identified using the same data used to calculate performance (optimization bias)
            - Past patterns may not continue into the future due to changing market conditions
            
            **Reality Check:**
            - These results represent the theoretical maximum performance with perfect timing
            - Actual trading results will likely be significantly lower
            - Market patterns change - what worked in the past may not work going forward
            - Consider this as pattern analysis, not a prediction of future returns
            
            **How to Use This Information:**
            - Look for consistent patterns across different time periods
            - Compare Rank 1, 2, and 3 performance to see how sensitive results are to timing
            - Use shorter analysis periods to see if patterns are stable
            - Consider this as ONE input among many in your trading decisions
            - Always use proper risk management regardless of historical performance
            """)
            
            st.info("""
            💡 **Tip**: The value in this analysis is identifying times that have *historically* shown better risk/reward 
            characteristics, not in expecting these exact returns. If these patterns persist, these times *may* continue 
            to offer an edge, but there's no guarantee.
            """)
        
        if not filtered_df.empty and not top_times_df.empty:
            # Add performance filters row WITHOUT using containers that cause tab switching
            col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 1, 1])
            
            with col1:
                # Use a form to prevent immediate rerun
                selected_rank = st.selectbox(
                    "Select Rank",
                    options=[1, 2, 3],
                    format_func=lambda x: f"{'1st' if x == 1 else '2nd' if x == 2 else '3rd'} Rank Times",
                    key="perf_rank_selector"
                )
            
            with col2:
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                selected_day = st.selectbox(
                    "Day of Week",
                    options=["All Days"] + days_order,
                    key="perf_day_selector"
                )
            
            with col3:
                perf_weeks = st.slider(
                    "Date Range (weeks)",
                    min_value=4,
                    max_value=52,
                    value=weeks_history,
                    key="perf_weeks_slider",
                    label_visibility="visible"
                )
            
            with col4:
                perf_exclude_fomc = st.checkbox(
                    "Exclude FOMC",
                    value=exclude_fomc,
                    key="perf_exclude_fomc"
                )
            
            with col5:
                perf_exclude_earnings = st.checkbox(
                    "Exclude Earnings",
                    value=exclude_earnings,
                    key="perf_exclude_earnings"
                )
            
            # Apply performance-specific filters
            perf_filtered_df = filter_data_by_weeks(
                df, 
                perf_weeks, 
                perf_exclude_fomc, 
                perf_exclude_earnings
            )
            
            # Recalculate metrics if filters changed
            if perf_weeks != weeks_history or perf_exclude_fomc != exclude_fomc or perf_exclude_earnings != exclude_earnings:
                perf_metrics_df = calculate_metrics_for_times(perf_filtered_df, contracts)
                perf_top_times_df = get_top_times_by_day(perf_metrics_df, active_weights)
            else:
                perf_top_times_df = top_times_df
                perf_filtered_df = filtered_df
            
            # Filter data based on selected rank times
            rank_times = perf_top_times_df[perf_top_times_df['rank'] == selected_rank][['Day_of_week', 'Entry_Time']]
            
            # Create a set of (day, time) tuples for filtering
            valid_times = set(zip(rank_times['Day_of_week'], rank_times['Entry_Time']))
            
            # Filter the original data to only include trades at the selected rank times
            # Build the set of rank-selected times
            rank_times = perf_top_times_df[
                perf_top_times_df['rank'] == selected_rank
            ][['Day_of_week', 'Entry_Time']]

            if selected_day != "All Days":
                rank_times = rank_times[rank_times['Day_of_week'] == selected_day]

            # Vectorized filter (replaces the apply/lambda block)
            rank_filtered_df = perf_filtered_df.merge(
                rank_times,
                on=['Day_of_week', 'Entry_Time'],
                how='inner'
            ).copy()

            
            # Apply day filter if not "All Days"
            if selected_day != "All Days":
                rank_filtered_df = rank_filtered_df[rank_filtered_df['Day_of_week'] == selected_day].copy()
            
            if not rank_filtered_df.empty:
                # Calculate cumulative P&L
                rank_filtered_df = rank_filtered_df.sort_values('Date')
                rank_filtered_df['Cumulative_PL'] = (rank_filtered_df['Profit'] * contracts).cumsum()
                
                # Create equity curve
                fig = go.Figure()
                
                # Add the equity curve for selected rank and day
                fig.add_trace(go.Scatter(
                    x=rank_filtered_df['Date'],
                    y=starting_balance + rank_filtered_df['Cumulative_PL'],
                    mode='lines',
                    name=f'Equity (Rank {selected_rank}, {selected_day})',
                    line=dict(color='green', width=4)
                ))
                
                # Add a reference line for starting balance
                fig.add_hline(
                    y=starting_balance, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Starting Balance"
                )
                
                fig.update_layout(
                    title=f"Equity Curve - {'1st' if selected_rank == 1 else '2nd' if selected_rank == 2 else '3rd'} Rank Times - {selected_day} ({perf_weeks} weeks)",
                    xaxis_title="Date",
                    yaxis_title="Account Value ($)",
                    yaxis_tickformat='$,.0f',
                    height=700,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_trades = len(rank_filtered_df)
                total_profit = rank_filtered_df['Profit'].sum() * contracts
                win_rate = (rank_filtered_df['Profit'] > 0).mean() * 100
                avg_profit = rank_filtered_df['Profit'].mean() * contracts
                
                col1.metric("Total Trades", f"{total_trades:,}")
                col2.metric("Total Profit", f"${total_profit:,.0f}")
                col3.metric("Win Rate", f"{win_rate:.1f}%")
                col4.metric("Avg Profit/Trade", f"${avg_profit:.0f}")
                
                # Additional metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                max_drawdown = calculate_max_drawdown(rank_filtered_df['Cumulative_PL'])
                sharpe = calculate_sharpe_ratio(rank_filtered_df['Profit'] * contracts)
                best_trade = (rank_filtered_df['Profit'] * contracts).max()
                worst_trade = (rank_filtered_df['Profit'] * contracts).min()
                
                col1.metric("Max Drawdown", f"${max_drawdown:,.0f}")
                col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                col3.metric("Best Trade", f"${best_trade:,.0f}")
                col4.metric("Worst Trade", f"${worst_trade:,.0f}")
                
                # Show which times are being used
                with st.expander(f"📅 Trading Times Used (Rank {selected_rank}, {selected_day})"):
                    times_display = rank_times.copy()
                    
                    # Filter by selected day if not "All Days"
                    if selected_day != "All Days":
                        times_display = times_display[times_display['Day_of_week'] == selected_day]
                    
                    times_display['Entry_Time'] = times_display['Entry_Time'].apply(lambda x: x.strftime('%H:%M'))
                    times_display = times_display.sort_values(['Day_of_week', 'Entry_Time'])
                    
                    # Create a formatted display
                    times_by_day = []
                    for day in days_order:
                        day_time = times_display[times_display['Day_of_week'] == day]
                        if not day_time.empty:
                            times_by_day.append(f"**{day}**: {day_time['Entry_Time'].iloc[0]}")
                    
                    if times_by_day:
                        st.markdown(" | ".join(times_by_day))
                    else:
                        st.markdown(f"**{selected_day}**: {times_display['Entry_Time'].iloc[0] if not times_display.empty else 'N/A'}")
                
                # Monthly breakdown
                st.subheader("Monthly Performance")
                monthly = rank_filtered_df.copy()
                monthly['YearMonth'] = monthly['Date'].dt.to_period('M')
                monthly_summary = monthly.groupby('YearMonth').agg({
                    'Profit': lambda x: (x * contracts).sum()
                }).reset_index()
                monthly_summary['YearMonth'] = monthly_summary['YearMonth'].astype(str)
                
                fig = px.bar(
                    monthly_summary,
                    x='YearMonth',
                    y='Profit',
                    color='Profit',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title=f"Monthly P&L (Rank {selected_rank} Times, {selected_day})"
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison across ranks (optional)
                with st.expander("📊 Compare All Ranks"):
                    comparison_data = []
                    
                    for rank in [1, 2, 3]:
                        rank_times_comp = perf_top_times_df[perf_top_times_df['rank'] == rank][['Day_of_week', 'Entry_Time']]
                        valid_times_comp = set(zip(rank_times_comp['Day_of_week'], rank_times_comp['Entry_Time']))
                        
                        rank_df_comp = perf_filtered_df[
                            perf_filtered_df.apply(lambda row: (row['Day_of_week'], row['Entry_Time']) in valid_times_comp, axis=1)
                        ].copy()
                        
                        # Apply day filter if not "All Days"
                        if selected_day != "All Days":
                            rank_df_comp = rank_df_comp[rank_df_comp['Day_of_week'] == selected_day]
                        
                        if not rank_df_comp.empty:
                            total_profit_comp = rank_df_comp['Profit'].sum() * contracts
                            win_rate_comp = (rank_df_comp['Profit'] > 0).mean() * 100
                            trades_comp = len(rank_df_comp)
                            
                            comparison_data.append({
                                'Rank': f'Rank {rank}',
                                'Total Profit': f'${total_profit_comp:,.0f}',
                                'Win Rate': f'{win_rate_comp:.1f}%',
                                'Total Trades': trades_comp,
                                'Avg Profit': f'${(total_profit_comp/trades_comp if trades_comp > 0 else 0):.0f}'
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        st.caption(f"Filtered by: {selected_day} | {perf_weeks} weeks")
            else:
                st.warning(f"No trades found for Rank {selected_rank} times on {selected_day}")
        else:
            st.warning("Not enough data to display performance metrics")
    
    # ========== TAB 3: FORWARD TESTING ==========
    with tab3:
        st.subheader("🔄 Forward Testing - Realistic Walk-Forward Analysis")
        
        # Add explanation dropdown
        with st.expander("📘 **Understanding Forward Testing Results**", expanded=False):
            st.info("""
            **What Is Forward Testing?**
                    
            This tab shows how the strategy would have performed in REAL TRADING conditions, 
            where only past data can be used to make decisions about the future.        
                    
            Easy explanation:  Use 20 weeks as Training Window to determine best times, 2 weeks to trade those times, repeat after 2 weeks. 
            This explanation assumes you set Training Window to 20 and Trading Window to 2.  Any metric filters you select are applied when 
            determining the best times to trade.
                    
            **How The Process Works:**
                    
             The graph begins at the end of the first x weeks Training Window (20 wks for example), since that is the earliest point where out-of-
             sample results exist. What the graph shows is the profit from each x-week Trading Window (2 wks for example), where a new 
             set of best times is selected after every Training Window and then applied to the following 2 weeks in this example. 
             This process repeats until the data runs out.
                    
            **Key Difference from Date Range (Weeks of History) slider in left panel:** 
                    
              The Date Range slider in the left panel simply limits how many past weeks are included in the analysis for the Best Times tab.
              The Target icon (Optimal Period) runs an optimization across multiple window lengths to find the period that 
              would have historically produced the best results, and uses that as the basis for selecting best trade times.  
              The Forward Testing tab, on the other hand, lets you experiment with different Training and Trading window sizes 
              to see how the strategy would have performed if applied step-by-step through time.                       
            
            **Key Difference from Performance Tab:**
                    
            - **Performance Tab**: Shows results using "perfect hindsight" - the best times are identified using ALL the data, then performance is calculated on that same data. This creates unrealistic expectations.
            - **Forward Testing Tab**: Shows realistic results - best times are identified using ONLY past data, then traded forward into unknown future data. This mimics actual trading conditions.
                    
            
            **How to Use These Settings:**
            
            **Training Window (4-26 weeks)**
            - Shorter (4-8 weeks): More responsive to recent patterns, but may overfit to noise
            - Medium (12-16 weeks): Balanced approach, recommended starting point
            - Longer (20+ weeks): More stable patterns, but slow to adapt to market changes
            
            **Trading Window (1-8 weeks)**
            - Shorter (1-2 weeks): Frequent retraining, adapts quickly but higher transaction costs
            - Medium (3-4 weeks): Good balance between adaptation and stability
            - Longer (6-8 weeks): Less frequent changes, but risk using stale patterns 

            """)
        
        # --- Forward Testing controls ---
    # --- Forward Testing controls ---
        # First row of controls
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 2, 2])
        
        with col1:
            training_weeks = st.slider(
                "Training Window",
                min_value=4,
                max_value=26,
                value=8,
                key="fwd_training_weeks",
                help="Weeks of historical data to identify patterns"
            )
        
        with col2:
            trading_weeks = st.slider(
                "Trading Window",
                min_value=1,
                max_value=8,
                value=2,
                key="fwd_trading_weeks",
                help="Weeks to trade before retraining"
            )
        
        with col3:
            fwd_rank = st.selectbox(
                "Select Rank",
                options=[1, 2, 3],
                format_func=lambda x: f"{'1st' if x == 1 else '2nd' if x == 2 else '3rd'} Rank",
                key="fwd_rank_selector"
            )
        
        with col4:
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            fwd_day = st.selectbox(
                "Day of Week",
                options=["All Days"] + days_order,
                key="fwd_day_selector"
            )
        
        # Second row for checkboxes
        col1, col2, col3 = st.columns([3, 1.5, 1.5])
        
        with col1:
            only_after_aug_upgrade = st.checkbox(
                "Only After Aug 2024 1.37 M8B upgrade",
                value=False,
                key="fwd_only_after_aug_2024",
                help="When checked, use only trades with Date ≥ 2024-08-01"
            )
        
        with col2:
            fwd_exclude_fomc = st.checkbox(
                "Exclude FOMC",
                value=exclude_fomc,
                key="fwd_exclude_fomc"
            )
        
        with col3:
            fwd_exclude_earnings = st.checkbox(
                "Exclude Earnings",
                value=exclude_earnings,
                key="fwd_exclude_earnings"
            )
        
        # Auto-run forward test when settings change
        # Check if we need to run based on changes to key parameters
        run_test = True  # Always run automatically
        
        if run_test:
            with st.spinner("Running forward test simulation..."):
                # Build dataset scope based on checkbox
                df_for_fwd = df.copy()
                if only_after_aug_upgrade:
                    cutoff = pd.Timestamp("2024-08-01")
                    df_for_fwd = df_for_fwd[df_for_fwd["Date"] >= cutoff]

                # Run forward test (exclusions applied inside the function)
                equity_df, trades_df, retraining_df = run_forward_test(
                    df_for_fwd,
                    training_weeks,
                    trading_weeks,
                    fwd_rank,
                    fwd_day,
                    active_weights,
                    contracts,
                    starting_balance,
                    fwd_exclude_fomc,
                    fwd_exclude_earnings
                )

                if not equity_df.empty:
                    # Hindsight comparison on the same scoped data
                    hindsight_filtered = df_for_fwd.copy()
                    if fwd_exclude_fomc:
                        hindsight_filtered = hindsight_filtered[
                            ~hindsight_filtered["Date"].isin(pd.to_datetime(cfg.FOMC_DATES))
                        ]
                    if fwd_exclude_earnings:
                        hindsight_filtered = hindsight_filtered[
                            ~hindsight_filtered["Date"].isin(pd.to_datetime(cfg.EARNINGS_DATES))
                        ]

                    hindsight_metrics = calculate_metrics_for_times(hindsight_filtered, contracts)
                    hindsight_top_times = get_top_times_by_day(hindsight_metrics, active_weights)
                    hindsight_times = hindsight_top_times[
                        hindsight_top_times["rank"] == fwd_rank
                    ][["Day_of_week", "Entry_Time"]]

                    hindsight_trades = hindsight_filtered.merge(
                        hindsight_times, on=["Day_of_week", "Entry_Time"], how="inner"
                    )
                    if fwd_day != "All Days":
                        hindsight_trades = hindsight_trades[hindsight_trades["Day_of_week"] == fwd_day]
                    hindsight_trades = hindsight_trades.sort_values("Date")
                    hindsight_trades["Cumulative_PL"] = (hindsight_trades["Profit"] * contracts).cumsum()

                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df["Date"],
                        y=equity_df["Balance"],
                        mode="lines",
                        name="Forward Test (Realistic)",
                        line=dict(color="blue", width=3)
                    ))
                    fig.add_hline(y=starting_balance, line_dash="dash", line_color="red", annotation_text="Starting Balance")
                    fig.update_layout(
                        title=f"Forward Test Results - Training: {training_weeks}w, Trading: {trading_weeks}w",
                        xaxis_title="Date",
                        yaxis_title="Account Value ($)",
                        yaxis_tickformat="$,.0f",
                        height=600,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics
                    if not trades_df.empty:
                        st.subheader("Trade Analysis")
                        
                        # Calculate metrics
                        # Calculate metrics
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
                    
                    # Show current times to trade based on latest training window
                    st.markdown("### 📍 Current Trading Times")
                    st.info("Based on your selected parameters, here are the times you should trade this week:")
                    
                    # Get the most recent training data
                    most_recent_date = df_for_fwd['Date'].max()
                    training_start = most_recent_date - timedelta(weeks=training_weeks)
                    
                    # Get training data for most recent window
                    recent_training = df_for_fwd[
                        (df_for_fwd['Date'] > training_start) & 
                        (df_for_fwd['Date'] <= most_recent_date)
                    ].copy()
                    
                    # Apply exclusions
                    if fwd_exclude_fomc:
                        recent_training = recent_training[~recent_training['Date'].isin(pd.to_datetime(cfg.FOMC_DATES))]
                    if fwd_exclude_earnings:
                        recent_training = recent_training[~recent_training['Date'].isin(pd.to_datetime(cfg.EARNINGS_DATES))]
                    
                    # Calculate metrics and get top times
                    recent_metrics = calculate_metrics_for_times(recent_training, contracts)
                    recent_top_times = get_top_times_by_day(recent_metrics, active_weights)
                    
                    # Filter for selected rank
                    current_times = recent_top_times[recent_top_times['rank'] == fwd_rank].copy()
                    
                    # Apply day filter if needed
                    if fwd_day != "All Days":
                        current_times = current_times[current_times['Day_of_week'] == fwd_day]
                    
                    if not current_times.empty:
                        # Display times by day
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
                        
                        st.caption(f"*Based on last {training_weeks} weeks of data through {most_recent_date.strftime('%m/%d/%Y')}*")
 
                        # Save/Load Settings Section
                        st.markdown("### 💾 Save/Load Configuration")
                        col_save, col_load = st.columns(2)
                        
                        with col_save:
                            # Prepare settings dictionary
                            settings_to_save = {
                                'training_weeks': training_weeks,
                                'trading_weeks': trading_weeks,
                                'rank': fwd_rank,
                                'day_filter': fwd_day,
                                'exclude_fomc': fwd_exclude_fomc,
                                'exclude_earnings': fwd_exclude_earnings,
                                'metric_weights': active_weights,
                                'contracts': contracts,
                                'starting_balance': starting_balance,
                                'symbol': symbol,
                                'strategy': strategy,
                                'current_times': current_times[['Day_of_week', 'Entry_Time', 'composite_score']].to_dict('records'),
                                'last_data_date': most_recent_date.strftime('%Y-%m-%d')
                            }
                            
                            # Convert Entry_Time to string for JSON serialization
                            for time_entry in settings_to_save['current_times']:
                                time_entry['Entry_Time'] = time_entry['Entry_Time'].strftime('%H:%M')
                            
                            import json
                            st.download_button(
                                label="📥 Download Settings",
                                data=json.dumps(settings_to_save, indent=2),
                                file_name=f"fwd_settings_{symbol}_{strategy}_{datetime.now():%Y%m%d_%H%M}.json",
                                mime="application/json",
                                help="Save your current configuration and trading times"
                            )
                        
                        with col_load:
                            uploaded_file = st.file_uploader("📤 Load Settings", type=['json'], key="fwd_settings_upload")
                            if uploaded_file is not None:
                                try:
                                    import json
                                    loaded_settings = json.load(uploaded_file)
                                    
                                    # Display what was loaded
                                    st.success("Settings loaded successfully!")
                                    st.caption(f"Config from: {loaded_settings.get('last_data_date', 'Unknown date')}")
                                    st.caption(f"Symbol: {loaded_settings.get('symbol', 'N/A')} | Strategy: {loaded_settings.get('strategy', 'N/A')}")
                                    
                                    # Show loaded parameters
                                    with st.expander("View Loaded Settings"):
                                        st.write(f"**Training Window:** {loaded_settings.get('training_weeks')} weeks")
                                        st.write(f"**Trading Window:** {loaded_settings.get('trading_weeks')} weeks")
                                        st.write(f"**Rank:** {loaded_settings.get('rank')}")
                                        st.write(f"**Day Filter:** {loaded_settings.get('day_filter')}")
                                        st.write(f"**Exclude FOMC:** {loaded_settings.get('exclude_fomc')}")
                                        st.write(f"**Exclude Earnings:** {loaded_settings.get('exclude_earnings')}")
                                        st.write("**Trading Times:**")
                                        for t in loaded_settings.get('current_times', []):
                                            st.write(f"  • {t['Day_of_week']}: {t['Entry_Time']} (Score: {t['composite_score']:.3f})")
                                    
                                    st.info("ℹ️ To apply these settings, manually adjust the controls above to match the loaded values")
                                    
                                except Exception as e:
                                    st.error(f"Error loading settings: {str(e)}")
                    else:
                        st.warning("No trading times found for current parameters")    
        
    # ========== TAB 4: ANALYSIS ==========
    with tab4:
        st.subheader("📊 Trading Analysis")

        if not metrics_df.empty:
            st.markdown("### Entry Time Performance Heatmap")

            # Compute composite scores for the heatmap
            metrics_df_display = metrics_df.copy()
            metrics_df_display["composite_score"] = metrics_df_display.apply(
                lambda row: calculate_composite_score(row.to_dict(), active_weights),
                axis=1
            )
            metrics_df_display["Entry_Time"] = metrics_df_display["Entry_Time"].apply(lambda x: x.strftime("%H:%M"))

            # Pivot to Entry_Time (rows) × Day_of_week (cols)
            heatmap_data = metrics_df_display.pivot_table(
                index="Entry_Time",
                columns="Day_of_week",
                values="composite_score",
                aggfunc="mean"
            )

            # Keep Mon–Fri order when present
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns])

            if not heatmap_data.empty:
                # Sort times chronologically (string "HH:MM" → datetime)
                heatmap_data = heatmap_data.sort_index(key=lambda s: pd.to_datetime(s, format="%H:%M"))

                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Day of Week", y="Entry Time", color="Score"),
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No heatmap data available with the current filters.")
        else:
            st.info("No metrics available to plot yet.")

    
    # ========== TAB 5: ALL TIMES ==========
    with tab5:
        st.subheader("📋 All Entry Times Ranked")
        
        if not metrics_df.empty:
            # Calculate composite scores for all times
            all_times = metrics_df.copy()
            all_times['composite_score'] = all_times.apply(
                lambda row: calculate_composite_score(row.to_dict(), active_weights),
                axis=1
            )
            
            # Sort by composite score
            all_times = all_times.sort_values('composite_score', ascending=False)
            
            # Format for display
            display_all = all_times[['Day_of_week', 'Entry_Time', 'composite_score',
                                     'sortino_ratio', 'avg_profit', 'win_rate',
                                     'profit_factor', 'trade_count']].copy()
            
            display_all.columns = ['Day', 'Time', 'Score', 'Sortino',
                                   'Avg Profit', 'Win Rate', 'Profit Factor', 'Trades']
            
            # Format columns
            display_all['Time'] = display_all['Time'].apply(lambda x: x.strftime('%H:%M'))
            display_all['Score'] = display_all['Score'].round(3)
            display_all['Sortino'] = display_all['Sortino'].round(2)
            display_all['Avg Profit'] = display_all['Avg Profit'].round(0)
            display_all['Win Rate'] = (display_all['Win Rate'] * 100).round(1)
            display_all['Profit Factor'] = display_all['Profit Factor'].round(2)
            
            # Add rank
            display_all.insert(0, 'Rank', range(1, len(display_all) + 1))
            
            # Display with pagination
            st.dataframe(
                display_all.head(50),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption(f"Showing top 50 of {len(display_all)} total entry times")
    
    # ========== TAB 6: Statistical Validation ==========        
    with tab6:
        st.subheader("✅ Statistical Validation")
        
        st.info("""
        **What This Validation Tests:**
        1. **Walk-Forward Analysis** - Tests if patterns work on future unseen data
        2. **Statistical Significance** - Determines if patterns are real or random
        3. **Pattern Stability** - Checks consistency over time
        """)
        
        if st.button("Run Complete Validation", type="primary"):
            # Create a more prominent processing display
            processing_container = st.empty()
            processing_container.warning("🔄 **VALIDATION IN PROGRESS** - Please wait 30-60 seconds...")
            
            with st.spinner("Running validation tests..."):
                try:
                    from robust_validation_module import WalkForwardValidator, MonteCarloTester, PatternStabilityAnalyzer
                    
                    # Create columns for progress
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    # 1. WALK-FORWARD ANALYSIS
                    progress_text.text("Running Walk-Forward Analysis...")
                    progress_bar.progress(10)
                    
                    st.markdown("### 1️⃣ Walk-Forward Analysis")
                    st.markdown("*Testing if patterns persist in out-of-sample data*")
                    
                    try:
                        wf_validator = WalkForwardValidator(filtered_df, training_weeks=12, testing_weeks=4)
                        wf_results = wf_validator.run_analysis(active_weights, top_n_times=3, contracts=contracts)
                        
                        if wf_results and wf_results['total_windows'] > 0:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Success Rate",
                                    f"{wf_results['success_rate']:.1f}%",
                                    delta=f"{wf_results['success_rate'] - 50:.1f}% vs random"
                                )
                            
                            with col2:
                                st.metric(
                                    "Profitable Windows",
                                    f"{wf_results['profitable_windows']}/{wf_results['total_windows']}"
                                )
                            
                            with col3:
                                st.metric(
                                    "Consistency Score",
                                    f"{wf_results['consistency_score']:.0f}/100"
                                )
                            
                            # Interpretation
                            if wf_results['success_rate'] >= 70:
                                st.success("✅ **Strong out-of-sample performance** - Patterns persist reliably")
                            elif wf_results['success_rate'] >= 50:
                                st.warning("🔶 **Moderate out-of-sample performance** - Use with caution")
                            else:
                                st.error("❌ **Poor out-of-sample performance** - Patterns don't persist")
                        else:
                            st.warning("⚠️ Insufficient data for walk-forward analysis. Try increasing the data range to 30+ weeks.")
                    
                    except Exception as e:
                        st.warning(f"Walk-forward analysis could not complete: {str(e)}")
                    
                    progress_bar.progress(40)
                    
                    # 2. MONTE CARLO SIGNIFICANCE TEST
                    progress_text.text("Running Statistical Significance Tests...")
                    
                    st.markdown("### 2️⃣ Statistical Significance Test")
                    st.markdown("*Determining if patterns are real or random chance*")
                    
                    # Get top times for testing
                    grouped = filtered_df.groupby(['Day_of_week', 'Entry_Time'])['Profit'].mean().nlargest(10)
                    best_times = [(day, time) for (day, time) in grouped.index]
                    
                    mc_tester = MonteCarloTester(filtered_df, n_simulations=300)
                    mc_results = mc_tester.test_time_pattern_significance(best_times, metric='profit')
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "P-Value",
                            f"{mc_results['p_value']:.4f}",
                            delta="Significant" if mc_results['p_value'] < 0.05 else "Not Significant"
                        )
                    
                    with col2:
                        st.metric(
                            "Z-Score",
                            f"{mc_results['z_score']:.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Percentile",
                            f"{mc_results['percentile']:.1f}%"
                        )
                    
                    if mc_results['is_significant']:
                        st.success("✅ **Patterns are STATISTICALLY SIGNIFICANT** (p < 0.05)")
                        st.write(f"Your patterns perform better than {mc_results['percentile']:.0f}% of random selections (based on {mc_tester.n_simulations:,} Monte Carlo simulations)")
                        
                    else:
                        st.error("❌ **Patterns are NOT statistically significant** - Could be random chance")
                        st.write(f"Based on {mc_tester.n_simulations:,} Monte Carlo simulations")
                   
                    progress_bar.progress(70)
                    
                    # 3. PATTERN STABILITY ANALYSIS
                    progress_text.text("Analyzing Pattern Stability...")
                    
                    st.markdown("### 3️⃣ Pattern Stability Analysis")
                    st.markdown("*Checking consistency of top patterns over time*")
                    
                    stability_analyzer = PatternStabilityAnalyzer(filtered_df)
                    stability_results = []
                    
                    for i, (day, time) in enumerate(best_times[:5]):
                        if i < 5:  # Test top 5 times
                            stability = stability_analyzer.analyze_time_stability(day, time)
                            if stability:
                                stability_results.append({
                                    'Day': day,
                                    'Time': time.strftime('%H:%M'),
                                    'Stability Score': f"{stability['stability_score']:.0f}/100",
                                    'Trend': '📈' if stability['trend_slope'] > 5 else '📉' if stability['is_deteriorating'] else '➡️',
                                    'Status': '✅' if stability['stability_score'] > 70 else '🔶' if stability['stability_score'] > 50 else '❌'
                                })
                    
                    if stability_results:
                        st.dataframe(pd.DataFrame(stability_results), use_container_width=True, hide_index=True)
                    
                    progress_bar.progress(90)
                    
                    # Clear the processing container
                    processing_container.empty()
                    
                    # 4. FINAL ASSESSMENT
                    st.markdown("### 📊 Overall Reliability Assessment")
                    
                    # Calculate reliability score
                    reliability_score = 0
                    components = []
                    
                    # Walk-forward component
                    if 'wf_results' in locals() and wf_results and wf_results['total_windows'] > 0:
                        wf_score = min(wf_results['success_rate'], 100) * 0.4
                        reliability_score += wf_score
                        components.append(('Walk-Forward Success', wf_score, 40))
                    else:
                        components.append(('Walk-Forward Success', 0, 40))
                    
                    # Statistical significance component
                    if mc_results['is_significant']:
                        mc_score = min(mc_results['percentile'], 100) * 0.3
                        reliability_score += mc_score
                        components.append(('Statistical Significance', mc_score, 30))
                    else:
                        components.append(('Statistical Significance', 0, 30))
                    
                    # Stability component (simplified)
                    stability_score = 20  # Default moderate
                    reliability_score += stability_score
                    components.append(('Pattern Stability', stability_score, 30))
                    
                    # Display reliability score with gauge
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(
                            "Overall Reliability Score",
                            f"{reliability_score:.0f}/100"
                        )
                        
                        # Color-coded interpretation
                        if reliability_score >= 70:
                            st.success("**HIGH RELIABILITY**")
                        elif reliability_score >= 50:
                            st.warning("**MODERATE RELIABILITY**")
                        else:
                            st.error("**LOW RELIABILITY**")
                    
                    with col2:
                        # Component breakdown
                        st.write("**Score Components:**")
                        for name, score, max_score in components:
                            pct = (score / max_score * 100) if max_score > 0 else 0
                            st.write(f"• {name}: {score:.1f}/{max_score} ({pct:.0f}%)")
                    
                    progress_bar.progress(100)
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # Final Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    if reliability_score >= 70:
                        st.success("""
                        ✅ **Pattern shows strong statistical validity**
                        - Do not overleverage your trades
                        - Start with conservative position sizes
                        - Monitor for pattern degradation weekly
                        """)
                    elif reliability_score >= 50:
                        st.warning("""
                        🔶 **Pattern shows moderate validity - Trade with caution**
                        - Mandatory paper trading for 2-4 weeks
                        - Use reduced position sizes (50% of normal)
                        - Re-validate monthly
                        - Stop if real results diverge from historical
                        """)
                    else:
                        st.error("""
                        ❌ **Pattern lacks statistical validity**
                        - High risk of overfitting detected
                        - NOT recommended for live trading
                        - Consider different time periods or strategies
                        - May need more data for reliable patterns
                        """)
                    
                    # Additional insights
                    with st.expander("📈 Detailed Interpretation Guide"):
                        st.markdown("""
                        **Understanding Your Results:**
                        
                        **Walk-Forward Analysis (40% of score)**
                        - Tests if patterns identified in past data work in future periods
                        - Success rate >60% is good, >70% is excellent
                        - This is the most important validation component
                        
                        **Statistical Significance (30% of score)**
                        - P-value <0.05 means patterns are unlikely to be random
                        - Z-score >2 indicates strong deviation from random
                        - Percentile shows how your patterns rank vs random selection
                        
                        **Pattern Stability (30% of score)**
                        - Checks if individual time slots maintain consistent performance
                        - Identifies deteriorating patterns that should be avoided
                        - Stability >70 means reliable, <50 means erratic
                        
                        **Risk Management Based on Score:**
                        - 70-100: Trade at normal position sizes
                        - 50-70: Reduce position sizes by 50%
                        - Below 50: Do not trade, continue research
                        """)
                    
                except ImportError as e:
                    processing_container.empty()
                    st.error("""
                    ❌ **Validation Module Not Found**
                    
                    Please ensure:
                    1. `robust_validation_module.py` is saved in the same directory
                    2. All required packages are installed: `pip install scipy pandas numpy`
                    
                    The validation module is essential for determining which patterns are real vs overfitted.
                    """)
                except Exception as e:
                    processing_container.empty()
                    st.error(f"An error occurred during validation: {str(e)}")
                    st.write("Please check your data and try again.")
    
    # Footer
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

# Run the application
if __name__ == "__main__":
    main()
                                         