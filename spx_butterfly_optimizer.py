#!/usr/bin/env python3
"""
SPX Butterfly Multi-Metric Trade Optimization System
Optimizes and ranks best trading times by 6 different metrics
Python version with performance optimizations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import time
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp

# Try to import optimization libraries
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not installed. Install with 'pip install numba' for 3-5x speed improvement")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings"""
    NUM_CONTRACTS = 1
    SYMBOL = "SPX"
    STRATEGY = "Butterfly"
    OUTPUT_DIR = Path("D:/_Documents/Magic 8 Ball/Best Time Reports/Best Times to Trade Reports")
    DATA_PATH = Path("D:/_Documents/Magic 8 Ball/data/dfe_table.parquet")
    
    # Lookback periods to test (in weeks)
    #LOOKBACK_VECTOR = [15, 20, 25, 30, 35, 40, 45, 50]
    LOOKBACK_VECTOR = list(range(15, 46, 2)) + [50]
    
    MIN_TRADES = 3
    
    # Exclusion dates
    FOMC_DATES = pd.to_datetime([
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"
    ])
    
    EARNINGS_DATES = pd.to_datetime([
    "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-21",
    "2024-02-22", "2024-04-24", "2024-04-25", "2024-04-26", "2024-04-30",
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-22", "2024-05-23",
    "2024-07-23", "2024-07-24", "2024-07-30", "2024-07-31", "2024-08-01",
    "2024-08-02", "2024-08-28", "2024-08-29", "2024-10-29", "2024-10-30",
    "2024-10-31", "2024-11-01", "2024-11-20", "2024-11-21", "2025-01-29",
    "2025-01-30", "2025-01-31", "2025-02-04", "2025-02-05", "2025-02-06",
    "2025-02-07", "2025-02-26", "2025-02-27", "2025-04-24", "2025-04-25",
    "2025-04-30", "2025-05-01", "2025-05-02", "2025-05-28", "2025-05-29",
    "2025-07-23", "2025-07-24", "2025-07-30", "2025-07-31", "2025-08-01",
    "2025-08-27", "2025-08-28"
    ])
    
    HOLIDAY_DATES = pd.to_datetime([
        "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04",
        "2024-09-02", "2024-11-28", "2024-12-25", "2025-01-01", "2025-01-20",
        "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01"
    ])
    
    DOW_LEVELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# ============================================================================
# OPTIMIZED METRIC CALCULATIONS
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def calculate_sortino_ratio_fast(returns, mar=0.0):
        """Numba-accelerated Sortino ratio calculation"""
        if len(returns) < 2:
            return np.nan
        
        mean_return = np.mean(returns)
        downside_returns = np.minimum(returns - mar, 0)
        downside_dev = np.sqrt(np.mean(downside_returns * downside_returns))
        
        if downside_dev == 0:
            return np.nan
        return mean_return / downside_dev
    
    @jit(nopython=True, cache=True)
    def calculate_sharpe_ratio_fast(returns, rf=0.0):
        """Numba-accelerated Sharpe ratio calculation"""
        if len(returns) < 2:
            return np.nan
        
        excess_returns = returns - rf
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns)
        
        if std_returns == 0:
            return np.nan
        return mean_excess / std_returns
    
    @jit(nopython=True, cache=True)
    def calculate_r_squared_fast(returns):
        """Numba-accelerated R-squared calculation"""
        if len(returns) < 3:
            return np.nan
        
        time_index = np.arange(len(returns), dtype=np.float64)
        cum_returns = np.cumsum(returns)
        
        # Calculate correlation coefficient
        mean_x = np.mean(time_index)
        mean_y = np.mean(cum_returns)
        
        num = np.sum((time_index - mean_x) * (cum_returns - mean_y))
        den_x = np.sum((time_index - mean_x) ** 2)
        den_y = np.sum((cum_returns - mean_y) ** 2)
        
        if den_x == 0 or den_y == 0:
            return np.nan
        
        r = num / np.sqrt(den_x * den_y)
        return r * r
else:
    # Pure Python fallbacks
    def calculate_sortino_ratio_fast(returns, mar=0):
        if len(returns) < 2:
            return np.nan
        mean_return = np.mean(returns)
        downside_returns = np.minimum(returns - mar, 0)
        downside_dev = np.sqrt(np.mean(downside_returns ** 2))
        if downside_dev == 0:
            return np.nan
        return mean_return / downside_dev
    
    def calculate_sharpe_ratio_fast(returns, rf=0):
        if len(returns) < 2:
            return np.nan
        excess_returns = returns - rf
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(returns)
        if std_returns == 0:
            return np.nan
        return mean_excess / std_returns
    
    def calculate_r_squared_fast(returns):
        if len(returns) < 3:
            return np.nan
        time_index = np.arange(len(returns))
        cum_returns = np.cumsum(returns)
        correlation = np.corrcoef(time_index, cum_returns)[0, 1]
        return correlation ** 2 if not np.isnan(correlation) else np.nan

# ============================================================================
# DATA PROCESSING
# ============================================================================

class TradingDataProcessor:
    """Handles all data loading and filtering operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.metrics_cache = {}
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare trading data"""
        print(f"\nLoading data from {self.config.DATA_PATH}...")
        
        # Load data
        df = pd.read_parquet(self.config.DATA_PATH)
        
        # Select relevant columns
        df = df[['Symbol', 'Name', 'Date', 'Entry_Time', 'Day_of_week', 
                 'Trade', 'Profit', 'Year', 'Month', 'Premium']]
        
        # Filter for symbol and strategy
        df = df[(df['Symbol'] == self.config.SYMBOL) & 
                (df['Name'] == self.config.STRATEGY)].copy()
        
        # Convert Entry_Time to time
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='%H:%M').dt.time
        
        # Set Day_of_week as categorical
        df['Day_of_week'] = pd.Categorical(df['Day_of_week'], 
                                           categories=self.config.DOW_LEVELS, 
                                           ordered=True)
        
        # Apply contract multiplier
        df['Profit'] = df['Profit'] * self.config.NUM_CONTRACTS
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date for efficiency
        df = df.sort_values('Date')
        
        self.df = df
        print(f"Loaded {len(df)} trades for {self.config.SYMBOL} {self.config.STRATEGY}")
    
    @lru_cache(maxsize=128)
    def filter_clean_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Filter data excluding FOMC, earnings, and holiday dates"""
        filtered = self.df[
            ~self.df['Date'].isin(self.config.FOMC_DATES) &
            ~self.df['Date'].isin(self.config.EARNINGS_DATES) &
            ~self.df['Date'].isin(self.config.HOLIDAY_DATES) &
            (self.df['Entry_Time'] <= pd.to_datetime('15:30', format='%H:%M').time())
        ].copy()
        
        if start_date:
            filtered = filtered[filtered['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[filtered['Date'] < pd.to_datetime(end_date)]
        
        return filtered

# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================

class MetricOptimizer:
    """Optimizes lookback periods and finds best trading times"""
    
    def __init__(self, data_processor: TradingDataProcessor, config: Config):
        self.data_processor = data_processor
        self.config = config
        self.results_cache = {}
    
    def calculate_metrics_for_group(self, group_data: pd.DataFrame) -> Dict:
        """Calculate all metrics for a group of trades"""
        if len(group_data) < self.config.MIN_TRADES:
            return None
        
        profits = group_data['Profit'].values
        
        return {
            'trade_count': len(group_data),
            'avg_profit': np.mean(profits),
            'total_profit': np.sum(profits),
            'win_rate': np.mean(profits > 0),
            'sortino_ratio': calculate_sortino_ratio_fast(profits),
            'sharpe_ratio': calculate_sharpe_ratio_fast(profits),
            'r_squared': calculate_r_squared_fast(profits)
        }
    
    def optimize_lookback_for_metric(self, metric_name: str) -> Optional[int]:
        """Optimize lookback period for a specific metric"""
        print(f"\nOptimizing lookback period for {metric_name}...")
        
        all_dates = sorted(self.data_processor.df['Date'].unique())
        results = []
        
        for weeks_back in self.config.LOOKBACK_VECTOR:
            print(f"  Testing {weeks_back} week lookback...", end='\r')
            
            # Find start index
            min_date = min(all_dates) + timedelta(weeks=weeks_back)
            start_idx = next((i for i, date in enumerate(all_dates) if date >= min_date), None)
            
            if start_idx is None:
                continue
            
            total_profit = 0
            total_trades = 0
            all_profits = []
            
            # Backtest using this lookback period
            for i in range(start_idx, len(all_dates)):
                current_date = all_dates[i]
                lookback_start = current_date - timedelta(weeks=weeks_back)
                
                # Get lookback data
                lookback_data = self.data_processor.filter_clean_data(
                    str(lookback_start.date()), 
                    str(current_date.date())
                )
                
                if len(lookback_data) == 0:
                    continue
                
                # Calculate metrics for each day/time combination
                performance = lookback_data.groupby(['Day_of_week', 'Entry_Time']).apply(
                    lambda x: pd.Series(self.calculate_metrics_for_group(x))
                ).reset_index()
                
                # Remove None results
                performance = performance.dropna()
                
                if len(performance) == 0:
                    continue
                
                # Select best time per day based on the current metric
                if metric_name in performance.columns:
                    best_times = (performance[performance[metric_name].notna()]
                                 .sort_values(metric_name, ascending=False)
                                 .groupby('Day_of_week')
                                 .first()
                                 .reset_index())
                    
                    # Get actual trades for current date
                    current_trades = self.data_processor.df[
                        self.data_processor.df['Date'] == current_date
                    ]
                    
                    # Join with best times
                    for _, best_time in best_times.iterrows():
                        matching_trades = current_trades[
                            (current_trades['Day_of_week'] == best_time['Day_of_week']) &
                            (current_trades['Entry_Time'] == best_time['Entry_Time'])
                        ]
                        
                        if len(matching_trades) > 0:
                            trade_profits = matching_trades['Profit'].values
                            total_profit += np.sum(trade_profits)
                            total_trades += len(trade_profits)
                            all_profits.extend(trade_profits)
            
            if len(all_profits) > 0:
                # Calculate metric value for this lookback period
                metric_value = self.calculate_metric_value(metric_name, all_profits)
                
                results.append({
                    'lookback_weeks': weeks_back,
                    'metric_value': metric_value,
                    'total_trades': total_trades,
                    'total_profit': total_profit
                })
        
        if not results:
            print(f"\n  WARNING: No valid results for {metric_name}")
            return None
        
        # Find optimal lookback
        results_df = pd.DataFrame(results)
        optimal = results_df.loc[results_df['metric_value'].idxmax()]
        
        print(f"\n  Optimal lookback for {metric_name}: {optimal['lookback_weeks']} weeks")
        print(f"    Metric value: {optimal['metric_value']:.4f}")
        print(f"    Total trades: {optimal['total_trades']}")
        
        return int(optimal['lookback_weeks'])
    
    def calculate_metric_value(self, metric_name: str, profits: List[float]) -> float:
        """Calculate a specific metric value"""
        if metric_name == 'sortino_ratio':
            return calculate_sortino_ratio_fast(np.array(profits))
        elif metric_name == 'sharpe_ratio':
            return calculate_sharpe_ratio_fast(np.array(profits))
        elif metric_name == 'r_squared':
            return calculate_r_squared_fast(np.array(profits))
        elif metric_name == 'avg_profit':
            return np.mean(profits)
        elif metric_name == 'win_rate':
            return np.mean(np.array(profits) > 0)
        elif metric_name == 'total_profit':
            return np.sum(profits)
        else:
            return np.nan
    
    def get_top_times_for_metric(self, metric_name: str, lookback_weeks: int) -> pd.DataFrame:
        """Get top 3 times per day for a metric"""
        last_date = self.data_processor.df['Date'].max()
        lookback_start = last_date - timedelta(weeks=lookback_weeks)
        
        # Get lookback data
        lookback_data = self.data_processor.filter_clean_data(
            str(lookback_start.date()),
            str(last_date.date())
        )
        
        if len(lookback_data) == 0:
            return pd.DataFrame()
        
        # Calculate all metrics
        performance = []
        for (dow, entry_time), group in lookback_data.groupby(['Day_of_week', 'Entry_Time']):
            metrics = self.calculate_metrics_for_group(group)
            if metrics:
                metrics['Day_of_week'] = dow
                metrics['Entry_Time'] = entry_time
                performance.append(metrics)
        
        if not performance:
            return pd.DataFrame()
        
        performance_df = pd.DataFrame(performance)
        
        # Get top 3 per day for this metric
        top_times = []
        for dow in self.config.DOW_LEVELS:
            dow_data = performance_df[
                (performance_df['Day_of_week'] == dow) &
                (performance_df[metric_name].notna())
            ]
            
            if len(dow_data) > 0:
                top_3 = dow_data.nlargest(3, metric_name).copy()
                top_3['rank'] = range(1, len(top_3) + 1)
                top_times.append(top_3)
        
        return pd.concat(top_times) if top_times else pd.DataFrame()

# ============================================================================
# PARALLEL OPTIMIZATION
# ============================================================================

def optimize_metric_parallel(args):
    """Worker function for parallel metric optimization"""
    metric_name, config_dict, df_dict = args
    
    # Reconstruct objects
    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    processor = TradingDataProcessor(config)
    processor.df = pd.DataFrame(df_dict)
    
    optimizer = MetricOptimizer(processor, config)
    
    # Optimize and get results
    optimal_weeks = optimizer.optimize_lookback_for_metric(metric_name)
    if optimal_weeks:
        top_times = optimizer.get_top_times_for_metric(metric_name, optimal_weeks)
        return metric_name, optimal_weeks, top_times
    
    return metric_name, None, pd.DataFrame()

# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generates HTML reports"""
    
    @staticmethod
    def get_trading_week_dates() -> pd.DataFrame:
        """Determine current trading week dates"""
        current_time = datetime.now()
        current_day = current_time.weekday()
        
        # Determine if we should use next week
        use_next_week = False
        
        if current_day >= 5:  # Saturday or Sunday
            use_next_week = True
        elif current_day == 4 and current_time.hour >= 16:  # Friday after 4 PM
            use_next_week = True
        
        # Calculate Monday
        if use_next_week:
            days_to_monday = 7 - current_day
            monday = current_time.date() + timedelta(days=days_to_monday)
        else:
            days_since_monday = current_day
            monday = current_time.date() - timedelta(days=days_since_monday)
        
        # Generate M-F dates
        week_dates = pd.DataFrame({
            'Day_of_week': Config.DOW_LEVELS,
            'Date': [monday + timedelta(days=i) for i in range(5)]
        })
        
        return week_dates
    
    @staticmethod
    def ordinal(n: int) -> str:
        """Convert number to ordinal"""
        if n == 1:
            return "1st"
        elif n == 2:
            return "2nd"
        elif n == 3:
            return "3rd"
        else:
            return f"{n}th"
    
    @staticmethod
    def format_metric(value: float, metric_name: str) -> str:
        """Format metric value for display"""
        if pd.isna(value):
            return "N/A"
        
        if metric_name in ['sortino_ratio', 'sharpe_ratio', 'r_squared']:
            return f"{value:.2f}"
        elif metric_name == 'win_rate':
            return f"{value:.2%}"
        elif metric_name in ['avg_profit', 'total_profit']:
            return f"${value:,.0f}"
        else:
            return f"{value:.2f}"
    
    @staticmethod
    def generate_consolidated_html(metrics_results: Dict, week_dates: pd.DataFrame, 
                                  config: Config) -> str:
        """Generate consolidated HTML report"""
        
        # Extract lookback info
        lookback_info = {}
        for metric_name, (optimal_weeks, top_times) in metrics_results.items():
            if optimal_weeks:
                lookback_info[metric_name] = optimal_weeks
        
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.SYMBOL} {config.STRATEGY} TRADE PLAN</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 20px;
            padding: 0;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 24px;
        }}
        .week-header {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }}
        .day-section {{
            background-color: #FFFACD;
            padding: 8px;
            margin-bottom: 10px;
            margin-top: 20px;
            font-weight: bold;
            font-size: 14px;
            border-left: 3px solid #FFD700;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th {{
            background-color: #f0f0f0;
            padding: 8px;
            text-align: center;
            font-size: 12px;
            border: 1px solid #ddd;
            font-weight: bold;
        }}
        .weeks-row {{
            background-color: #f8f8f8;
            font-weight: normal;
            font-size: 11px;
            color: #666;
        }}
        td {{
            padding: 8px;
            border: 1px solid #ddd;
            font-size: 13px;
            text-align: center;
            font-family: "Courier New", monospace;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .rank-cell {{
            font-weight: bold;
            font-family: Arial, sans-serif;
            background-color: #fafafa;
        }}
        .highlight-2 {{ background-color: #e6f3ff; font-weight: bold; }}
        .highlight-3 {{ background-color: #cce7ff; font-weight: bold; }}
        .highlight-4 {{ background-color: #b3ddff; font-weight: bold; }}
        .highlight-5 {{ background-color: #99d3ff; font-weight: bold; }}
        .highlight-6 {{ background-color: #80c9ff; font-weight: bold; }}
        .footer {{
            margin-top: 30px;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{config.SYMBOL} {config.STRATEGY} TRADE PLAN</h1>
        <div class="week-header">
            Week of: {week_dates['Date'].min():%m/%d/%Y} - {week_dates['Date'].max():%m/%d/%Y}
            &nbsp;&nbsp;&nbsp;&nbsp; (Report Generated: {datetime.now():%m/%d/%Y})
        </div>
'''
        
        # Process each day
        for day in Config.DOW_LEVELS:
            day_date = week_dates[week_dates['Day_of_week'] == day]['Date'].iloc[0]
            
            # Collect all times for this day to count frequencies
            all_day_times = []
            for metric_name, (_, top_times) in metrics_results.items():
                if not top_times.empty:
                    day_data = top_times[top_times['Day_of_week'] == day]
                    for _, row in day_data.iterrows():
                        if pd.notna(row['Entry_Time']):
                            time_str = row['Entry_Time'].strftime('%H:%M')
                            all_day_times.append(time_str)
            
            # Count frequencies
            from collections import Counter
            time_frequencies = Counter(all_day_times)
            
            html += f'''
        <div class="day-section">{day} - {day_date:%m/%d/%Y}</div>
        <table>
            <thead>
                <tr>
                    <th style="width:10%"></th>
                    <th style="width:15%">Sortino</th>
                    <th style="width:15%">Avg Profit</th>
                    <th style="width:15%">R²</th>
                    <th style="width:15%">Sharpe</th>
                    <th style="width:15%">Win Rate</th>
                    <th style="width:15%">Total Profit</th>
                </tr>
                <tr class="weeks-row">
                    <th></th>'''
            
            # Add weeks headers
            metric_names = ['sortino_ratio', 'avg_profit', 'r_squared', 
                           'sharpe_ratio', 'win_rate', 'total_profit']
            for metric in metric_names:
                weeks = lookback_info.get(metric, '')
                weeks_str = f"{weeks} wk" if weeks else ""
                html += f'<th>{weeks_str}</th>'
            
            html += '''
                </tr>
            </thead>
            <tbody>'''
            
            # Add top 3 times
            for rank in range(1, 4):
                html += f'''
                <tr>
                    <td class="rank-cell">{ReportGenerator.ordinal(rank)}</td>'''
                
                for metric_name in metric_names:
                    time_value = ""
                    cell_class = ""
                    
                    if metric_name in metrics_results:
                        _, top_times = metrics_results[metric_name]
                        if not top_times.empty:
                            day_data = top_times[
                                (top_times['Day_of_week'] == day) & 
                                (top_times['rank'] == rank)
                            ]
                            
                            if not day_data.empty:
                                time_value = day_data.iloc[0]['Entry_Time'].strftime('%H:%M')
                                
                                # Determine highlighting
                                freq = time_frequencies.get(time_value, 0)
                                if freq >= 6:
                                    cell_class = "highlight-6"
                                elif freq >= 5:
                                    cell_class = "highlight-5"
                                elif freq >= 4:
                                    cell_class = "highlight-4"
                                elif freq >= 3:
                                    cell_class = "highlight-3"
                                elif freq >= 2:
                                    cell_class = "highlight-2"
                    
                    html += f'<td class="{cell_class}">{time_value}</td>'
                
                html += '</tr>'
            
            html += '''
            </tbody>
        </table>'''
        
        # Add footer
        html += f'''
        <div class="footer">
            Report generated on: {datetime.now():%Y-%m-%d %H:%M:%S}<br>
            Contract Size: {config.NUM_CONTRACTS}<br>
            Data filtered for FOMC dates, earnings dates, and holidays<br>
            Each metric optimized with its own lookback period<br><br>
            <strong>Highlighting:</strong> Times appearing in multiple metrics are highlighted in blue<br>
            (Darker blue = appears in more metrics)
        </div>
    </div>
</body>
</html>'''
        
        return html

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("\n" + "="*60)
    print(f"     {Config.SYMBOL} {Config.STRATEGY} MULTI-METRIC OPTIMIZATION")
    print("="*60)
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize data processor
    processor = TradingDataProcessor(Config())
    
    # Get trading week dates
    week_dates = ReportGenerator.get_trading_week_dates()
    print(f"\nTrading week: {week_dates['Date'].min():%m/%d} - {week_dates['Date'].max():%m/%d/%Y}")
    
    # Define metrics
    metrics = ['sortino_ratio', 'avg_profit', 'r_squared', 'sharpe_ratio', 'win_rate', 'total_profit']
    
    # Optimize each metric (can be parallelized)
    optimizer = MetricOptimizer(processor, Config())
    metrics_results = {}
    
    # Check if we can use parallel processing
    use_parallel = mp.cpu_count() > 2 and len(metrics) > 3
    
    if use_parallel:
        print(f"\nUsing parallel processing with {mp.cpu_count()} cores...")
        
        # Prepare data for parallel processing
        config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
        df_dict = processor.df.to_dict('records')
        
        # Create tasks
        tasks = [(metric, config_dict, df_dict) for metric in metrics]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(metrics))) as executor:
            futures = {executor.submit(optimize_metric_parallel, task): task[0] 
                      for task in tasks}
            
            for future in as_completed(futures):
                metric_name = futures[future]
                try:
                    metric_name, optimal_weeks, top_times = future.result()
                    if optimal_weeks:
                        metrics_results[metric_name] = (optimal_weeks, top_times)
                except Exception as e:
                    print(f"Error processing {metric_name}: {e}")
    else:
        print("\nUsing sequential processing...")
        
        for metric in metrics:
            optimal_weeks = optimizer.optimize_lookback_for_metric(metric)
            if optimal_weeks:
                top_times = optimizer.get_top_times_for_metric(metric, optimal_weeks)
                metrics_results[metric] = (optimal_weeks, top_times)
    
    # Generate HTML report
    print("\nGenerating HTML reports...")
    
    html_content = ReportGenerator.generate_consolidated_html(
        metrics_results, week_dates, Config()
    )
    
    # Save consolidated report
    output_file = Config.OUTPUT_DIR / f"{Config.SYMBOL}_{Config.STRATEGY}_Consolidated_Trade_Plan_{datetime.now():%Y%m%d}.html"
    output_file.write_text(html_content, encoding='utf-8')
    print(f"Report saved to: {output_file}")
    
    # Performance summary
    end_time = time.time()
    runtime = (end_time - start_time) / 60
    print(f"\nTotal runtime: {runtime:.2f} minutes")
    print("Optimization complete!\n")

if __name__ == "__main__":
    main()
    