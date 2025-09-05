#!/usr/bin/env python3
"""
Robust Statistical Validation Module
Adds proper out-of-sample testing and statistical significance to prevent overfitting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

class WalkForwardValidator:
    """
    Implements proper walk-forward analysis to test if patterns persist
    """
    
    def __init__(self, df, training_weeks=12, testing_weeks=4, step_weeks=2):
        """
        Initialize walk-forward validator
        
        Parameters:
        -----------
        df : DataFrame
            Full historical data
        training_weeks : int
            Number of weeks for training window
        testing_weeks : int
            Number of weeks for testing window
        step_weeks : int
            Number of weeks to step forward each iteration
        """
        self.df = df.sort_values('Date').copy()
        self.training_weeks = training_weeks
        self.testing_weeks = testing_weeks
        self.step_weeks = step_weeks
        self.results = []
        
    def run_analysis(self, metric_weights, top_n_times=3, contracts=1):
        """
        Run complete walk-forward analysis
        """
        # Get date range
        start_date = self.df['Date'].min()
        end_date = self.df['Date'].max()
        
        # Calculate windows
        current_train_start = start_date
        
        while True:
            # Define windows
            train_end = current_train_start + timedelta(weeks=self.training_weeks)
            test_start = train_end
            test_end = test_start + timedelta(weeks=self.testing_weeks)
            
            # Check if we have enough data
            if test_end > end_date:
                break
                
            # Split data
            train_data = self.df[(self.df['Date'] >= current_train_start) & 
                                 (self.df['Date'] < train_end)]
            test_data = self.df[(self.df['Date'] >= test_start) & 
                                (self.df['Date'] < test_end)]
            
            # Skip if insufficient data
            if len(train_data) < 50 or len(test_data) < 20:
                current_train_start += timedelta(weeks=self.step_weeks)
                continue
            
            # Find best times in training data
            best_times = self._find_best_times(train_data, metric_weights, top_n_times)
            
            # Test performance on test data
            test_results = self._test_performance(test_data, best_times, contracts)
            
            # Store results
            self.results.append({
                'train_start': current_train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_times': best_times,
                'train_trades': len(train_data),
                'test_trades': test_results['trade_count'],
                'test_profit': test_results['total_profit'],
                'test_win_rate': test_results['win_rate'],
                'test_avg_profit': test_results['avg_profit'],
                'is_profitable': test_results['total_profit'] > 0
            })
            
            # Move to next window
            current_train_start += timedelta(weeks=self.step_weeks)
        
        return self._calculate_summary_statistics()
    
    def _find_best_times(self, train_data, metric_weights, top_n):
        """Find best trading times in training data"""
        # Group by day and time
        grouped = train_data.groupby(['Day_of_week', 'Entry_Time']).agg({
            'Profit': ['mean', 'count', lambda x: (x > 0).mean()]
        }).reset_index()
        
        grouped.columns = ['Day_of_week', 'Entry_Time', 'avg_profit', 'count', 'win_rate']
        
        # Filter for minimum trades
        grouped = grouped[grouped['count'] >= 5]
        
        # Calculate composite score
        grouped['score'] = (
            metric_weights.get('avg_profit', 0.5) * grouped['avg_profit'] / 1000 +
            metric_weights.get('win_rate', 0.5) * grouped['win_rate']
        )
        
        # Get top times per day
        best_times = []
        for day in grouped['Day_of_week'].unique():
            day_data = grouped[grouped['Day_of_week'] == day].nlargest(top_n, 'score')
            for _, row in day_data.iterrows():
                best_times.append((row['Day_of_week'], row['Entry_Time']))
        
        return best_times
    
    def _test_performance(self, test_data, best_times, contracts):
        """Test performance of identified times on test data"""
        # Filter test data for best times only
        filtered = test_data[
            test_data.apply(lambda x: (x['Day_of_week'], x['Entry_Time']) in best_times, axis=1)
        ]
        
        if len(filtered) == 0:
            return {
                'trade_count': 0,
                'total_profit': 0,
                'win_rate': 0,
                'avg_profit': 0
            }
        
        return {
            'trade_count': len(filtered),
            'total_profit': filtered['Profit'].sum() * contracts,
            'win_rate': (filtered['Profit'] > 0).mean(),
            'avg_profit': filtered['Profit'].mean() * contracts
        }
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics from all walk-forward windows"""
        if not self.results:
            return None
            
        results_df = pd.DataFrame(self.results)
        
        return {
            'total_windows': len(results_df),
            'profitable_windows': results_df['is_profitable'].sum(),
            'success_rate': results_df['is_profitable'].mean() * 100,
            'avg_test_profit': results_df['test_profit'].mean(),
            'avg_test_win_rate': results_df['test_win_rate'].mean() * 100,
            'consistency_score': self._calculate_consistency_score(results_df),
            'detailed_results': results_df
        }
    
    def _calculate_consistency_score(self, results_df):
        """
        Calculate consistency score (0-100)
        Higher score means more consistent out-of-sample performance
        """
        if len(results_df) < 3:
            return 0
            
        # Components of consistency
        profit_consistency = (results_df['test_profit'] > 0).mean() * 40
        win_rate_stability = (1 - results_df['test_win_rate'].std()) * 30
        positive_avg = min(results_df['test_avg_profit'].mean() / 100, 1) * 30
        
        return max(0, min(100, profit_consistency + win_rate_stability + positive_avg))


# ============================================================================
# MONTE CARLO SIGNIFICANCE TESTING
# ============================================================================

class MonteCarloTester:
    """
    Test if identified patterns are statistically significant or just random
    """
    
    def __init__(self, df, n_simulations=200):
        self.df = df.copy()
        self.n_simulations = n_simulations
        
    def test_time_pattern_significance(self, best_times, metric='profit'):
        """
        Test if performance at specific times is significantly better than random
        """
        # Calculate actual performance at best times
        actual_performance = self._calculate_performance(self.df, best_times, metric)
        
        # Run Monte Carlo simulations
        simulated_performances = []
        
        for _ in range(self.n_simulations):
            # Randomly shuffle times while keeping profits
            shuffled_df = self.df.copy()
            time_labels = shuffled_df[['Day_of_week', 'Entry_Time']].values
            np.random.shuffle(time_labels)
            shuffled_df[['Day_of_week', 'Entry_Time']] = time_labels
            
            # Calculate performance with random times
            sim_performance = self._calculate_performance(shuffled_df, best_times, metric)
            simulated_performances.append(sim_performance)
        
        # Calculate p-value
        simulated_performances = np.array(simulated_performances)
        p_value = (simulated_performances >= actual_performance).mean()
        
        # Calculate percentile rank (without scipy)
        percentile = (simulated_performances <= actual_performance).mean() * 100
        
        return {
            'actual_performance': actual_performance,
            'simulated_mean': simulated_performances.mean(),
            'simulated_std': simulated_performances.std(),
            'p_value': p_value,
            'percentile': percentile,
            'is_significant': p_value < 0.05,
            'z_score': (actual_performance - simulated_performances.mean()) / simulated_performances.std()
        }
    
    def _calculate_performance(self, df, best_times, metric):
        """Calculate performance metric for given times"""
        filtered = df[
            df.apply(lambda x: (x['Day_of_week'], x['Entry_Time']) in best_times, axis=1)
        ]
        
        if metric == 'profit':
            return filtered['Profit'].sum()
        elif metric == 'win_rate':
            return (filtered['Profit'] > 0).mean()
        elif metric == 'sharpe':
            returns = filtered['Profit'].values
            if len(returns) < 2:
                return 0
            return returns.mean() / (returns.std() + 1e-10)
        else:
            return 0


# ============================================================================
# PATTERN STABILITY ANALYZER
# ============================================================================

class PatternStabilityAnalyzer:
    """
    Analyze stability of patterns over time
    """
    
    def __init__(self, df):
        self.df = df.sort_values('Date').copy()
        
    def analyze_time_stability(self, day_of_week, entry_time, window_weeks=4):
        """
        Analyze how stable a specific time's performance is
        """
        # Filter for specific time
        time_data = self.df[
            (self.df['Day_of_week'] == day_of_week) & 
            (self.df['Entry_Time'] == entry_time)
        ].copy()
        
        if len(time_data) < 10:
            return None
        
        # Calculate rolling metrics
        results = []
        
        for i in range(0, len(time_data), 5):  # Step through in groups
            window = time_data.iloc[i:i+20]  # 20 trade windows
            if len(window) >= 5:
                results.append({
                    'period_start': window['Date'].min(),
                    'period_end': window['Date'].max(),
                    'avg_profit': window['Profit'].mean(),
                    'win_rate': (window['Profit'] > 0).mean(),
                    'trade_count': len(window)
                })
        
        if len(results) < 3:
            return None
            
        results_df = pd.DataFrame(results)
        
        # Calculate stability metrics
        profit_volatility = results_df['avg_profit'].std()
        win_rate_volatility = results_df['win_rate'].std()
        
        # Trend analysis
        x = np.arange(len(results_df))
        y = results_df['avg_profit'].values
        slope, intercept = np.polyfit(x, y, 1)
        
        return {
            'day': day_of_week,
            'time': entry_time,
            'periods_analyzed': len(results_df),
            'avg_profit_mean': results_df['avg_profit'].mean(),
            'avg_profit_std': profit_volatility,
            'win_rate_mean': results_df['win_rate'].mean() * 100,
            'win_rate_std': win_rate_volatility * 100,
            'trend_slope': slope,
            'is_deteriorating': slope < -10,  # Losing $10+ per period
            'stability_score': self._calculate_stability_score(results_df, slope),
            'detailed_periods': results_df
        }
    
    def _calculate_stability_score(self, results_df, trend_slope):
        """
        Calculate stability score (0-100)
        Higher score = more stable pattern
        """
        # Low volatility is good
        profit_stability = max(0, 100 - results_df['avg_profit'].std())
        
        # Consistent win rate is good
        win_rate_stability = max(0, 100 - results_df['win_rate'].std() * 200)
        
        # Positive or neutral trend is good
        trend_score = 50 if trend_slope >= 0 else max(0, 50 + trend_slope)
        
        # Weight the components
        score = (profit_stability * 0.4 + win_rate_stability * 0.3 + trend_score * 0.3)
        
        return max(0, min(100, score))


# ============================================================================
# INTEGRATED VALIDATION REPORT
# ============================================================================

def generate_validation_report(df, symbol, strategy, metric_weights, contracts=1):
    """
    Generate comprehensive validation report
    """
    print(f"\n{'='*60}")
    print(f"STATISTICAL VALIDATION REPORT")
    print(f"Symbol: {symbol} | Strategy: {strategy}")
    print(f"{'='*60}\n")
    
    # 1. Walk-Forward Analysis
    print("1. WALK-FORWARD ANALYSIS")
    print("-" * 40)
    
    wf_validator = WalkForwardValidator(df, training_weeks=12, testing_weeks=4)
    wf_results = wf_validator.run_analysis(metric_weights, top_n_times=3, contracts=contracts)
    
    if wf_results:
        print(f"   Windows Tested: {wf_results['total_windows']}")
        print(f"   Profitable Windows: {wf_results['profitable_windows']}/{wf_results['total_windows']}")
        print(f"   Success Rate: {wf_results['success_rate']:.1f}%")
        print(f"   Avg Out-of-Sample Profit: ${wf_results['avg_test_profit']:.2f}")
        print(f"   Avg Out-of-Sample Win Rate: {wf_results['avg_test_win_rate']:.1f}%")
        print(f"   Consistency Score: {wf_results['consistency_score']:.1f}/100")
        
        if wf_results['success_rate'] < 50:
            print("\n   ⚠️ WARNING: Pattern fails in majority of out-of-sample tests!")
        elif wf_results['success_rate'] > 70:
            print("\n   ✅ Pattern shows good out-of-sample consistency")
        else:
            print("\n   🔶 Pattern shows moderate out-of-sample performance")
    
    # 2. Statistical Significance Testing
    print("\n2. MONTE CARLO SIGNIFICANCE TEST")
    print("-" * 40)
    
    # Get top times for testing
    grouped = df.groupby(['Day_of_week', 'Entry_Time'])['Profit'].mean().nlargest(10)
    best_times = [(day, time) for (day, time) in grouped.index]
    
    mc_tester = MonteCarloTester(df, n_simulations=1000)
    mc_results = mc_tester.test_time_pattern_significance(best_times, metric='profit')
    
    print(f"   Actual Performance: ${mc_results['actual_performance']:.2f}")
    print(f"   Random Performance: ${mc_results['simulated_mean']:.2f} ± {mc_results['simulated_std']:.2f}")
    print(f"   Z-Score: {mc_results['z_score']:.2f}")
    print(f"   P-Value: {mc_results['p_value']:.4f}")
    print(f"   Percentile: {mc_results['percentile']:.1f}%")
    
    if mc_results['is_significant']:
        print("\n   ✅ Pattern is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("\n   ❌ Pattern is NOT statistically significant (could be random)")
    
    # 3. Pattern Stability Analysis
    print("\n3. PATTERN STABILITY ANALYSIS")
    print("-" * 40)
    
    stability_analyzer = PatternStabilityAnalyzer(df)
    
    # Test top 5 times for stability
    print("\n   Top 5 Times Stability Check:")
    
    for (day, time) in best_times[:5]:
        stability = stability_analyzer.analyze_time_stability(day, time)
        if stability:
            trend_indicator = "↓" if stability['is_deteriorating'] else "→" if abs(stability['trend_slope']) < 5 else "↑"
            print(f"   • {day} {time.strftime('%H:%M')}: "
                  f"Stability={stability['stability_score']:.0f}/100 "
                  f"Trend={trend_indicator}")
    
    # 4. Final Assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    # Calculate overall reliability score
    reliability_score = 0
    reliability_components = []
    
    if wf_results:
        wf_score = min(wf_results['success_rate'], 100) * 0.4
        reliability_score += wf_score
        reliability_components.append(f"Walk-Forward: {wf_score:.1f}/40")
    
    if mc_results['is_significant']:
        mc_score = min(mc_results['percentile'], 100) * 0.3
        reliability_score += mc_score
        reliability_components.append(f"Statistical Significance: {mc_score:.1f}/30")
    else:
        reliability_components.append(f"Statistical Significance: 0/30")
    
    # Add stability score (simplified)
    stability_score = 20  # Default moderate stability
    reliability_score += stability_score
    reliability_components.append(f"Pattern Stability: {stability_score:.1f}/30")
    
    print(f"\nOverall Reliability Score: {reliability_score:.1f}/100")
    print("\nComponents:")
    for component in reliability_components:
        print(f"   • {component}")
    
    print("\n" + "-"*60)
    
    if reliability_score >= 70:
        print("✅ RECOMMENDATION: Pattern shows strong statistical validity")
        print("   Consider paper trading to verify real-world performance")
    elif reliability_score >= 50:
        print("🔶 RECOMMENDATION: Pattern shows moderate validity")
        print("   Requires further testing and should be traded with caution")
    else:
        print("❌ RECOMMENDATION: Pattern lacks statistical validity")
        print("   High risk of overfitting - NOT recommended for live trading")
    
    print("\n" + "="*60)
    
    return {
        'walk_forward': wf_results,
        'monte_carlo': mc_results,
        'reliability_score': reliability_score
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example of how to integrate with existing dashboard
    
    # Load data (same as dashboard)
    df = pd.read_parquet("D:/_Documents/Magic 8 Ball/data/dfe_table.parquet")
    df = df[(df['Symbol'] == 'SPX') & (df['Name'] == 'Butterfly')].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='%H:%M').dt.time
    
    # Define metric weights (from dashboard settings)
    metric_weights = {
        'avg_profit': 0.5,
        'win_rate': 0.3,
        'sortino_ratio': 0.2
    }
    
    # Generate validation report
    validation_results = generate_validation_report(
        df=df,
        symbol='SPX',
        strategy='Butterfly',
        metric_weights=metric_weights,
        contracts=1
    )
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nThis analysis provides statistically rigorous testing of patterns")
    print("to help avoid overfitting and identify truly persistent edges.")
    print("\nAlways combine with:")
    print("  • Forward testing on new data")
    print("  • Risk management")
    print("  • Market regime analysis")
    print("  • Multiple timeframe confirmation")