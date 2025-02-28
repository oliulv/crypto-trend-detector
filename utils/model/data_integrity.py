import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings

# Suppress numpy warnings about NaN values
warnings.filterwarnings('ignore', category=RuntimeWarning, 
                       message='invalid value encountered in subtract')


@dataclass
class LeakageReport:
    """Container for data leakage analysis results"""
    has_leakage: bool
    details: Dict[str, List[str]]
    summary: str

class DataIntegrityChecker:
    def __init__(
        self,
        timestamp_col: str = 'timestamp',
        label_col: str = 'label',
        ignore_cols: Optional[List[str]] = None
    ):
        self.timestamp_col = timestamp_col
        self.label_col = label_col
        self.ignore_cols = ignore_cols or {
            'hour', 'minute', 'hour_sin', 'hour_cos',
            'day_of_week', 'is_weekend', 'asian_session', 'us_session'
        }
        self.report = None

    def check_temporal_integrity(self, df: pd.DataFrame) -> LeakageReport:
        """Check for basic temporal integrity issues"""
        issues = {      # Create dictionary to store issues
            'duplicate_timestamps': [],
            'ordering_issues': [],
            'irregular_intervals': [],
            'missing_values': []
        }
        
        # Check timestamp duplicates
        if df[self.timestamp_col].duplicated().any():
            issues['duplicate_timestamps'].append(
                f"Found {df[self.timestamp_col].duplicated().sum()} duplicate timestamps"
            )

        # Check chronological ordering
        if not df[self.timestamp_col].is_monotonic_increasing:
            issues['ordering_issues'].append("Timestamps are not strictly increasing")

        # Enhanced irregular intervals check
        time_diff = df[self.timestamp_col].diff() # Get the difference between timestamps

        # Skip the first row as it will have NaT (Not a Time) difference
        time_diff = time_diff[1:]

        expected_interval = time_diff.mode()[0] # Get the most common interval

        irregular_intervals = time_diff[time_diff != expected_interval] # Get irregular intervals compared to our expected interval
        
        if not irregular_intervals.empty:       # If there are irregular intervals
            issues['irregular_intervals'].append(
                f"Expected interval: {expected_interval}, found {len(irregular_intervals)} irregular intervals"
            )
            
            # Add details for the first 5 irregular intervals
            for idx in list(irregular_intervals.index)[:5]:
                current_timestamp = df.iloc[idx][self.timestamp_col]
                previous_timestamp = df.iloc[idx-1][self.timestamp_col]
                interval = irregular_intervals[idx]
                issues['irregular_intervals'].append(
                    f"  - At {current_timestamp}: Interval of {interval} (previous: {previous_timestamp})"
                )

        has_leakage = any(len(v) > 0 for v in issues.values()) # Check if there are any issues to set bool value to has_leakage
        return LeakageReport(
            has_leakage=has_leakage,
            details=issues,
            summary="Temporal integrity check complete"
        )

    def check_feature_leakage(self, df: pd.DataFrame) -> LeakageReport:
        """Check for feature-based data leakage"""
        issues = {
            'high_correlation': [],
            'constant_features': [],
            'missing_values': []  # Added missing_values category
        }

        feature_cols = [
            col for col in df.columns 
            if col not in [self.timestamp_col, self.label_col] 
            and col not in self.ignore_cols
        ]

        # First check for NaN values
        for feature in feature_cols:
            nan_count = df[feature].isna().sum()
            if nan_count > 0:
                percentage = (nan_count / len(df)) * 100
                issues['missing_values'].append(
                    f"Feature '{feature}' has {nan_count} missing values ({percentage:.2f}%)"
                )

        # Correlation check with NaN handling
        for feature in tqdm(feature_cols, desc="Checking correlations"):
            # Get valid data only
            valid_data = df[[feature, self.label_col]].dropna()
            if len(valid_data) > 0:
                try:
                    corr = valid_data[feature].corr(valid_data[self.label_col])
                    if pd.notna(corr) and abs(corr) > 0.9:
                        issues['high_correlation'].append(
                            f"Feature '{feature}' has {corr:.3f} correlation with target"
                        )
                except Exception as e:
                    print(f"\nWarning: Could not calculate correlation for {feature}: {str(e)}")

        has_leakage = any(len(v) > 0 for k, v in issues.items() if k != 'missing_values')
        return LeakageReport(
            has_leakage=has_leakage,  # NaN values don't count as leakage
            details=issues,
            summary="Feature leakage check complete"
        )

    def _parse_cooldown_from_filename(self, filepath: str) -> int:
        """Extract cooldown period from filename pattern"""
        import re
        
        # Look for pattern like '1h-pump' or '4h-pump'
        pattern = r'(\d+)h-pump'
        match = re.search(pattern, filepath)
        
        if match:
            hours = int(match.group(1))
            return hours * 60  # Convert hours to minutes
        
        # Default to 1 hour if pattern not found
        return 60

    def analyze_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """Run comprehensive data integrity analysis"""
        print("\n🔍 Starting comprehensive data integrity analysis...")
        
        # Calculate window sizes
        warm_up_period = 250  # 4 hours of minute data
        cool_down_period = self._parse_cooldown_from_filename(filepath)
        
        # Trim the dataset
        valid_data = df.iloc[warm_up_period:-cool_down_period].copy()
        print(f"\nℹ️ Analyzing data after {warm_up_period} minute warm-up period")
        print(f"ℹ️ Using {cool_down_period} minute cool-down period (from filename)")
        print(f"ℹ️ Original dataset size: {len(df)} rows")
        print(f"ℹ️ Analysis dataset size: {len(valid_data)} rows")
        
        temporal_report = self.check_temporal_integrity(valid_data)
        feature_report = self.check_feature_leakage(valid_data)
        
        # Combine reports
        all_issues = {
            **temporal_report.details,
            **feature_report.details
        }
        
        has_any_issues = temporal_report.has_leakage or feature_report.has_leakage  # Check if there are any issues
        
        # Store report
        self.report = LeakageReport(
            has_leakage=has_any_issues,
            details=all_issues,
            summary=self._generate_summary(all_issues)
        )
        
        # Print report
        self._print_report()

    def _generate_summary(self, issues: Dict[str, List[str]]) -> str:
        """Generate a summary of all issues found"""
        total_issues = sum(len(v) for v in issues.values())
        return f"Found {total_issues} potential data integrity issues"

    def _print_report(self) -> None:
        """Print the analysis report"""
        if not self.report:
            print("No analysis has been run yet.")
            return

        print("\n📊 Data Integrity Analysis Report")
        print("=" * 40)
        print(f"\nSummary: {self.report.summary}")
        
        # First print missing values as information, not errors
        if self.report.details.get('missing_values'):
            print("\nMissing Values (Information):")
            for issue in self.report.details['missing_values']:
                print(f"ℹ️  {issue}")

        # Then print actual issues
        print("\nDetailed Findings:")
        for category, issues in self.report.details.items():
            if issues and category != 'missing_values':  # Skip missing values here
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issues:
                    print(f"❌ {issue}")

        if not self.report.has_leakage:
            print("\n✅ No critical data integrity issues detected!")

def main():
    """Script entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Check data integrity and leakage")    # Create an argument parser
    parser.add_argument("data_path", help="Path to the CSV file to analyze")            # Add an argument for the data path
    args = parser.parse_args()                                                          # Parse the arguments         

    df = pd.read_csv(args.data_path, parse_dates=['timestamp'])
    checker = DataIntegrityChecker()
    checker.analyze_dataset(df, args.data_path)

if __name__ == "__main__":
    main()