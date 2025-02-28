import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import warnings
from tqdm import tqdm

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
        issues = {
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

        # Check time intervals
        time_diff = df[self.timestamp_col].diff().dt.total_seconds()
        irregular_intervals = time_diff[time_diff != time_diff.mode()[0]]
        if not irregular_intervals.empty:
            issues['irregular_intervals'].append(
                f"Found {len(irregular_intervals)} irregular time intervals"
            )

        has_leakage = any(len(v) > 0 for v in issues.values())
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
            'future_leakage': []
        }

        feature_cols = [
            col for col in df.columns 
            if col not in [self.timestamp_col, self.label_col] 
            and col not in self.ignore_cols
        ]

        # Correlation check
        for feature in tqdm(feature_cols, desc="Checking correlations"):
            corr = df[feature].corr(df[self.label_col])
            if abs(corr) > 0.9:
                issues['high_correlation'].append(
                    f"Feature '{feature}' has {corr:.3f} correlation with target"
                )

        # Future information leakage check
        for feature in tqdm(feature_cols, desc="Checking for future information"):
            rolling_mean = df[feature].rolling(window=3).mean()
            if rolling_mean.corr(df[self.label_col]) > 0.9:
                issues['future_leakage'].append(
                    f"Feature '{feature}' shows signs of future information usage"
                )

        has_leakage = any(len(v) > 0 for v in issues.values())
        return LeakageReport(
            has_leakage=has_leakage,
            details=issues,
            summary="Feature leakage check complete"
        )

    def analyze_dataset(self, df: pd.DataFrame) -> None:
        """Run comprehensive data integrity analysis"""
        print("\n🔍 Starting comprehensive data integrity analysis...")
        
        temporal_report = self.check_temporal_integrity(df)
        feature_report = self.check_feature_leakage(df)
        
        # Combine reports
        all_issues = {
            **temporal_report.details,
            **feature_report.details
        }
        
        has_any_issues = temporal_report.has_leakage or feature_report.has_leakage
        
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
        
        if not self.report.has_leakage:
            print("\n✅ No data integrity issues detected!")
            return

        print("\nDetailed Findings:")
        for category, issues in self.report.details.items():
            if issues:
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issues:
                    print(f"❌ {issue}")

def main():
    """Script entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Check data integrity and leakage")
    parser.add_argument("data_path", help="Path to the CSV file to analyze")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, parse_dates=['timestamp'])
    checker = DataIntegrityChecker()
    checker.analyze_dataset(df)

if __name__ == "__main__":
    main()