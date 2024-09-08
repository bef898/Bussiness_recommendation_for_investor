import pandas as pd
import numpy as np
from scipy import stats

class DataFrameCleaner:
    def __init__(self, df):
        self.df = df
    
    def null_percentage(self):
        """Returns a Series with the percentage of null values per column."""
        null_counts = self.df.isnull().sum()
        total_rows = len(self.df)
        null_percentage = (null_counts / total_rows) * 100
        return null_percentage
    
    def outlier_percentage(self, threshold=3):
        """Returns a Series with the percentage of outliers per column using the Z-score method."""
        outlier_counts = pd.Series(0, index=self.df.select_dtypes(include=[np.number]).columns)
        total_rows = len(self.df)
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = (z_scores > threshold)
            outlier_counts[col] = outliers.sum()
        
        outlier_percentage = (outlier_counts / total_rows) * 100
        return outlier_percentage
    
    def missing_values(self):
        """Returns a Series with the count of missing values per column."""
        return self.df.isnull().sum()

    def display_summary(self, threshold=3):
        """Displays a table with null values, outliers, and missing values."""
        null_percentage = self.null_percentage()
        outlier_percentage = self.outlier_percentage(threshold=threshold)
        missing_values = self.missing_values()
        
        summary_df = pd.DataFrame({
            'Null Percentage (%)': null_percentage,
            'Outlier Percentage (%)': outlier_percentage,
            'Missing Values Count': missing_values
        })
        
        return summary_df

# Example usage:
# df = pd.read_csv('your_data.csv')
# cleaner = DataFrameCleaner(df)
# summary = cleaner.display_summary(threshold=3)
# print(summary)
