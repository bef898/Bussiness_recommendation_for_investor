import pandas as pd
import numpy as np
from scipy import stats

class DataFrameProcessor:
    def __init__(self, df):
        self.df = df
    
    # Handle null values based on the percentage thresholds
    def handle_null_values(self):
        """Impute null values based on percentage of missing data."""
        total_rows = len(self.df)
        for col in self.df.columns:
            null_percentage = (self.df[col].isnull().sum() / total_rows) * 100
            
            if null_percentage > 0 and null_percentage <= 30:
                # Option 1: Impute with mean/median for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)  # Mean for numeric
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)  # Mode for categorical
            elif 30 < null_percentage <= 90:
                # Option 1: Impute with median for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)  # Median for numeric
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)  # Mode for categorical
            elif null_percentage > 90:
                # Option 1: Drop the column if too many nulls
                self.df.drop(columns=[col], inplace=True)

    # Handle outliers based on the Z-score method
    def handle_outliers(self, threshold=3):
        """Cap outliers based on Z-score thresholds."""
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].nunique() > 1:  # Ensure the column has variability
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = z_scores > threshold
                # Option 1: Cap extreme values at 95th percentile for high outliers
                upper_cap = self.df[col].quantile(0.95)
                self.df[col] = np.where(outliers, upper_cap, self.df[col])

    # Handle missing values explicitly marked (e.g., zeros or placeholders)
    def handle_missing_values(self):
        """Replace missing values like zero or placeholders with NaN and then impute."""
        # Option 1: If there's missing (non-NaN) values, replace them with NaN for clarity
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Assume zeros might be placeholder for missing values in numeric columns
                self.df[col].replace(0, np.nan, inplace=True)
                # After replacing, impute with mean/median
                self.df[col].fillna(self.df[col].mean(), inplace=True)  # Mean imputation
    
    # Combine all steps
    def clean_data(self):
        """Run all data cleaning steps in sequence."""
        self.handle_null_values()  # Step 1: Handle null values
        self.handle_outliers()  # Step 2: Handle outliers
        self.handle_missing_values()  # Step 3: Handle missing values
    
    def get_cleaned_data(self):
        """Return the cleaned DataFrame."""
        return self.df

# Example usage:
# df = pd.read_csv('your_data.csv')  # Replace with your actual DataFrame
# processor = DataFrameProcessor(df)
# processor.clean_data()  # Clean the data
# cleaned_df = processor.get_cleaned_data()  # Get the cleaned DataFrame
