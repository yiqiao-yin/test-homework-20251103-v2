#!/usr/bin/env python3
"""
Comprehensive Pandas DataFrame Analysis Toolkit
Provides essential functions for data analysis and manipulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataFrameAnalyzer:
    """
    A comprehensive toolkit for pandas DataFrame analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze
        """
        self.df = df.copy()
        self.original_shape = df.shape
        
    def quick_overview(self) -> None:
        """Display a comprehensive overview of the DataFrame"""
        print("="*60)
        print("📊 DATAFRAME OVERVIEW")
        print("="*60)
        
        print(f"Shape: {self.df.shape} (rows × columns)")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Data types: {self.df.dtypes.value_counts().to_dict()}")
        
        # Missing values summary
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            print(f"Missing values: {total_missing} ({total_missing/len(self.df)*100:.1f}%)")
        else:
            print("Missing values: None ✅")
            
        # Duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Duplicate rows: {duplicates} ({duplicates/len(self.df)*100:.1f}%)")
        else:
            print("Duplicate rows: None ✅")
            
        print(f"\nFirst few rows:")
        print(self.df.head(3))
        
    def column_analysis(self) -> pd.DataFrame:
        """
        Comprehensive analysis of all columns
        
        Returns:
            pd.DataFrame: Summary statistics for each column
        """
        analysis = []
        
        for col in self.df.columns:
            col_data = {
                'column': col,
                'dtype': str(self.df[col].dtype),
                'non_null': self.df[col].count(),
                'null_count': self.df[col].isnull().sum(),
                'null_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique_values': self.df[col].nunique(),
                'unique_percentage': (self.df[col].nunique() / len(self.df)) * 100
            }
            
            if self.df[col].dtype in ['int64', 'float64']:
                col_data.update({
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std()
                })
            else:
                col_data.update({
                    'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'median': None,
                    'std': None
                })
                
            analysis.append(col_data)
            
        return pd.DataFrame(analysis)
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Detect outliers in numerical columns
        
        Args:
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            
        Returns:
            Dict: Outlier information for each numerical column
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            else:  # zscore
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_mask = z_scores > threshold
                
            outlier_count = outlier_mask.sum()
            outliers[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(self.df)) * 100,
                'indices': self.df[outlier_mask].index.tolist()
            }
            
        return outliers
    
    def correlation_analysis(self, method: str = 'pearson', threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze correlations between numerical variables
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            threshold (float): Minimum correlation threshold to highlight
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return pd.DataFrame()
            
        corr_matrix = self.df[numerical_cols].corr(method=method)
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val >= threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print(f"\n🔍 High correlations (|r| >= {threshold}):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
                print(f"  {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
        
        return corr_matrix
    
    def categorical_summary(self) -> Dict:
        """
        Analyze categorical variables
        
        Returns:
            Dict: Summary of categorical variables
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        summary = {}
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            summary[col] = {
                'unique_count': self.df[col].nunique(),
                'most_frequent': value_counts.index[0],
                'most_frequent_count': value_counts.iloc[0],
                'most_frequent_percentage': (value_counts.iloc[0] / len(self.df)) * 100,
                'top_5_values': value_counts.head().to_dict(),
                'cardinality': 'high' if self.df[col].nunique() > len(self.df) * 0.5 else 'low'
            }
            
        return summary

def load_and_analyze(file_path: str, **kwargs) -> DataFrameAnalyzer:
    """
    Load a CSV file and return a DataFrameAnalyzer instance
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv()
        
    Returns:
        DataFrameAnalyzer: Analyzer instance
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        print(f"✅ Successfully loaded {file_path}")
        print(f"   Shape: {df.shape}")
        return DataFrameAnalyzer(df)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                      df1_name: str = "DataFrame 1", 
                      df2_name: str = "DataFrame 2") -> None:
    """
    Compare two DataFrames
    
    Args:
        df1, df2 (pd.DataFrame): DataFrames to compare
        df1_name, df2_name (str): Names for the DataFrames
    """
    print("="*60)
    print(f"📊 DATAFRAME COMPARISON: {df1_name} vs {df2_name}")
    print("="*60)
    
    # Shape comparison
    print(f"Shape:")
    print(f"  {df1_name}: {df1.shape}")
    print(f"  {df2_name}: {df2.shape}")
    
    # Columns comparison
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1 & cols2
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1
    
    print(f"\nColumns:")
    print(f"  Common columns: {len(common_cols)}")
    print(f"  Only in {df1_name}: {len(only_in_df1)}")
    print(f"  Only in {df2_name}: {len(only_in_df2)}")
    
    if only_in_df1:
        print(f"    {list(only_in_df1)}")
    if only_in_df2:
        print(f"    {list(only_in_df2)}")

def create_sample_dataframe(dataset_type: str = "mixed") -> pd.DataFrame:
    """
    Create sample DataFrames for testing
    
    Args:
        dataset_type (str): Type of dataset ('mixed', 'numerical', 'categorical')
        
    Returns:
        pd.DataFrame: Sample DataFrame
    """
    np.random.seed(42)
    n_samples = 1000
    
    if dataset_type == "mixed":
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'salary': np.random.normal(50000, 15000, n_samples),
            'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations'], n_samples),
            'experience': np.random.randint(0, 40, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'rating': np.random.uniform(1, 5, n_samples),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples)
        }
    elif dataset_type == "numerical":
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(2, n_samples),
            'feature_3': np.random.uniform(-10, 10, n_samples),
            'feature_4': np.random.poisson(3, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
    else:  # categorical
        data = {
            'color': np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], n_samples),
            'size': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
            'quality': np.random.choice(['Good', 'Average', 'Poor'], n_samples),
            'brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D'], n_samples)
        }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    missing_col = np.random.choice(df.columns)
    df.loc[missing_indices, missing_col] = np.nan
    
    return df

# Quick utility functions
def quick_stats(df: pd.DataFrame) -> None:
    """Quick statistics overview"""
    analyzer = DataFrameAnalyzer(df)
    analyzer.quick_overview()

def find_missing(df: pd.DataFrame) -> pd.Series:
    """Find missing values in each column"""
    return df.isnull().sum().sort_values(ascending=False)

def memory_usage(df: pd.DataFrame) -> None:
    """Display memory usage information"""
    usage = df.memory_usage(deep=True)
    total_mb = usage.sum() / 1024**2
    print(f"Total memory usage: {total_mb:.2f} MB")
    print(f"Per column:")
    for col in df.columns:
        col_mb = usage[col] / 1024**2
        print(f"  {col}: {col_mb:.2f} MB")

if __name__ == "__main__":
    # Demo usage
    print("🐼 Pandas DataFrame Analysis Toolkit Demo")
    print("="*50)
    
    # Create sample data
    sample_df = create_sample_dataframe("mixed")
    print(f"Created sample dataset with shape: {sample_df.shape}")
    
    # Analyze
    analyzer = DataFrameAnalyzer(sample_df)
    analyzer.quick_overview()
    
    print("\n" + "="*50)
    print("Available methods:")
    methods = [method for method in dir(analyzer) if not method.startswith('_')]
    for method in methods:
        print(f"  - {method}()")
