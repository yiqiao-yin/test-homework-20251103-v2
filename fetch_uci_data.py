#!/usr/bin/env python3
"""
UCI ML Repository Data Fetcher
Fetches a dataset from UCI ML Repository and saves it as CSV
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def fetch_and_save_dataset(dataset_id, filename=None):
    """
    Fetch a dataset from UCI ML Repository and save as CSV
    
    Args:
        dataset_id (int): UCI dataset ID
        filename (str, optional): Output filename. If None, uses dataset name
    
    Returns:
        str: Path to saved CSV file
    """
    try:
        print(f"Fetching dataset with ID: {dataset_id}")
        
        # Fetch dataset
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Get features and targets
        X = dataset.data.features
        y = dataset.data.targets
        
        # Combine features and targets into single dataframe
        if y is not None:
            data = pd.concat([X, y], axis=1)
        else:
            data = X
        
        # Generate filename if not provided
        if filename is None:
            dataset_name = dataset.metadata.name.replace(' ', '_').replace('/', '_')
            filename = f"{dataset_name}.csv"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Save to CSV
        data.to_csv(filename, index=False)
        
        print(f"Dataset information:")
        print(f"- Name: {dataset.metadata.name}")
        print(f"- Shape: {data.shape}")
        print(f"- Columns: {list(data.columns)}")
        print(f"- Missing values: {data.isnull().sum().sum()}")
        print(f"- Saved to: {filename}")
        
        # For Adult dataset, show target distribution
        if dataset_id == 2 and y is not None:
            print(f"\nTarget variable distribution:")
            target_col = y.columns[0]
            print(data[target_col].value_counts())
        
        return filename
        
    except Exception as e:
        print(f"Error fetching or saving dataset: {e}")
        return None

def analyze_adult_dataset(data_file):
    """
    Perform basic analysis on the Adult dataset
    
    Args:
        data_file (str): Path to the CSV file
    """
    try:
        df = pd.read_csv(data_file)
        
        print("\n" + "="*50)
        print("ADULT DATASET ANALYSIS")
        print("="*50)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Samples: {df.shape[0]}")
        
        # Basic info
        print("\nColumn Information:")
        print(df.dtypes)
        
        # Target variable analysis
        target_col = df.columns[-1]  # Assuming target is last column
        print(f"\nTarget Variable: '{target_col}'")
        print(df[target_col].value_counts())
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            print(missing[missing > 0])
        else:
            print(f"\nNo missing values found!")
            
        # Categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        print(f"\nCategorical Features ({len(categorical_cols)}):")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  - {col}: {unique_count} unique values")
            
        # Numerical features  
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        print(f"\nNumerical Features ({len(numerical_cols)}):")
        for col in numerical_cols:
            print(f"  - {col}: [{df[col].min()}, {df[col].max()}]")
            
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

def main():
    """Main function - fetches Adult dataset (ID=2)"""
    
    print("UCI ML Repository Data Fetcher")
    print("=" * 40)
    print("Fetching Adult Dataset (Census Income)")
    
    # Fetch Adult dataset
    dataset_id = 2
    saved_file = fetch_and_save_dataset(dataset_id)
    
    if saved_file:
        print(f"\nSuccess! Dataset saved to: {os.path.abspath(saved_file)}")
        
        # Perform analysis
        analyze_adult_dataset(saved_file)
        
        print(f"\n" + "="*50)
        print("NEXT STEPS:")
        print("="*50)
        print("1. Load the data: df = pd.read_csv('Adult.csv')")
        print("2. Handle categorical variables (encoding)")
        print("3. Split into train/test sets")
        print("4. Apply machine learning algorithms")
        print("5. Common algorithms for this dataset:")
        print("   - Logistic Regression")
        print("   - Random Forest")
        print("   - Gradient Boosting")
        print("   - SVM")
    else:
        print("\nFailed to fetch and save dataset.")

if __name__ == "__main__":
    main()
