#!/usr/bin/env python3
"""
Demonstrate adult.variables attribute
Shows what print(adult.variables) outputs for the Adult dataset
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

def show_adult_variables():
    """
    Fetch Adult dataset and show the variables attribute
    This demonstrates what print(adult.variables) shows
    """
    print("🔍 Fetching Adult Dataset from UCI ML Repository...")
    
    try:
        # Fetch the Adult dataset (ID = 2)
        adult = fetch_ucirepo(id=2)
        
        print("✅ Dataset loaded successfully!")
        print("\n" + "="*60)
        print("📋 ADULT.VARIABLES OUTPUT")
        print("="*60)
        
        # This is what print(adult.variables) shows
        print(adult.variables)
        
        print("\n" + "="*60)
        print("📊 ADDITIONAL DATASET INFO")
        print("="*60)
        
        # Show other useful attributes
        print(f"\nDataset Name: {adult.metadata.name}")
        print(f"Dataset Description: {adult.metadata.description[:200]}...")
        
        # Show features and targets shapes
        print(f"\nFeatures Shape: {adult.data.features.shape}")
        print(f"Targets Shape: {adult.data.targets.shape}")
        
        # Show feature names
        print(f"\nFeature Names:")
        for i, col in enumerate(adult.data.features.columns, 1):
            print(f"  {i:2d}. {col}")
            
        # Show target names
        print(f"\nTarget Names:")
        for i, col in enumerate(adult.data.targets.columns, 1):
            print(f"  {i:2d}. {col}")
            
        print("\n" + "="*60)
        print("🔍 VARIABLES DATAFRAME STRUCTURE")
        print("="*60)
        
        # Show the structure of the variables DataFrame
        variables_df = adult.variables
        print(f"Variables DataFrame Shape: {variables_df.shape}")
        print(f"\nVariables DataFrame Columns: {list(variables_df.columns)}")
        print(f"\nFirst few rows of variables:")
        print(variables_df.head())
        
        print(f"\nData types in variables DataFrame:")
        print(variables_df.dtypes)
        
        # Show specific information about each variable
        print("\n" + "="*60)
        print("📋 DETAILED VARIABLE INFORMATION")
        print("="*60)
        
        for idx, row in variables_df.iterrows():
            print(f"\n{idx + 1}. Variable: {row['name']}")
            print(f"   Role: {row['role']}")
            print(f"   Type: {row['type']}")
            if pd.notna(row['demographic']):
                print(f"   Demographic: {row['demographic']}")
            if pd.notna(row['description']):
                print(f"   Description: {row['description']}")
            if pd.notna(row['units']):
                print(f"   Units: {row['units']}")
            if pd.notna(row['missing_values']):
                print(f"   Missing Values: {row['missing_values']}")
        
        return adult
        
    except Exception as e:
        print(f"❌ Error fetching Adult dataset: {e}")
        return None

def compare_variables_with_data(adult):
    """
    Compare the variables metadata with actual data
    """
    if adult is None:
        return
        
    print("\n" + "="*60)
    print("🔍 VARIABLES vs ACTUAL DATA COMPARISON")
    print("="*60)
    
    variables_df = adult.variables
    features_df = adult.data.features
    targets_df = adult.data.targets
    
    # Compare feature names
    variable_names = variables_df[variables_df['role'] == 'Feature']['name'].tolist()
    actual_feature_names = features_df.columns.tolist()
    
    print(f"Variables metadata shows {len(variable_names)} features")
    print(f"Actual features DataFrame has {len(actual_feature_names)} features")
    
    # Check for differences
    missing_in_metadata = set(actual_feature_names) - set(variable_names)
    missing_in_data = set(variable_names) - set(actual_feature_names)
    
    if missing_in_metadata:
        print(f"Features in data but not in metadata: {missing_in_metadata}")
    if missing_in_data:
        print(f"Features in metadata but not in data: {missing_in_data}")
    if not missing_in_metadata and not missing_in_data:
        print("✅ All feature names match between metadata and data")
    
    # Show target comparison
    target_names_metadata = variables_df[variables_df['role'] == 'Target']['name'].tolist()
    actual_target_names = targets_df.columns.tolist()
    
    print(f"\nTarget in metadata: {target_names_metadata}")
    print(f"Target in data: {actual_target_names}")

def main():
    """
    Main function - demonstrates adult.variables
    """
    print("🚀 Adult Dataset Variables Demonstration")
    print("="*60)
    print("This script shows what print(adult.variables) outputs")
    print("="*60)
    
    # Show variables
    adult = show_adult_variables()
    
    # Compare with actual data
    compare_variables_with_data(adult)
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*60)
    print("Key takeaways:")
    print("- adult.variables contains metadata about each variable")
    print("- It shows variable names, roles, types, and descriptions")
    print("- This is useful for understanding the dataset structure")
    print("- Use adult.data.features for actual feature data")
    print("- Use adult.data.targets for actual target data")

if __name__ == "__main__":
    main()
