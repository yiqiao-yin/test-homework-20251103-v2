#!/usr/bin/env python3
"""
Adult Dataset Feature Extraction and Analysis
Demonstrates how to work with X = adult.data.features
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_adult_dataset():
    """
    Load the Adult dataset and extract features
    This demonstrates the X = adult.data.features line in context
    """
    print("🔍 Fetching Adult Dataset from UCI ML Repository...")
    
    # Fetch the Adult dataset (ID = 2)
    adult = fetch_ucirepo(id=2)
    
    # Extract features and targets (this is your X = adult.data.features line!)
    X = adult.data.features  # This is the line you mentioned
    y = adult.data.targets
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Features shape: {X.shape}")
    print(f"🎯 Targets shape: {y.shape}")
    
    # Display dataset metadata
    print(f"\n📋 Dataset Information:")
    print(f"   Name: {adult.metadata.name}")
    print(f"   Description: {adult.metadata.description[:200]}...")
    
    return X, y, adult

def explore_features(X):
    """
    Explore the features extracted from adult.data.features
    """
    print("\n" + "="*60)
    print("🔍 FEATURE EXPLORATION")
    print("="*60)
    
    print(f"Total features: {X.shape[1]}")
    print(f"Total samples: {X.shape[0]}")
    
    print(f"\n📝 Feature names and types:")
    for i, col in enumerate(X.columns):
        dtype = X[col].dtype
        unique_vals = X[col].nunique()
        print(f"  {i+1:2d}. {col:15} | {str(dtype):8} | {unique_vals:3d} unique values")
    
    print(f"\n🔢 Numerical features:")
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   {numerical_features}")
    
    print(f"\n📝 Categorical features:")
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    print(f"   {categorical_features}")
    
    return numerical_features, categorical_features

def analyze_target(y):
    """
    Analyze the target variable
    """
    print("\n" + "="*60)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    target_col = y.columns[0]
    print(f"Target variable: '{target_col}'")
    
    # Value counts
    value_counts = y[target_col].value_counts()
    print(f"\nClass distribution:")
    for value, count in value_counts.items():
        percentage = (count / len(y)) * 100
        print(f"  {value}: {count:5d} samples ({percentage:.1f}%)")
    
    return target_col, value_counts

def preprocess_features(X, categorical_features):
    """
    Preprocess the features for machine learning
    """
    print("\n" + "="*60)
    print("🔧 FEATURE PREPROCESSING")
    print("="*60)
    
    # Create a copy to avoid modifying original data
    X_processed = X.copy()
    
    # Handle categorical variables with Label Encoding
    label_encoders = {}
    
    for feature in categorical_features:
        print(f"Encoding {feature}...")
        le = LabelEncoder()
        X_processed[feature] = le.fit_transform(X[feature].astype(str))
        label_encoders[feature] = le
        
        # Show the encoding
        unique_original = X[feature].unique()[:5]  # Show first 5
        unique_encoded = le.transform(unique_original.astype(str))
        print(f"  Sample encoding: {dict(zip(unique_original, unique_encoded))}")
    
    print(f"\n✅ Preprocessing completed!")
    print(f"   Original shape: {X.shape}")
    print(f"   Processed shape: {X_processed.shape}")
    
    return X_processed, label_encoders

def build_model(X_processed, y, target_col):
    """
    Build and evaluate a machine learning model
    """
    print("\n" + "="*60)
    print("🤖 MODEL BUILDING & EVALUATION")
    print("="*60)
    
    # Prepare target variable
    y_processed = y[target_col].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print(f"\n🌲 Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    print(f"\n📊 Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:15} | {row['importance']:.4f}")
    
    return rf_model, scaler, feature_importance

def visualize_data(X, y, target_col, categorical_features):
    """
    Create visualizations of the dataset
    """
    print("\n" + "="*60)
    print("📈 DATA VISUALIZATION")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Target distribution
    axes[0, 0].pie(y[target_col].value_counts().values, 
                   labels=y[target_col].value_counts().index,
                   autopct='%1.1f%%')
    axes[0, 0].set_title('Target Variable Distribution')
    
    # 2. Age distribution by target
    combined_data = pd.concat([X, y], axis=1)
    for target_value in y[target_col].unique():
        subset = combined_data[combined_data[target_col] == target_value]
        axes[0, 1].hist(subset['age'], alpha=0.6, label=target_value, bins=30)
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Age Distribution by Target')
    axes[0, 1].legend()
    
    # 3. Education levels
    education_counts = X['education'].value_counts()
    axes[1, 0].barh(range(len(education_counts)), education_counts.values)
    axes[1, 0].set_yticks(range(len(education_counts)))
    axes[1, 0].set_yticklabels(education_counts.index, fontsize=8)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_title('Education Level Distribution')
    
    # 4. Correlation heatmap (numerical features only)
    numerical_data = X.select_dtypes(include=[np.number])
    corr_matrix = numerical_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1, 1])
    axes[1, 1].set_title('Numerical Features Correlation')
    
    plt.tight_layout()
    plt.savefig('adult_dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Visualizations saved as 'adult_dataset_analysis.png'")
    
    return fig

def main():
    """
    Main function demonstrating the complete workflow
    """
    print("🚀 Adult Dataset Analysis Pipeline")
    print("="*60)
    print("This script demonstrates how to use: X = adult.data.features")
    print("="*60)
    
    try:
        # Step 1: Load data (includes the X = adult.data.features line)
        X, y, adult_dataset = load_adult_dataset()
        
        # Step 2: Explore features
        numerical_features, categorical_features = explore_features(X)
        
        # Step 3: Analyze target
        target_col, value_counts = analyze_target(y)
        
        # Step 4: Preprocess features
        X_processed, label_encoders = preprocess_features(X, categorical_features)
        
        # Step 5: Build and evaluate model
        model, scaler, feature_importance = build_model(X_processed, y, target_col)
        
        # Step 6: Create visualizations
        fig = visualize_data(X, y, target_col, categorical_features)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("Key outputs:")
        print("  - Model trained and evaluated")
        print("  - Feature importance calculated")
        print("  - Visualizations saved")
        print("  - Label encoders available for future use")
        
        return {
            'features': X,
            'targets': y,
            'processed_features': X_processed,
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print(f"❌ Error in analysis pipeline: {e}")
        return None

if __name__ == "__main__":
    # Run the complete analysis
    results = main()
    
    if results:
        print(f"\n🎉 Success! All results stored in 'results' dictionary")
        print(f"Access features with: results['features']")
        print(f"Access processed data with: results['processed_features']")
        print(f"Access trained model with: results['model']")
