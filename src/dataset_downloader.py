"""
Real Dataset Downloader and Processor for k-NN Analysis
This module downloads real datasets from various sources for comprehensive k-NN testing.
"""

import pandas as pd
import numpy as np
import requests
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def download_banknote_authentication_dataset():
    """Download Banknote Authentication dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    
    try:
        print("Downloading Banknote Authentication dataset...")
        
        # Column names for the dataset
        columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        
        df = pd.read_csv(url, names=columns)
        
        # Save to data folder
        data_path = '../data/banknote_authentication.csv'
        df.to_csv(data_path, index=False)
        
        print(f"Banknote Authentication dataset saved to {data_path}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error downloading Banknote Authentication dataset: {e}")
        return None


def download_seeds_dataset():
    """Download Seeds dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    
    try:
        print("Downloading Seeds dataset...")
        
        # Column names for the dataset
        columns = [
            'area', 'perimeter', 'compactness', 'length_kernel',
            'width_kernel', 'asymmetry_coefficient', 'length_groove', 'variety'
        ]
        
        # Read the data (tab-separated)
        df = pd.read_csv(url, sep='\t', names=columns, skipinitialspace=True)
        
        # Clean any potential whitespace issues
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Convert to numeric where needed
        for col in df.columns[:-1]:  # All except the last column (variety)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Convert variety to integer (should be 1, 2, 3)
        df['variety'] = df['variety'].astype(int)
        
        # Save to data folder
        data_path = '../data/seeds.csv'
        df.to_csv(data_path, index=False)
        
        print(f"Seeds dataset saved to {data_path}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error downloading Seeds dataset: {e}")
        return None


def download_glass_identification_dataset():
    """Download Glass Identification dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    
    try:
        print("Downloading Glass Identification dataset...")
        
        # Column names for the dataset
        columns = [
            'id', 'refractive_index', 'sodium', 'magnesium', 'aluminum',
            'silicon', 'potassium', 'calcium', 'barium', 'iron', 'glass_type'
        ]
        
        df = pd.read_csv(url, names=columns)
        
        # Drop the ID column as it's not useful for classification
        df = df.drop('id', axis=1)
        
        # Save to data folder
        data_path = '../data/glass_identification.csv'
        df.to_csv(data_path, index=False)
        
        print(f"Glass Identification dataset saved to {data_path}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error downloading Glass Identification dataset: {e}")
        return None


def download_ionosphere_dataset():
    """Download Ionosphere dataset from UCI repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    
    try:
        print("Downloading Ionosphere dataset...")
        
        # Column names for the dataset (34 features + 1 target)
        columns = [f'feature_{i+1}' for i in range(34)] + ['class']
        
        df = pd.read_csv(url, names=columns)
        
        # Convert class labels to binary (g=1, b=0)
        df['class'] = df['class'].map({'g': 1, 'b': 0})
        
        # Save to data folder
        data_path = '../data/ionosphere.csv'
        df.to_csv(data_path, index=False)
        
        print(f"Ionosphere dataset saved to {data_path}")
        print(f"Shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"Error downloading Ionosphere dataset: {e}")
        return None


def create_synthetic_high_dimensional_dataset():
    """Create synthetic high-dimensional dataset to demonstrate curse of dimensionality"""
    from sklearn.datasets import make_classification
    
    print("Creating synthetic high-dimensional dataset...")
    
    # Create dataset with many features
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to data folder
    data_path = '../data/synthetic_high_dim.csv'
    df.to_csv(data_path, index=False)
    
    print(f"Synthetic dataset saved to {data_path}")
    print(f"Shape: {df.shape}")
    
    return df


def prepare_dataset_for_knn(df, target_column, test_size=0.3):
    """
    Prepare dataset for k-NN analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    target_column : str
        Name of the target column
    test_size : float
        Proportion of test data
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Prepared and scaled data
    feature_names : list
        Names of features
    """
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Handle categorical variables if any
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)


def analyze_dataset_characteristics(df, target_column):
    """Analyze and display dataset characteristics"""
    print(f"\nDataset Analysis:")
    print("=" * 40)
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    print(f"Target column: {target_column}")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(df[target_column].value_counts())
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(df.describe())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found.")


def main():
    """Main function to download and prepare all datasets"""
    print("Real Dataset Preparation for k-NN Analysis")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Download datasets
    datasets = {}
    
    # Banknote Authentication
    banknote_df = download_banknote_authentication_dataset()
    if banknote_df is not None:
        datasets['banknote_authentication'] = banknote_df
        analyze_dataset_characteristics(banknote_df, 'class')
    
    print("\n" + "-" * 50)
    
    # Seeds Classification
    seeds_df = download_seeds_dataset()
    if seeds_df is not None:
        datasets['seeds'] = seeds_df
        analyze_dataset_characteristics(seeds_df, 'variety')
    
    print("\n" + "-" * 50)
    
    # Glass Identification
    glass_df = download_glass_identification_dataset()
    if glass_df is not None:
        datasets['glass_identification'] = glass_df
        analyze_dataset_characteristics(glass_df, 'glass_type')
    
    print("\n" + "-" * 50)
    
    # Ionosphere
    ionosphere_df = download_ionosphere_dataset()
    if ionosphere_df is not None:
        datasets['ionosphere'] = ionosphere_df
        analyze_dataset_characteristics(ionosphere_df, 'class')
    
    print("\n" + "-" * 50)
    
    # Synthetic High-Dimensional
    synthetic_df = create_synthetic_high_dimensional_dataset()
    datasets['synthetic_high_dim'] = synthetic_df
    analyze_dataset_characteristics(synthetic_df, 'target')
    
    print(f"\n\nDataset preparation completed!")
    print(f"Total datasets prepared: {len(datasets)}")
    
    return datasets


if __name__ == "__main__":
    datasets = main()
