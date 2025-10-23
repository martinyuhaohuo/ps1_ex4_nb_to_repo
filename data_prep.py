"""
Data preparation module for Titanic analysis.
"""

import pandas as pd
from pathlib import Path


def load_data(train_path="train.csv", test_path="test.csv"):
    """
    Load train and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Training and test dataframes
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def quick_glimpse(df, name="Dataset"):
    """
    Provide a quick glimpse of the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to inspect
    name : str
        Name of the dataset for display
        
    Returns:
    --------
    None
        Prints information about the dataframe
    """
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types and missing values:")
    print(df.info())
    print(f"\nSummary statistics:")
    print(df.describe())


def check_missing_values(df):
    """
    Check for missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to check
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values by column
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    missing_table = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    })
    
    return missing_table[missing_table['Missing Values'] > 0].sort_values(
        'Missing Values', ascending=False
    )