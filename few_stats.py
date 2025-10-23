"""
Statistical analysis module for Titanic dataset.
Functions for analyzing missing data, frequencies, and unique values.
"""

import pandas as pd
import numpy as np


def create_missing_data_report(df, dataset_name="Dataset"):
    """
    Create a detailed missing data report with totals, percentages, and data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    dataset_name : str
        Name of the dataset for display
        
    Returns:
    --------
    pd.DataFrame
        Transposed dataframe with missing data statistics
    """
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    
    tt['Types'] = types
    df_missing = np.transpose(tt)
    
    print(f"\n=== Missing Data Report: {dataset_name} ===")
    return df_missing


def get_most_frequent_values(df, dataset_name="Dataset"):
    """
    Get the most frequent value for each column in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    dataset_name : str
        Name of the dataset for display
        
    Returns:
    --------
    pd.DataFrame
        Transposed dataframe with most frequent items, their frequency, and percentage
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    items = []
    vals = []
    
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(f"Error processing column {col}: {ex}")
            items.append(0)
            vals.append(0)
            continue
    
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    
    print(f"\n=== Most Frequent Values: {dataset_name} ===")
    return np.transpose(tt)


def get_unique_values_count(df, dataset_name="Dataset"):
    """
    Count unique values for each column in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to analyze
    dataset_name : str
        Name of the dataset for display
        
    Returns:
    --------
    pd.DataFrame
        Transposed dataframe with unique value counts
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    
    tt['Uniques'] = uniques
    
    print(f"\n=== Unique Values Count: {dataset_name} ===")
    return np.transpose(tt)