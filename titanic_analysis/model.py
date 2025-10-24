"""
Model training module for Titanic analysis.
"""

import pandas
from sklearn.ensemble import RandomForestClassifier

def cat_to_num(dataset, cat_var, map_dict):
    """
    Map categorical value to numerical values
    
    Parameters:
    -----------
    dataset : pandas dataframe
        the dataset that contains categorical variables
    cat_var : string
        the name of the categorical variable
    map_dict : dictionary
        the distionary storing the mapping relationship between categorical variable and numerical variable
    
    Returns:
    --------
    dataset : pandas dataframe
        the dataset with categorical variable converted to numerical variable
    """

    dataset[cat_var] = dataset[cat_var].map(map_dict).astype(int)
    
    return dataset

def split_feature_target(dataset, predictors, target):
    """
    Split features and targets from a given dataset
    
    Parameters:
    -----------
    dataset : pandas dataframe
        the dataset that contains features and target variable
    predictors : list
        a list of the name of features
    target : string
        the name of the target variable
    
    Returns:
    --------
    data_X : pandas dataframe
        the dataframe that contains all features in the original dataset
    data_Y : numpy array
        the arrary that represent the target variable in the original dataset
    """
    data_X = dataset[predictors]
    data_Y = dataset[target].values

    return data_X, data_Y


def random_forest_prediction(train_x, train_y, valid_x, n_jobs=-1,random_state=42, criterion="gini", n_estimators=100, verbose=False):
    """
    Fit a random forest classifier and generate prediction for both training and validation data
    
    Parameters:
    -----------
    train_x : pandas dataframe
        the dataframe that contains all features in the training set
    train_y : numpy array
        the arrary that represent the target variable in the training set
    valid_x : pandas dataframe
        the dataframe that contains all features in the validation set
    cat_var : string
        the name of the categorical variable
    map_dict : dictionary
        the distionary storing the mapping relationship between categorical variable and numerical variable
    n_jobs : interger
        number of cpu cores used for model training
    random_state: interger
        the random seed set for model training
    criterion: string
        pins down criterion for deciding best split
    n_estimators: interger
        number of trees in the random forest
    verbose: boolean
        controls logging for model training
    
    Returns:
    --------
    preds_train : numpy arrary
        the array that contains prediction to each training set observation based on provided features
    preds_valid : numpy arrary
        the array that contains prediction to each validation set observation based on provided features
    """

    clf = RandomForestClassifier(n_jobs = n_jobs, 
                             random_state = random_state,
                             criterion = criterion,
                             n_estimators = n_estimators,
                             verbose = verbose)
    
    clf.fit(train_x, train_y)
    preds_train = clf.predict(train_x)
    preds_valid = clf.predict(valid_x)
    
    return preds_train, preds_valid