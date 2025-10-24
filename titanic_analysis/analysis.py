import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_data(df1, df2):
    df_new = pd.concat([df1, df2], axis=0)
    df_new["set"] = "train"
    df_new.loc[df_new.Survived.isna(), "set"] = "test"
    df_new = df_new.reset_index(drop=True)
    return df_new

def family_size(df):
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df

def add_age_cat(df):
    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4
    return df

def plot_distribution_pairs(distcol, huecol, df):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(df[huecol].unique()):
        g = sns.histplot(df.loc[df[huecol]==h, distcol], 
                                    ax=ax, 
                                    label=h)
    ax.set_title(f"Number of passengers / {distcol} and {huecol}")
    g.legend()
    plt.show()
    
def plot_count_pairs(column, df, huevar="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=column, data=df, hue=huevar)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {column}")
    plt.show() 
    
def add_age_cat(df):
    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4
    return df

def fare_interval(df):
    df['Fare Interval'] = 0.0
    df.loc[ df['Fare'] < 7.9, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] < 14.453), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] < 31), 'Fare Interval']   = 2
    df.loc[ df['Fare'] > 31.001, 'Fare Interval'] = 3
    return df

def sexandpclass(df):
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df

def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")
        
def apply_parsed_names(df):
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df
    
def prepare(df1, df2):
    for dataset in [df1, df2]:
        dataset["Titles"] = dataset["Title"]
    return dataset
            
def title_classification(df1, df2):
    for dataset in [df1, df2]:
        #unify `Miss`
        dataset['Titles'] = dataset['Titles'].replace('Mlle.', 'Miss.')
        dataset['Titles'] = dataset['Titles'].replace('Ms.', 'Miss.')
        #unify `Mrs`
        dataset['Titles'] = dataset['Titles'].replace('Mme.', 'Mrs.')
        # unify Rare
        dataset['Titles'] = dataset['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.',\
        'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
        return dataset
        
def title_sex_gb(df):
    df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()
        
    