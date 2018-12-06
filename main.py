import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

def print_value_range(df):
    column_names = df.columns
    all_vals = []
    for label in column_names:
        z = set(df[label])
        all_vals.append(z)
    prints(all_vals)
 
if __name__ == '__main__':
 
    # Labels for columns in our spreadsheet that we can index the data with
    labels = [
        'TIMESTAMP',
        'GO',
        'LOCATION',
        'COLLEGE_NEAR',
        'NUM_PARENTS',
        'STATUS',
        'INCOME',
        'NUM_SIBLINGS',
        'COLLEGE_SIBLINGS',
        'PARENT_EDU',
        'RACE',
        'GENDER',
        'GPA',
        'HS_TYPE',
        'BIRTH',
        'HS_JOB',
        'COLLEGE_AGE'
    ]
    
    # Read the csv into a Pandas DataFrame
    df = pd.read_csv('./data.csv', skiprows=1)

    # Apply label fix
    df.columns = labels

    # Drop rows without a yes/no for college attendance
    df = df[pd.notnull(df['GO'])]

    # Drop redundant columns
    df.drop(columns=['TIMESTAMP', 'COLLEGE_AGE', 'INCOME', 'COLLEGE_NEAR', 'NUM_SIBLINGS'], inplace=True)

    """
    Columns after drop
    ['GO', 'LOCATION', 'NUM_PARENTS', 'STATUS', 'COLLEGE_SIBLINGS',
       'PARENT_EDU', 'RACE', 'GENDER', 'GPA', 'HS_TYPE', 'BIRTH', 'HS_JOB']
    """
    
    # Partition our dataset between training and test data
    df_train = df.head(30)
    df_test = df.tail(9)

    # Transform categorical data into a numerical representation
    df_dummies = pd.get_dummies(df)
    df_train_dummies = pd.get_dummies(df_train)
    df_test_dummies = pd.get_dummies(df_test)

    # Fix dummy dataframes to add missing categories from quantitization
    df_cols = df_dummies.columns
    df_train_cols = df_train_dummies.columns
    df_test_cols = df_test_dummies.columns
    
    missing_training_col = []
    missing_test_col = []

    for col in df_cols:
        if col not in df_train_cols:
            missing_training_col.append(col)
        if col not in df_test_cols:
            missing_test_col.append(col)

    missing = missing_test_col + missing_training_col

    for col in missing_training_col:
        df_train_dummies[col] = 0

    for col in missing_test_col:
        df_test_dummies[col] = 0

    for miss in missing:
        try:
            df_train_dummies.drop(columns=[miss], inplace=True)
        except: pass
        try:
            df_test_dummies.drop(columns=[miss], inplace=True)
        except: pass

    # Create a Linear SVC with the numerical data and class label (College or not)
    clf = LinearSVC()
    clf.fit(df_train_dummies, df_train_dummies['GO_Yes'])

    # AI Voodoo Magic
    results = clf.predict(df_test_dummies)

    # Output results
    for i, (response) in enumerate(df_test['GO']):
        if results[i] == 1:
            predict = 'Yes'
        if results[i] == 0:
            predict = 'No'
        print('Individual', i + 1, 'responded:', response.upper(), '\t', 'AI:', predict.upper())
