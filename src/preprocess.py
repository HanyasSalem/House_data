import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
print ("Libraries imported successfully...")


# Data Preprocessing
# Date preprocessing function
def preprocess_dates(df, date_column):
    """
    Preprocess date columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing date column.
    date_column (str): Name of the column to be converted to datetime.
    
    Returns:
    pd.DataFrame: DataFrame with preprocessed date columns.
    """
    # data preprocessing
    new_df = df.copy()
    new_df[date_column] = pd.to_datetime(new_df[date_column], format='%Y%m%dT%H%M%S', errors='coerce')
    new_df['year'] = new_df[date_column].dt.year
    new_df['month'] = new_df[date_column].dt.month
    new_df['quartile'] = new_df[date_column].dt.quarter
    new_df['day'] = new_df[date_column].dt.day
    new_df['day_of_week'] = new_df[date_column].dt.dayofweek
    new_df[date_column] = new_df[date_column].dt.date
    # print(new_df[[date_column, 'year', 'month', 'quartile', 'day', 'day_of_week']].head(10))
    # print(f"Column '{date_column}' converted to datetime.")
    return new_df



# flag zero columns 
def flag_zero_values(df):
    """
    This function takes a DataFrame as input and creates a new column for each existing column that contains zero values. 
    The new column will have a flag (1 or 0) indicating whether the corresponding value in the original column is zero or not.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to process.
    
    Returns:
    pd.DataFrame: The modified DataFrame with new flag columns added.
    """
    for col in df.columns:
        if (df[col] == 0).any():
            mask = np.where(df[col]==0,0,1)
            df[f"{col}_flag"] = pd.Series(mask)
    return df



# columns validation
def valid_col (df,*cols):
    missing_col = [col for col in cols if col not in df.columns]
    if missing_col:
        print (f"missing columns: {missing_col}")
        
    return [col for col in cols if col in df.columns]

def feature_engineer (df, **featuers):
    '''
    The function takes the df and a dictionary containing the features to be engineered and the calculation process
    '''
    # working_cols = valid_col(df,*cols)
    for new_col, attrib in featuers.items():
        operator = attrib['math']
        col1,col2 = attrib['cols']
        working_cols = valid_col(df,col1,col2)
        
        if operator == 'adding':
            df[new_col] = df[working_cols].sum(axis=1)
        elif operator =="subt":
            df[new_col] =(df[working_cols].iloc[:,0] - df[working_cols].iloc[:,1:].sum(axis=1)).abs()
        elif operator == "mult":
            df[new_col] = df[working_cols].prod(axis=1)
        elif operator == "div":
            num= df[working_cols[0]]
            denom = df[working_cols[1]].replace(0,np.nan)
            df[new_col] = (num / denom).round(2)
                       
        else:
            print ('"No process was identified ... \n for math="adding","subt","mult","div"')
            print("operator value:", repr(operator))


    return df

        
def data_type_to_category (df,*category):
    '''
    A preprocessing function : change columns to category for regression models

    Parameters:
    df (df.DataFrame): The dataframe in question.
    category: a list of columns that should be of catergoy data type.

    Attribute: valid_col () external function

    Returns:
    pd.DataFrame with change of the column data types to categorical 
    '''
    categorical_cols = valid_col(df,*category)
    df[categorical_cols] = df[categorical_cols].apply(pd.Categorical)
    return df

def data_type_to_number (df, *numbers):
    '''
    A preprocessing function : change columns to numerical for regression models

    Parameters:
    df (df.DataFrame): The dataframe in question.
    numbers: a list of columns that should be of numerical data type.

    Attribute: valid_col () external function

    Returns:
    pd.DataFrame with change of the column data types to numerical 
    '''
    number_cols = valid_col(df,*numbers)
    df[number_cols] = df[number_cols].apply(pd.to_numeric,errors="coerce")
    return df


# Return the max and min price 
def get_max_min(df, price, target):
    max_raw= df.loc[df[price].idxmax()]
    min_raw = df.loc[df[price].idxmin()]
    return max_raw[target], max_raw[price],min_raw[target],min_raw[price]

# Return the topmost and lowermost price
def get_topmost_lowermost (df, price, n=3):
    topmost_price = df.nlargest(n,price)
    lowermost_price = df.nsmallest(n,price)
    return topmost_price, lowermost_price
    

# print ('functions work correctly ....')

if __name__ == "__main__":
    # import house_data
    house_data = pd.read_csv(r'D:\House_data\raw_data\house_data.csv')
    print (f" original house data dim: {house_data.shape}\n")
    # import feature mapping
    # with open (r"D:\House_data\src\feature_eng_map.json","r") as f:
    #     feature_mapping = json.load(f)
    # # preprocess data: date and feature engineering
    house_eng_data = preprocess_dates(house_data, 'date')
    # house_eng_data = flag_zero_values(house_eng_data)
    # house_eng_data = feature_engineer(house_eng_data, **feature_mapping)
    # # cleaning yr_renovation and yr_since_renovation columns
    # correcting the misscalulation on houses not renovated
    # house_eng_data['yr_since_renovation']= np.where(house_eng_data['yr_renovated_flag']==1,
    #                                                 house_eng_data['year']-house_eng_data['yr_renovated'],
    #                                                 np.nan)
    
    # fixing the negative values in yr_since_renovation column
    # house_eng_data.loc[house_eng_data['yr_since_renovation'] < 0, 'yr_since_renovation'] = np.nan
    print (f"engineered house data dim: {house_eng_data.shape}\n")
    # Engineering bedrooms
    # Dropping houses with bedrooms 0 : 13 records
    # Calculating rooms_per_floor with no np.nan
    house_eng_data = house_eng_data[house_eng_data['bedrooms'] != 0].copy()
    # house_eng_data = feature_engineer(
	#     house_eng_data,\
	#     **{'rooms_per_floor': {'cols': ["total_rooms", "floors"], 'math': 'div'}})
    # print (f"engineered house data after dropping 13 records:\n")
    print (f"{house_eng_data.shape}")
    print ("Data preprocessing completed successfully...")
    # house_eng_data.to_csv(r'D:\House_data\raw_data\house_eng_data.csv',index=False)
    print('mission done ...')
    
