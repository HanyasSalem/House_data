import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


def eda_house_data(df, target):
    """
    Perform exploratory data analysis on the house data.

    Parameters:
    df (pd.DataFrame): The house data DataFrame.
    target (str): The name of the target variable.

    Returns:
    None
    """
    # Data exploration
    print(f"Data set: house data for this project\n rows: {df.shape[0]}\ncolumns: {df.shape[1]}")
    print("-" * 100)
    # Numerical and categorical features
    num_features=df.select_dtypes(include=[np.number]).columns.tolist()
    cat_features=df.select_dtypes(include=[object]).columns.tolist()
    
    print(f"House data demoenstration\n{df.head()}")
    print(f"House dataset info:\n{df.info()}")
    print(f"numerical features:\n*{num_features}\n")
    print(f"categorical features:\n*{cat_features}\n")
    print("_" * 100)
    print(f"House dataset describe:\n{(df.describe().T).round(2)}")
    print(f"House dataset null values:\n{df.isnull().sum()}")

    # Data insight
    print(f"House dataset insights:")
    col_zero = df.columns[(df == 0).any()]
    for col in col_zero:
        df_zero = df[df[col] == 0]
        df_zero_pct = round((df_zero.shape[0] / df.shape[0]) * 100, 2)
        print(f"Number of zero values in {col}: {df_zero.shape[0]} - ({df_zero_pct}%)")
    
    # Unique values
    print(f"Unique values in each column:")
    for col in df.columns:
        unique_values = df[col].nunique()
        if col in cat_features:
            col_value_count = df[col].value_counts()
            print(f"{col}:\n unique values: {unique_values}\n Value counts: {col_value_count}")
        else:
            print(f"{col}:\n unique values: {unique_values}")

    # Visualize the frequency distribution of the target variable
    house_data_target = df[target]
    df = df.drop(columns=[target])  # Drop the target column from the DataFrame for correlation analysis
    plt.figure(figsize=(15, 8))
    sns.histplot(house_data_target, bins=30, kde=True)
    plt.title(f'Frequency Distribution of {target}')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.show()

    # Visualize the distribution of numerical features
    # Detect outliers using Z-score method
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_features:
        house_data_col = df[col].dropna()
        mask = np.abs(zscore(house_data_col)) > 3
        house_data_outliers = house_data_col[mask].tolist()
        print(f"Column: {col}")
        print(f"Total Outliers Found: {len(house_data_outliers)}")
        print(f"Outliers: {house_data_outliers}\n")

    # Visualize boxplots and histograms for numerical features
    for col in num_features:
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.boxplot(df[col].dropna(), patch_artist=True, boxprops=dict(facecolor="green"))
            plt.title(f"Boxplot of house data: {col}")
            plt.xlabel(f"{col} values")
            plt.ylabel(f"{col} values")
            plt.subplot(1, 2, 2)
            sns.histplot(df[col].dropna(), kde=True, color="blue", bins=30)
            plt.title(f"Histogram of house data: {col}")
            plt.ylabel(f"Frequency of {col} values")
            plt.xlabel(f"{col} values")
            plt.tight_layout()
            plt.show()
    
    # Visualize the distribution of categorical features
    for col in cat_features:
        plt.figure(figsize=(12, 8))
        sns.countplot(x=col, data=df, palette="Set2")
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


    # Visualize correlations between features
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_features) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[num_features].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix of house data features')
        plt.show()
    else:
        print('Not enough numeric features for a correlation matrix.')
    
    # Visualize of scatter plots for numerical features against the target variable
    for col in num_features:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df[col], y=house_data_target, color='red', alpha=0.5)
        plt.title(f'Scatter Plot of {col} vs {target}')
        plt.xlabel(col)
        plt.ylabel(target)
        plt.show()


if __name__ =="__main__":
    house_data = pd.read_csv(r'D:\House_data\raw_data\house_data.csv')
    house_data = house_data.drop(columns=['id', 'zipcode', 'lat', 'long',"date"])
    eda_house_data(house_data, target='price')
    print("EDA completed successfully.")