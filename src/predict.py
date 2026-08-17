# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
import joblib
from timeit import default_timer as timer
print ("libraries were imported successfully...")


def prediction():
    # prediction of the house class
    pipeline = joblib.load(r"D:\House_data\models\unsupervised_KMean.joblib")#loading pipeline 
    X_test= pd.read_csv(r"D:\House_data\train_test\X_test.csv")#loading X_test
    y_test = pd.read_csv(r"D:\House_data\train_test\y_test.csv")
    prediction = pipeline.predict(X_test)# prediction
    inertia = pipeline.named_steps['classifier'].inertia_#estimating the inertia
    print (f"{float(inertia):.2f}")
    # Classification of the features
    X_train = pd.read_csv(r"D:\House_data\train_test\X_train.csv")
    X_train_clustered = X_train.copy()
    X_train_clustered['cluster_labels'] = pipeline.named_steps['classifier'].labels_  
    # print (X_train_clustered.head())

    X_train_clustered=X_train_clustered.groupby("cluster_labels")[X_train_clustered.columns].median()
    
    cluster_lab = {
    0: "Spacious Single-Story Homes",
    1: "Large / Luxury Estates",
    2: "Compact Starter Homes",
    3: "Multi-Story Family Homes"
    }

    X_train_clustered.index = X_train_clustered.index.map(cluster_lab)
    print (f"Clusters of the house classes ...")
    print (X_train_clustered)

    # prediction of house price...
    grid_rf = joblib.load(r"D:\House_data\models\supervised_rf.joblib")
    print (f"best parameters: {grid_rf.best_params_}")
    print (f"best score: {grid_rf.best_score_}")
    # Model Random forest rf prediciton
    pred = grid_rf.predict(X_test)
    # Metrics
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)

    print(f"R2: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")

    print ("supervised model succeeded ...")

if __name__=="__main__":
    prediction()
    

    