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
def main ():
    # Importing data engineered
    house_eng_data = pd.read_csv(r"D:\House_data\raw_data\house_eng_data.csv")

    # Selected using: duplicate check (exact match test), relevance check
    # (Pearson correlation), overlap check (pairwise correlation + VIF),
    # consistency check (variance/frequency test), and holdout R2
    # validation (train/test split) confirming distance_to_downtown +
    # zip_mean_price preserve the signal lost by dropping lat/long/zipcode.
    Features_selected = [
        'sqft_living',
        'sqft_basement',
        'sqft_lot',
        'sqft_living15',
        'sqft_lot15',
        'bedrooms',
        'bathrooms',
        'floors',
        'grade',
        'condition',
        'waterfront',
        'view',
        'house_age',
        'year',
        'yr_renovated_flag',
        'distance_to_downtown',   
        'zip_price_means',  
    ]
    with open(r"D:\House_data\src\feature_selected.txt",'w') as f:
        f.write('\n'.join(Features_selected)) # saving the selected features as a list

    target="price"
    house_eng_data.to_csv(r"D:\House_data\raw_data\house_eng_data.csv")
    print (f"engineered data: {house_eng_data.shape}")
    # splitting the df into train and test
    # engineering the distance to downtown feature for train and test
    # engineering the zipcode price mean for train only to prevent leakage of price to test
    train,test = train_test_split(house_eng_data,test_size=0.25,random_state=42)
    seattle_lat, seattle_long = 47.6062, -122.3321
    for df in (train, test):
        df['distance_to_downtown'] = np.sqrt(
            (df['lat'] - seattle_lat) ** 2 + (df['long'] - seattle_long) ** 2
        )
        df.drop(["lat","long"],axis=1,inplace=True,errors='ignore')

    # zipcode and price means 
    zipcode_price_mean = train.groupby('zipcode')['price'].mean()
    general_mean = train['price'].mean()
    train['zip_price_means']=train['zipcode'].map(zipcode_price_mean)
    test["zip_price_means"]=test['zipcode'].map(zipcode_price_mean).fillna(general_mean)
    
    print ("distance_to_downtown and zip_price_means were created successfully ...")

    # X_train, y_train, X_test, y_test
    X_train,y_train = train[Features_selected],train[target]
    print (f"X_train: {X_train.shape}")
    print (f"y_train: {y_train.shape}")
    X_test,y_test = test[Features_selected],test[target]
    print (f"X_test: {X_test.shape}")
    print (f"y_test: {y_test.shape}")
    # saving train, test
    train_test={
        "X_train":X_train,
        "X_test":X_test,
        "y_train":y_train,
        "y_test":y_test
    }
    for name, split in train_test.items():
        split.to_csv(f"{r"D:\House_data\train_test"}\\{name}.csv",index=False)
    print ("data split was saved successfully ...")
    print ("*"*100)
    # Unsupervised Model: K-mean Classifier
    # Pipeline classifier
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("dim_reduction", PCA(n_components=3)),
    ("classifier", KMeans(n_clusters=4, random_state=42)) # assuming n_classifer = 4
])  
    # Elbo method, estimating best n_clusters
    best_inertia=[]
    for i in range(1,11):
        pipeline.set_params(classifier__n_clusters= i)
        pipeline.fit(X_train)
        best_inertia.append(pipeline.named_steps["classifier"].inertia_)
    best_inertia
    # elbow method
    plt.plot(range(1,11),best_inertia,marker='o')
    plt.title("elbow method")
    plt.xlabel("number of clusters")
    plt.ylabel("inertia")
    plt.show()
    # setting n_clusters to 4 
    pipeline.set_params(classifier__n_clusters= 4) # set parameters based on elbow method
    pipeline.fit(X_train)  
    # saving model 
    joblib.dump(pipeline,r"D:\House_data\models\unsupervised_KMean.joblib")
    print ("unsupervised KMeans model was saved successfully ...")
    # Supervised Model: Random Forest Regessor
    #Pipeline 
    rf_reg = Pipeline([("scaler",StandardScaler()),
                    ("randomforest",RandomForestRegressor(random_state=42))])
    cv = KFold(n_splits=5,shuffle=True,random_state=42)
    param_grid = {
        'randomforest__n_estimators':[50,100,200],
        'randomforest__min_samples_leaf':[1,2,3,4],
        'randomforest__max_depth': [None, 10, 20],
        'randomforest__max_features': [1.0,'sqrt']
    }
    grid_search =GridSearchCV(
        estimator=rf_reg,
        param_grid= param_grid,
        cv = cv,
        scoring='r2',
        verbose=2,
        n_jobs = -1
    )
    # fitting the model
    grid_search.fit(X_train,y_train)
    # best parameters and score
    print (f"best parameters: {grid_search.best_params_}")
    print (f"best R2 score: {grid_search.best_score_}")
    # save model
    joblib.dump(grid_search,r"D:\House_data\models\supervised_rf.joblib")
    print ("supervised model run successfully ")










if __name__ == "__main__":
    main()







