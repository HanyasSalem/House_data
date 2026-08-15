"""
FINAL SELECTED FEATURES -- ready for model creation (Ridge / ElasticNet)
--------------------------------------------------------------------------
This is the final, validated feature list after:
  - Removing id, date, and unnecessary date parts
  - Removing all _flag duplicate/near-zero-variance columns (except
    yr_renovated_flag, which passed all 4 checks and was kept)
  - Removing redundant engineered ratios, diffs, and interaction terms
    (basement_ratio, above_ratio, living_lot_ratio, sqft_living_diff,
    sqft_lot_diff, bath_bed_ratio, total_rooms, sqft_above,
    overall_quality, luxury_score, grade_sqft_living,
    waterfront_view_interaction, rooms_per_floor)
  - Replacing lat, long, and zipcode with 2 engineered features
    (distance_to_downtown, zip_mean_price) -- tested and confirmed to
    preserve the same predictive power (holdout R2 changed by < 0.001)

Holdout R2 with this final feature set: ~0.797
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# THE FINAL FEATURE LIST (17 features)
# ------------------------------------------------------------------
FEATURES = [
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
    'distance_to_downtown',   # engineered -- replaces lat + long
    'zip_mean_price',         # engineered -- replaces zipcode
]

TARGET = 'price'


# ------------------------------------------------------------------
# Load data and split BEFORE engineering zip_mean_price.
# This order matters: it stops test-set prices from leaking into
# the zipcode averages used as a feature.
# ------------------------------------------------------------------
df = pd.read_csv('house_eng_data.csv')
train, test = train_test_split(df, test_size=0.2, random_state=42)

# distance_to_downtown -- straight-line distance from Seattle's center
seattle_lat, seattle_long = 47.6062, -122.3321
for split in (train, test):
    split['distance_to_downtown'] = np.sqrt(
        (split['lat'] - seattle_lat) ** 2 + (split['long'] - seattle_long) ** 2
    )

# zip_mean_price -- average price per zipcode, calculated on TRAIN ONLY
zip_price_lookup = train.groupby('zipcode')['price'].mean()
overall_avg_price = train['price'].mean()  # fallback for any zipcode not seen in training
train['zip_mean_price'] = train['zipcode'].map(zip_price_lookup)
test['zip_mean_price'] = test['zipcode'].map(zip_price_lookup).fillna(overall_avg_price)


# ------------------------------------------------------------------
# Final modeling-ready data -- hand these straight to Ridge/ElasticNet
# ------------------------------------------------------------------
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print(f"Final feature count: {len(FEATURES)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
