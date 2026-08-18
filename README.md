# House Data

This project explores a house price dataset with Python and common data science libraries.

## Contents
- Data cleaning and exploratory data analysis in [src/eda.py](src/eda.py)
- Notebook workflow in [src/house_data.ipynb](src/house_data.ipynb)
- Dependencies listed in [requirements.txt](requirements.txt)

## House Data documentation

The dataset contains housing-related features used for regression analysis of sale prices.

### Features
- id: Unique identifier for each house
- date: Date of house sale
- price: Sale price (target for regression)
- bedrooms: Number of bedrooms
- bathrooms: Number of bathrooms using fractional values for partial baths; e.g. 2.5 means two full bathrooms plus one half bath, 1.75 means one full bathroom plus one three-quarter bath
- sqft_living: Living space area in square feet for the house itself
- sqft_lot: Lot size in square feet for the house itself
- floors: Number of floors in the house
- waterfront: Binary indicator for waterfront access; 1 means the house has a waterfront view or frontage, 0 means it does not
- view: View quality rating from 0 to 4, where 0 means no significant view and higher values indicate progressively better views
- condition: Overall house condition rating from 1 to 5, where 1 is poor and 5 is excellent
- grade: Construction and architecture grade rating from 1 to 13
- sqft_above: Above-ground living area in square feet (excludes basement area)
- sqft_basement: Basement living area in square feet; 0 means no basement area recorded
- yr_built: Year the house was built
- yr_renovated: Year the house was renovated (0 if never renovated)
- zipcode: Postal area code for the property
- lat: Latitude coordinate
- long: Longitude coordinate
- sqft_living15: Average living area of the 15 nearest neighboring homes, used as a neighborhood-level living-area estimate
- sqft_lot15: Average lot area of the 15 nearest neighboring homes, used as a neighborhood-level lot-size estimate

## Getting started

```bash
pip install -r requirements.txt
python src/eda.py
```
'''
Run Notebooks eda.ipynb and house_data.ipynb in VS Code or Jupyter Notebook 
to explore the dataset and visualize the results.
Make sure to have the dataset file `house_data.csv` in the same directory as the notebooks for proper execution.
Make sure that plotly is installed in your environment to visualize the data interactively.
'''
# Warning⚠️
Any one using this repo must run:
git fetch origin
git reset --hard origin/main