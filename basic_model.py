import pandas as pd
import glob
import numpy as np
import geopandas as gpd
from scipy import stats
import libpysal as lp
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error, mean_absolute_percentage_error
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point
from shapely.ops import nearest_points


# Read in the data
cols = ['X', 'Y', 'Vurdering', 'Salgspris', 'Boligareal', 'Boligalder', 'Ombygningsalder', 'Ydervaegsmateriale', 'Tagtype', 'Varmekilde', 'Supplerende_varmekilde', 'Grundareal', 'Toilet_bad', 'Kommunekode', 'Ejendomsnummer', 'Salgsdato']
all_files = glob.glob("house_price_data_0*.csv")
dfs = [pd.read_csv(f, sep=',', usecols=cols) for f in all_files]
df = pd.concat(dfs, ignore_index=True)
df['Ombygningsalder'] = df['Ombygningsalder'].fillna(df['Boligalder'])
df = df.dropna()

# THE FOLLOWING IS PRICE PROJECTION - DONT USE IF PRICE ADJUSTER WAS USED
df['Salgspris'] = df['Salgspris'] / (1 + 0.0237) ** (2022 - df['Salgsdato']) # UNDO THE BASIC COMPOUND FROM DATA COLLECTION

# Define the price index
price_index = {
    2016: 100.9,
    2017: 104.175,
    2018: 109.85,
    2019: 113.875,
    2020: 118.15,
    2021: 128.975,
    2022: 128.95,
}

# # Convert all sales prices to 2022 money
df['Salgspris'] = df.apply(lambda row: row['Salgspris'] * (price_index[2022] / price_index[row['Salgsdato']]), axis=1)
df.drop(["Salgsdato"], axis=1, inplace=True)

# One hot encoding
ydervaegsmateriale = pd.get_dummies(df["Ydervaegsmateriale"], prefix="Ydervaegsmateriale")
tagtype = pd.get_dummies(df["Tagtype"], prefix="Tagtype")
varmekilde = pd.get_dummies(df["Varmekilde"], prefix="Varmekilde")
supplerende_varmekilde = pd.get_dummies(df["Supplerende_varmekilde"], prefix="Supplerende_varmekilde")

df = pd.concat([df, ydervaegsmateriale, tagtype, varmekilde, supplerende_varmekilde], axis=1)
df.drop(["Ydervaegsmateriale", "Tagtype", "Varmekilde", "Supplerende_varmekilde"], axis=1, inplace=True)

# Remove outliers
Q1 = df['Salgspris'].quantile(0.05)
Q3 = df['Salgspris'].quantile(0.95)
IQR = Q3 - Q1
df = df[~((df['Salgspris'] < (Q1 - 1.5 * IQR)) |(df['Salgspris'] > (Q3 + 1.5 * IQR)))]

# Remove observations with z-score > 3
z_scores = np.abs(stats.zscore(df['Salgspris']))
df = df[z_scores < 3]

# Valuation / Sales price deviation
z_scores = np.abs(stats.zscore(df["Vurdering"] - df["Salgspris"]))
df.drop(["Vurdering"], axis=1, inplace=True)
data = df[z_scores < 3]


# # Load the coastline shapefile as a GeoDataFrame
# coast = gpd.read_file("KYSTLINJE2.shp")

# # Spatial lag
# geometry = gpd.points_from_xy(data['X'], data['Y'])
# gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=coast.crs)

# # Define a function to calculate the distance from a point to the nearest point on the coast
# def distance_to_coast_edge(point):
#     nearest_edge_point = coast.geometry.boundary.distance(point).idxmin()
#     nearest_edge_point = coast.geometry.boundary[nearest_edge_point]
#     return point.distance(nearest_edge_point)

# # Apply the function to each point in the GeoDataFrame to calculate the distance to the coast edge
# gdf['distance_to_coast'] = gdf.geometry.apply(distance_to_coast_edge)


# w = lp.weights.DistanceBand.from_dataframe(gdf, threshold=500, binary=False)

# gdf['salgspris_lag'] = lp.weights.lag_spatial(w, gdf['Salgspris'])

# # Export the GeoDataFrame to a CSV file
# gdf.drop('geometry', axis=1).to_csv('FYN_HOUSE_PRICE_DATA_v2.csv', index=False)

y = df['Salgspris']
X = df.drop(['Salgspris'], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the parameter grid
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
#                'max_features': [None, 'sqrt'],
#                'max_depth': max_depth,
#                'min_samples_split': [2, 5, 10],
#                'min_samples_leaf': [1, 2, 4]}

# # Create the RandomForestRegressor object
# rf = RandomForestRegressor()

# # Perform grid search cross-validation
# grid_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameter values
# best_params = grid_search.best_params_

# print the best hyperparameters and the corresponding mean cross-validated score
#print('Best Hyperparameters:', grid_search.best_params_)

best_params = {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 30} # basic model
#best_params = {'max_depth': 53, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 245}

rf = RandomForestRegressor(random_state=42, **best_params).fit(X_train, y_train)

# Make predictions on the testing set
predictions = rf.predict(X_test)
train_predictions = rf.predict(X_train)


# Evaluate the performance of the model
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mpe = mean_absolute_percentage_error(y_test, predictions)

print("PREDICTION")
print("R-squared value on test data:", r2)
print("Mean Absolute Error on test data:", mae)
print("Mean Squared Error on test data:", mse)
print("Mean Percentage Error:", mpe)

r2 = r2_score(y_train, train_predictions)
mae = mean_absolute_error(y_train, train_predictions)
mse = mean_squared_error(y_train, train_predictions)

print("\nTRAIN")
print("R-squared value on test data:", r2)
print("Mean Absolute Error on test data:", mae)
print("Mean Squared Error on test data:", mse)

# define the ranges of sales prices
ranges = [(0, 1000000), (1000000, 2000000),(2000000, 3000000), (3000000, 4000000), (4000000, np.inf)]

# initialize an empty list to store the MPEs for each range
mpes = []

# calculate the MPE for each range
for r in ranges:
    # subset the data based on the sales price range
    mask = (y_test >= r[0]) & (y_test < r[1])
    y_test_range = y_test[mask]
    y_pred_range = rf.predict(X_test[mask])
    
    # calculate the MPE for the range
    mpe_range = np.mean((y_test_range - y_pred_range) / y_test_range) * 100
    
    # add the MPE to the list
    mpes.append(mpe_range)

# print the MPEs for each range
for i, r in enumerate(ranges):
    print(f"MPE for range {r}: {mpes[i]:.2f}%")

# Set the threshold value
threshold = 0.05

# Get the feature importances from the random forest model
importances = rf.feature_importances_

# Get the names of the features
feature_names = X.columns

# Filter the feature names based on the importance values
important_features = feature_names[importances > threshold]

# Get the importances of the important features
important_importances = importances[importances > threshold]

# Plot the important features and their importances
plt.figure(figsize=(10, 6))
plt.bar(important_features, important_importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances (threshold = {})'.format(threshold))
plt.show()