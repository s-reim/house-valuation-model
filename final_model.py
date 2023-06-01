import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from statsmodels.robust.scale import mad

df = pd.read_csv('house_price_data_final.csv')

X = df.drop(['Sales price of the neighborhood', 'Sales price', 'Energylabel', 'Jordstykke_id', 'BFEnummer', 'Kommunekode', 'Ejendomsnummer', 'Salgsdato', 'Evaluation', 'Distance to lake', 'Distance to stream', 'Distance to windturbines', 'Distance to forest', 'Year of reconstruction'], axis=1)
y = df['Sales price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {'max_depth': 15, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 500} # Can be tweaked in previous scripts with hyperparameter tuning

rf = RandomForestRegressor(random_state=42, **best_params).fit(X_train, y_train)

# Save the model to a file
joblib.dump(rf, 'random_forest_model.pkl')

# Make predictions on the testing set
predictions = rf.predict(X_test)

def calculate_pm20(predictions, actual_prices):
    upper_limit = actual_prices * 1.2
    lower_limit = actual_prices * 0.8
    
    within_limits = np.logical_and(predictions >= lower_limit, predictions <= upper_limit)
    pm20 = np.round(np.mean(within_limits) * 100, 1)
    under_20 = np.round(np.mean(predictions < lower_limit) * 100, 1)
    over_20 = np.round(np.mean(predictions > upper_limit) * 100, 1)
    return pm20, under_20, over_20

def calculate_cod(predictions, actual_prices):
    median = np.median(actual_prices)
    relative_deviations = np.abs(predictions - actual_prices) / median
    cod = np.round(np.mean(relative_deviations) * 100, 1)
    return cod

def calculate_mad(predictions, actual_prices):
    absolute_deviations = np.abs(predictions - actual_prices)
    mad1 = np.round(np.mean(absolute_deviations) / 1000) * 1000
    return mad1

pm20, under_20, over_20 = calculate_pm20(predictions, y_test)
cod = calculate_cod(predictions, y_test)
mad1 = calculate_mad(predictions, y_test)

print("Pm20:", pm20)
print("Percentage under 20%:", under_20)
print("Percentage over 20%:", over_20)
print("CoD:", cod)
print("MAD:", mad1)

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

# Evaluate the performance of the model
r2 = r2_score(y_train, train_predictions)
mae = mean_absolute_error(y_train, train_predictions)
mse = mean_squared_error(y_train, train_predictions)

print("\nTRAIN")
print("R-squared value on test data:", r2)
print("Mean Absolute Error on test data:", mae)
print("Mean Squared Error on test data:", mse)

# define the ranges of sales prices
ranges = [(0, 500000), (500000, 1000000), (1000000, 2000000),(2000000, 3000000), (3000000, 4000000), (4000000, 5000000), (5000000, np.inf)]

# initialize an empty list to store the MPEs for each range
mpes = []

# calculate the MPE for each range
for r in ranges:
    # subset the data based on the sales price range
    mask = (y_test >= r[0]) & (y_test < r[1])
    y_test_range = y_test[mask]
    y_pred_range = rf.predict(X_test[mask])
    
    pm20, under_20, over_20 = calculate_pm20(y_pred_range, y_test_range)
    cod = calculate_cod(y_pred_range, y_test_range)
    mad1 = calculate_mad(y_pred_range, y_test_range)
    print(f"\n{r}")
    print(len(y_test_range))
    print("Pm20:", pm20)
    print("Percentage under 20%:", under_20)
    print("Percentage over 20%:", over_20)
    print("CoD:", cod)
    print("MAD:", mad1)

# Set the threshold value
threshold = 0

# Get the feature importances from the random forest model
importances = rf.feature_importances_

# Get the names of the features
feature_names = X.columns

# Filter the feature names based on the importance values
important_features = feature_names[importances > threshold]

# Get the importances of the important features
important_importances = importances[importances > threshold]

# Sort the features and importances together
sorted_indices = sorted(range(len(important_importances)), key=lambda k: important_importances[k])
sorted_features = [important_features[i] for i in sorted_indices]
sorted_importances = [important_importances[i] for i in sorted_indices]

# Set the plot style using Seaborn
sns.set(style="whitegrid")

# Create a figure and axes
plt.figure(figsize=(6, 10))
ax = sns.barplot(x=sorted_importances[::-1], y=sorted_features[::-1], palette="viridis")

# Rotate and align the feature names
plt.yticks(fontsize=14)
plt.xticks([i * 0.1 for i in range(7)], fontsize=14)

# Set labels and title
plt.xlabel('Feature', fontsize=16)
plt.ylabel('Importance', fontsize=16)

# Add values on the right of the bars
for i, v in enumerate(sorted_importances[::-1]):
    ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontsize=14)

# Display the plot
plt.savefig(f"feature_importances_lag.png", dpi=300, bbox_inches='tight')
plt.close()