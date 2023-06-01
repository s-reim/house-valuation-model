import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np

def recursive_feature_elimination(X, y, model, threshold):
    global X_train, X_test, y_train, y_test

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the initial model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    initial_accuracy = r2_score(y_test, y_pred)
    print(f'Initial accuracy: {initial_accuracy}')
    # Recursive feature elimination loop
    while True:
        # Calculate feature importances
        importances = model.feature_importances_
        
        # Find the least important feature
        least_important_feature = np.argmin(importances)
        least_important_feature_name = X_train.columns[least_important_feature]

        # Remove the least important feature
        X_train = X_train.drop(X_train.columns[least_important_feature], axis=1)
        X_test = X_test.drop(X_test.columns[least_important_feature], axis=1)

        # Retrain the model without the least important feature
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        
        print(f'Feature removed: {least_important_feature_name}, new accuracy: {accuracy}')
        # Check if the accuracy dropped below the threshold
        if accuracy < threshold * initial_accuracy:
            break
    
    return model


df = pd.read_csv('house_price_data_final.csv')

X = df.drop(['Sales price of the neighborhood', 'Sales price', 'Energylabel', 'Jordstykke_id', 'BFEnummer', 'Kommunekode', 'Ejendomsnummer', 'Salgsdato', 'Evaluation'], axis=1)
y = df['Sales price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {'max_depth': 15, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 500}
rf = RandomForestRegressor(random_state=42, **best_params).fit(X_train, y_train)    

rf_reduced = recursive_feature_elimination(X, y, rf, threshold=0.95)