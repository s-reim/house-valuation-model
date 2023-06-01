import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('house_price_data_final.csv')

X = df.drop(['Sales price of the neighborhood', 'Sales price', 'Energylabel', 'Jordstykke_id', 'BFEnummer', 'Kommunekode', 'Ejendomsnummer', 'Salgsdato', 'Evaluation'], axis=1)
y = df['Sales price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the random forest regressor model
rf = RandomForestRegressor()

# define the hyperparameter space to search over
param_dist = {
    'n_estimators': Integer(500, 1500),
    'max_depth': Integer(5, 30),
    'max_features': Categorical(['sqrt', None]),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
}

# set up the Bayesian search
rf_bayes = BayesSearchCV(estimator = rf, search_spaces = param_dist,
                         n_iter = 20, scoring='neg_mean_squared_error', cv = 5, n_jobs=-1)

# fit the Bayesian search model to the data
rf_bayes.fit(X_train, y_train)

# print the best hyperparameters and the corresponding mean cross-validated score
print('Best Hyperparameters:', rf_bayes.best_params_)
print('Best Score:', rf_bayes.best_score_)