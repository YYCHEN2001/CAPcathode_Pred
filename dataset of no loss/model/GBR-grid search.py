import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV

# Load the cleaned dataset
df = pd.read_csv('carbon_20240322.csv')  # Update this path to your dataset

# One-hot encode the categorical columns 'Electrolyte' and 'Current collector'
df_encoded = pd.get_dummies(df, columns=['Electrolyte', 'Current collector'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

# Initialize K-Fold Cross-Validator
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# Set up the parameter grid
param_grid = {
    'n_estimators': range(100, 2100, 100),
    'learning_rate': np.arange(0, 0.325, 0.025),
    'max_depth': range(2, 7)
}

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(random_state=21)

scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error',
}

grid_search = GridSearchCV(estimator=gbr,
                           param_grid=param_grid,
                           cv=kf,
                           scoring=scoring,
                           verbose=2,
                           refit=False,
                           n_jobs=-1)

# Perform Grid Search
grid_search.fit(X, y)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Predictions and metrics with the best model can be done similarly
# For example, to calculate R2 score using the best model:
y_pred = best_model.predict(X)
print("R2 Score with the Best Model:", r2_score(y, y_pred))
