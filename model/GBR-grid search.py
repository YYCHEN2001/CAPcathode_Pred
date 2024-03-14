import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV

# Load the cleaned dataset
df = pd.read_csv('carbon_cathode_cleaned.csv')  # Update this path to your dataset

# Features and Target separation
X = df.iloc[:, :-1]  # Exclude the last column (target)
y = df.iloc[:, -1]  # Target column

# Initialize K-Fold Cross-Validator
kf = KFold(n_splits=5, shuffle=True, random_state=21)

# Set up the parameter grid
param_grid = {
    'n_estimators': range(100, 2100, 100),
    'learning_rate': np.arange(0.05, 0.325, 0.025),
    'max_depth': range(2, 7)
}

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(random_state=21)

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=kf, scoring='r2', verbose=2, n_jobs=-1)

# Perform Grid Search
grid_search.fit(X, y)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score (R2):", grid_search.best_score_)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Predictions and metrics with the best model can be done similarly
# For example, to calculate R2 score using the best model:
y_pred = best_model.predict(X)
print("R2 Score with the Best Model:", r2_score(y, y_pred))
