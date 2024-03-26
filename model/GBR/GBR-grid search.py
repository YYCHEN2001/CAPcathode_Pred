import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, make_scorer, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV

# Load the cleaned dataset
df = pd.read_csv('../../dataset/carbon_20240326_2.csv')  # Update this path to your dataset

# One-hot encode the categorical columns 'Electrolyte' and 'Current collector'
df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

# Initialize K-Fold Cross-Validator
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# Set up the parameter grid
param_grid = {
    'n_estimators': range(100, 2100, 100),
    'learning_rate': np.arange(0.025, 0.2, 0.025),
    'max_depth': range(3, 5)
}


def custom_evaluation(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse


# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(random_state=21)

scoring = {'MAE': make_scorer(custom_evaluation, greater_is_better=False),
           'RMSE': make_scorer(custom_evaluation, greater_is_better=False)}

grid_search = GridSearchCV(estimator=gbr,
                           param_grid=param_grid,
                           cv=kf,
                           scoring=scoring,
                           verbose=2,
                           refit=False,
                           n_jobs=-1)

# Perform Grid Search
grid_search.fit(X, y)

# 输出结果
results = grid_search.cv_results_
print("RMSE:", -results['mean_test_RMSE'])
