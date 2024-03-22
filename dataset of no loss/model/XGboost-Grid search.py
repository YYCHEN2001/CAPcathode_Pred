import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score, \
    make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Load the datasets
df_train = pd.read_csv('data_75.csv')
df_test = pd.read_csv('data_25.csv')

# Drop the first two columns
df_train = df_train.drop(df_train.columns[[0, 1]], axis=1)
df_test = df_test.drop(df_test.columns[[0, 1]], axis=1)

# One-hot encode some columns
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_columns_train = encoder.fit_transform(df_train[['Electrolyte', 'Current collector']]).toarray()
encoded_columns_test = encoder.transform(df_test[['Electrolyte', 'Current collector']]).toarray()
encoded_df_train = pd.DataFrame(encoded_columns_train,
                                columns=encoder.get_feature_names_out(['Electrolyte', 'Current collector']))
encoded_df_test = pd.DataFrame(encoded_columns_test,
                               columns=encoder.get_feature_names_out(['Electrolyte', 'Current collector']))

# Combine data
df_train_encoded = pd.concat([df_train.drop(['Electrolyte', 'Current collector'], axis=1), encoded_df_train], axis=1)
df_test_encoded = pd.concat([df_test.drop(['Electrolyte', 'Current collector'], axis=1), encoded_df_test], axis=1)

# Split the dataset into features and target variable
X_train = df_train_encoded.drop('Cs', axis=1)
y_train = df_train_encoded['Cs']
X_test = df_test_encoded.drop('Cs', axis=1)
y_test = df_test_encoded['Cs']

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': range(1000, 2000, 200),
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'reg_alpha': [0.1, 0.5, 1],
    'reg_lambda': [0.01, 0.25, 0.5, 1]
}

# Create custom scorer
scorers = {
    'r2_score': make_scorer(r2_score),
    'rmse': make_scorer(root_mean_squared_error),
    'mape': make_scorer(mean_absolute_percentage_error)
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(XGBRegressor(random_state=21), param_grid, scoring=scorers, refit='r2_score', cv=5,
                           verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters and scores
print("Best parameters:", grid_search.best_params_)
print("Best R2 score:", grid_search.best_score_)

# Evaluate on test data
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(root_mean_squared_error(y_test, y_pred_test))
test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

print("Test set R2:", test_r2)
print("Test set RMSE:", test_rmse)
print("Test set MAPE:", test_mape)
