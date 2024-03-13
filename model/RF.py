import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Import datasets
data = pd.read_csv('carbon_cathode_cleaned.csv')
# Setting the first three columns as indexes and separating features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Target

# Splitting the dataset into training and testing sets with a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Create a base model
rf = RandomForestRegressor(max_depth=None,
                           min_samples_leaf=1, min_samples_split=2, n_estimators=1000, random_state=21)

# Training the best model
rf.fit(X_train, y_train)

# Predicting the target values for the test set
y_pred = rf.predict(X_test)

# Calculating the evaluation metrics
# Predicting the target values for both the training and testing sets
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Calculating evaluation metrics for both sets
training_metrics = {
    "R^2": r2_score(y_train, y_train_pred),
    "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "MAE": mean_absolute_error(y_train, y_train_pred),
    "MAPE": mean_absolute_percentage_error(y_train, y_train_pred) * 100
}

testing_metrics = {
    "R^2": r2_score(y_test, y_test_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    "MAE": mean_absolute_error(y_test, y_test_pred),
    "MAPE": mean_absolute_percentage_error(y_test, y_test_pred) * 100
}

metrics_df = pd.DataFrame({
    "Metric": ["R^2", "RMSE", "MAE", "MAPE"],
    "Training Set": [training_metrics["R^2"], training_metrics["RMSE"], training_metrics["MAE"],
                     training_metrics["MAPE"]],
    "Testing Set": [testing_metrics["R^2"], testing_metrics["RMSE"], testing_metrics["MAE"], testing_metrics["MAPE"]]
})

print(metrics_df)
