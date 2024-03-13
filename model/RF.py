import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Import datasets
data = pd.read_csv('carbon cathode.csv')
# Setting the first three columns as indexes and separating features and target
X = data.iloc[:, 3:-1]  # Features
y = data.iloc[:, -1]  # Target

# Splitting the dataset into training and testing sets with a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Create a base model
rf = RandomForestRegressor(max_depth=None,
    min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=21)

# Training the best model
rf.fit(X_train, y_train)

# Predicting the target values for the test set
y_pred = rf.predict(X_test)

# Calculating the evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

print("R^2:", r2, "RMSE:", rmse, "MAPE:", mape)
