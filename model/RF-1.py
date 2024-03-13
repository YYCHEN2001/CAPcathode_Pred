import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
rf = RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100, random_state=21)

# Training the model
rf.fit(X_train, y_train)

# Predicting the target values for the training and test set
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Calculating the evaluation metrics for both training and test sets
metrics = {
    "R^2": [r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)],
    "RMSE": [np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_test_pred))],
    "MAPE": [mean_absolute_percentage_error(y_train, y_train_pred) * 100,  # MAPE as a percentage
             mean_absolute_percentage_error(y_test, y_test_pred) * 100]  # MAPE as a percentage
}

# Plotting the metrics for visual comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Titles for subplots
titles = ["R^2", "RMSE", "MAPE"]

for i, (metric, values) in enumerate(metrics.items()):
    ax[i].bar(["Training", "Testing"], values, color=['blue', 'orange'])
    ax[i].set_title(titles[i])
    ax[i].set_ylabel(metric)
    ax[i].set_ylim([0, max(values) + (0.1 * max(values))])  # Adjust ylim to give some space

plt.suptitle('Random Forest Model Performance: Training vs Testing')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
