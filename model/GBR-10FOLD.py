import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold

# Load the cleaned dataset
df = pd.read_csv('carbon_cathode_cleaned.csv')  # Make sure to use the correct path for your dataset

# Features and Target separation
X = df.iloc[:, :-1]  # Exclude the last column (target)
y = df.iloc[:, -1]  # Target column

# Initialize 10-Fold Cross Validator
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=1300, learning_rate=0.175, max_depth=3,
                                min_samples_leaf=1, min_samples_split=2, random_state=21)

# Prepare DataFrame to store metrics for each fold
metrics_df = pd.DataFrame(columns=['Fold', 'R2', 'MAE', 'MAPE', 'RMSE'])

# Perform 10-Fold CV
rows = []  # To collect rows before creating the DataFrame
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    gbr.fit(X_train, y_train)

    # Predict
    y_pred = gbr.predict(X_test)

    # Calculate and store metrics in the list
    rows.append({
        'Fold': fold,
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,  # Convert to percentage
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    })

# Convert list of rows to DataFrame
metrics_df = pd.DataFrame(rows)

# Calculate average metrics and append
average_metrics = metrics_df.mean(numeric_only=True)
average_metrics['Fold'] = 'Average'
metrics_df = pd.concat([metrics_df, pd.DataFrame([average_metrics])], ignore_index=True)

# Display the metrics for each fold and the averages
print(metrics_df)
