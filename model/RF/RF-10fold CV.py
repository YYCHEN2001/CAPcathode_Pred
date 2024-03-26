import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold

# Load the cleaned dataset
df = pd.read_csv('../../dataset/carbon_20240326_2.csv')

# One-hot encode the categorical columns 'Electrolyte'
df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

# Initialize 10-Fold Cross Validator
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=500,
                            max_depth=9,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# Prepare DataFrame to store metrics for each fold
metrics_df = pd.DataFrame(columns=['Fold', 'R2', 'MAE', 'MAPE', 'RMSE'])

# Perform 10-Fold CV
rows = []  # To collect rows before creating the DataFrame
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    rfr.fit(X_train, y_train)

    # Predict
    y_pred = rfr.predict(X_test)

    # Calculate and store metrics in the list
    rows.append({
        'Fold': fold,
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
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
# Save the DataFrame to a CSV file
metrics_df.to_csv('./result output/KFold results of RF.csv', index=False)