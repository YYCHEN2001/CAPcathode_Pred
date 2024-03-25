import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, \
    root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# Load the cleaned dataset
df = pd.read_csv('carbon_20240322.csv')

# One-hot encode the categorical columns 'Electrolyte' and 'Current collector'
df_encoded = pd.get_dummies(df, columns=['Electrolyte', 'Current collector'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=2000,
                                learning_rate=0.15,
                                max_depth=4,
                                min_samples_leaf=1,
                                min_samples_split=2,
                                random_state=21)
gbr.fit(X_train, y_train)
y_pred_train = gbr.predict(X_train)
y_pred_test = gbr.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
rmse_train = root_mean_squared_error(y_train, y_pred_train)
rmse_test = root_mean_squared_error(y_test, y_pred_test)

results = pd.DataFrame({
    'Metric': ['R2', 'MAE', 'MAPE', 'RMSE'],
    'train set': [r2_train, mae_train, mape_train, rmse_train],
    'Test set': [r2_test, mae_test, mape_test, rmse_test]
})

print(results)

# Initialize 10-Fold Cross Validator
kf = KFold(n_splits=10, shuffle=True, random_state=21)

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
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
    })

# Convert list of rows to DataFrame
metrics_df = pd.DataFrame(rows)

# Calculate average metrics and append
average_metrics = metrics_df.mean(numeric_only=True)
average_metrics['Fold'] = 'Average'
metrics_df = pd.concat([metrics_df, pd.DataFrame([average_metrics])], ignore_index=True)

# Display the metrics for each fold and the averages
print(metrics_df)
