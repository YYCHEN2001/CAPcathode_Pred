from xgboost import XGBRegressor

from dataset_function_for_v3 import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v3.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=200,
                   learning_rate=0.16,
                   max_depth=6,
                   min_child_weight=3,
                   gamma=0.55,
                   subsample=0.65,
                   reg_alpha=0.42,
                   reg_lambda=2,
                   colsample_bytree=0.7,
                   colsample_bylevel=0.45,
                   colsample_bynode=0.8,
                   random_state=21)

# Train and evaluate the model
results = train_evaluate(xgb, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, xgb.predict(X_train), y_test, xgb.predict(X_test))

# Combine X_test, y_test, y_pred, and y_error into a single DataFrame
# y_error = abs(xgb.predict(X_test) - y_test)
# results_df = X_test.copy()
# results_df['y_test'] = y_test
# results_df['y_pred'] = xgb.predict(X_test)
# results_df['y_error'] = y_error
# results_df.to_csv('test_set_with_predictions_and_errors.csv', index=False)
