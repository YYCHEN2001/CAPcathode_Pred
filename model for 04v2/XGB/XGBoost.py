from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=1000,
                   learning_rate=0.06,
                   max_depth=6,
                   min_child_weight=3,
                   gamma=0.25,
                   subsample=0.6,
                   reg_alpha=0.81,
                   reg_lambda=2,
                   colsample_bytree=1,
                   colsample_bylevel=0.4,
                   colsample_bynode=0.65,
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
