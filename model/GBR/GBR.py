from sklearn.ensemble import GradientBoostingRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=100,
                                learning_rate=0.23,
                                max_depth=4,
                                min_samples_leaf=4,
                                min_samples_split=2,
                                alpha=0.001,
                                random_state=21)

# Train and evaluate the model
results = train_evaluate(gbr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, gbr.predict(X_train), y_test, gbr.predict(X_test))

# Calculate errors

y_error = abs(gbr.predict(X_test) - y_test)

# Combine X_test, y_test, y_pred, and y_error into a single DataFrame

results_df = X_test.copy()

results_df['y_test'] = y_test

results_df['y_pred'] = gbr.predict(X_test)

results_df['y_error'] = y_error

results_df.to_csv('test_set_with_predictions_and_errors.csv', index=False)
