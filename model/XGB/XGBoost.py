from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=100,
                   learning_rate=0.2,
                   max_depth=9,
                   min_child_weight=2,
                   gamma=0.2,
                   subsample=0.4,
                   reg_alpha=0.8,
                   reg_lambda=0.4,
                   random_state=21)

# Train and evaluate the model
results = train_evaluate(xgb, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, xgb.predict(X_train), y_test, xgb.predict(X_test))
