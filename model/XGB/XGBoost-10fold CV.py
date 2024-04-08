from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

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

metrics_df = perform_kfold_cv(xgb, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of XGB.csv', index=False)
