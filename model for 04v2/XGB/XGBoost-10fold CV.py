from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=200,
                   learning_rate=0.25,
                   max_depth=4,
                   min_child_weight=3,
                   gamma=0.1,
                   subsample=0.7,
                   reg_alpha=0.1,
                   colsample_bytree=0.7,
                   colsample_bylevel=0.7,
                   colsample_bynode=0.9,
                   random_state=21)

metrics_df = perform_kfold_cv(xgb, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)

# Output:
