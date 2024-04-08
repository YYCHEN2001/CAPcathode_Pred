from lightgbm import LGBMRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with LightGBM Regression
lgbm = LGBMRegressor(n_estimators=1000,
                     max_depth=8,
                     min_child_samples=3,
                     min_child_weight=1.8,
                     num_leaves=5,
                     subsample=0.7,
                     learning_rate=0.16,
                     colsample_bytree=0.97,
                     colsample_bylevel=0.75,
                     colsample_bynode=0.69,
                     random_state=21)

metrics_df = perform_kfold_cv(lgbm, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
