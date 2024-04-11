from xgboost import XGBRegressor

from dataset_function_for_v3 import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

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

metrics_df = perform_kfold_cv(xgb, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)

# Output:
#        Fold        R2       MAE      MAPE       RMSE
# 0         1  0.779165  7.932741  0.075820  18.268069
# 1         2  0.963968  4.609417  0.064476   6.325809
# 2         3  0.972405  4.771830  0.052349   6.833538
# 3         4  0.966261  4.713017  0.048933   6.956359
# 4         5  0.867539  6.986173  0.083456  12.478676
# 5         6  0.957191  5.837267  0.069968   9.006416
# 6         7  0.958969  3.861656  0.071402   6.469750
# 7         8  0.947042  6.314301  0.059383  10.939678
# 8         9  0.946968  6.168785  0.052459  12.080988
# 9        10  0.947363  5.285874  0.072655   9.021324
# 10  Average  0.930687  5.648106  0.065090   9.838061
