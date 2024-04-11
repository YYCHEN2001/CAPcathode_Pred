from sklearn.ensemble import RandomForestRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=200,
                            max_depth=15,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# 执行十折交叉验证
metrics_df = perform_kfold_cv(rfr, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)

# Output:(n_estimators=400, max_depth=15, min_samples_leaf=1, min_samples_split=2)
#        Fold        R2        MAE      MAPE       RMSE
# 0         1  0.885215  10.200213  0.183857  14.759763
# 1         2  0.833618  10.354297  0.105471  15.976380
# 2         3  0.887825   8.219780  0.112015  10.821471
# 3         4  0.837704  12.072536  0.188582  16.752492
# 4         5  0.761896  11.857680  0.130298  21.514621
# 5         6  0.870178  11.463780  0.118466  16.258788
# 6         7  0.814142  11.155735  0.112127  17.493334
# 7         8  0.755021  10.621494  0.150788  17.635576
# 8         9  0.908807   7.516084  0.097040  11.090646
# 9        10  0.828628  11.863542  0.121399  17.732095
# 10  Average  0.838303  10.532514  0.132004  16.003516
