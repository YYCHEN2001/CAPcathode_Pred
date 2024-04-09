from sklearn.ensemble import GradientBoostingRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=900,
                                learning_rate=0.1,
                                max_depth=13,
                                min_samples_leaf=6,
                                min_samples_split=7,
                                subsample=0.8,
                                max_features=0.12,
                                random_state=21)

metrics_df = perform_kfold_cv(gbr, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)

# Output:
#        Fold        R2       MAE      MAPE       RMSE
# 0         1  0.938270  6.439321  0.096369  10.823954
# 1         2  0.907777  7.005155  0.063216  11.894446
# 2         3  0.960039  4.750581  0.062042   6.458841
# 3         4  0.963630  5.013961  0.064376   7.930485
# 4         5  0.905277  6.638227  0.058949  13.569937
# 5         6  0.921850  7.200571  0.072995  12.614728
# 6         7  0.931099  5.992748  0.061307  10.651120
# 7         8  0.928394  6.048907  0.084838   9.534555
# 8         9  0.953297  5.243917  0.057398   7.936820
# 9        10  0.946872  5.521947  0.063228   9.873027
# 10  Average  0.935651  5.985534  0.068472  10.128791
