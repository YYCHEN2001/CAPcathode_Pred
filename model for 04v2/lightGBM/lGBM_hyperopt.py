from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


def objective(params):
    lgbm = LGBMRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_child_samples=int(params['min_child_samples']),
        num_leaves=int(params['num_leaves']),
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        learning_rate=params['learning_rate'],
        colsample_bytree=params['colsample_bytree'],
        colsample_bylevel=params['colsample_bylevel'],
        colsample_bynode=params['colsample_bynode'],
        random_state=21
    )

    metric = cross_val_score(lgbm, X_train, y_train, cv=10, scoring='neg_mean_absolute_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'min_child_samples': hp.quniform('min_child_samples', 1, 10, 1),
    'num_leaves': hp.quniform('num_leaves', 5, 50, 5),
    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    'colsample_bynode': hp.uniform('colsample_bynode', 0.5, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print("Best: ", best)

# Output:
# Best:  {
# 'colsample_bylevel': 0.7487899493443447,
# 'colsample_bynode': 0.6914924672998691,
# 'colsample_bytree': 0.9754041809754789,
# 'learning_rate': 0.16,
# 'max_depth': 8.0,
# 'min_child_samples': 3.0,
# 'min_child_weight': 1.829377903857563,
# 'n_estimators': 1000.0,
# 'num_leaves': 5.0,
# 'subsample': 0.7107910173966024}
