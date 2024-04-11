import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

from dataset_function_for_v3 import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v3.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


def objective(params):
    clf = xgb.XGBRegressor(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        max_depth=int(params['max_depth']),
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        min_child_weight=int(params['min_child_weight']),
        colsample_bytree=params['colsample_bytree'],
        colsample_bylevel=params['colsample_bylevel'],
        colsample_bynode=params['colsample_bynode'],
        random_state=21
    )
    metric = cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 50),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'subsample': hp.quniform('subsample', 0.1, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.1, 1.0, 0.05),
    'reg_alpha': hp.quniform('reg_alpha', 0.01, 1, 0.01),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.05),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.1, 1, 0.05),
    'colsample_bynode': hp.quniform('colsample_bynode', 0.1, 1, 0.05)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print("Best: ", best)
