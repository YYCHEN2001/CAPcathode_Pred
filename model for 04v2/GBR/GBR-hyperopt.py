from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


def objective(params):
    gbr = GradientBoostingRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        max_features=params['max_features'],
        random_state=21
    )
    metric = cross_val_score(gbr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


# Define the hyperparameter configuration space
space = {
    'n_estimators': hp.quniform('n_estimators', 200, 1000, 100),
    'max_depth': hp.quniform('max_depth', 3, 18, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'subsample': hp.quniform('subsample', 0.1, 1.0, 0.1),
    'max_features': hp.uniform('max_features', 0.01, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best: ", best)

# Output:
# Best:  {
#     'learning_rate': 0.09954434211651723,
#     'max_depth': 13.0,
#     'max_features': 0.11836604315796628,
#     'min_samples_leaf': 6.0,
#     'min_samples_split': 7.0,
#     'n_estimators': 900.0,
#     'subsample': 0.8}
