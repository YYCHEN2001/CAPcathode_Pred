from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


def objective(params):
    rfr = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_leaf=int(params['min_samples_leaf']),
        min_samples_split=int(params['min_samples_split']),
        random_state=21
    )
    metric = cross_val_score(rfr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


# Define the hyperparameter configuration space
space = {'n_estimators': hp.quniform('n_estimators', 100, 500, 100),
         'max_depth': hp.quniform('max_depth', 3, 15, 1),
         'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
         'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best: ", best)

# Output:
# {'max_depth': 15.0, 'min_samples_leaf': 1.0, 'min_samples_split': 2.0, 'n_estimators': 400.0}
