from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


def objective(params):
    krr = KernelRidge(
        alpha=params['alpha'],
        gamma=params['gamma'],
        kernel=params['kernel'],
        degree=params['degree'],
        coef0=params['coef0'],
    )
    metric = cross_val_score(krr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


# Define the hyperparameter configuration space
space = {'alpha': hp.uniform('alpha', 0, 1),
         'gamma': hp.uniform('gamma', 0.1, 1),
         'kernel': 'polynomial',
         'degree': 2,
         'coef0': hp.uniform('coef0', 0, 10)}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best: ", best)

# Output:
# Best:
