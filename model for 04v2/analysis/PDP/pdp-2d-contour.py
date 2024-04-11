from sklearn.ensemble import GradientBoostingRegressor

from dataset_function import dataset_load, dataset_split
from pdp_function import (combine_data, pdp_contour_plot)

df = dataset_load('../../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')
base_features = X_train.columns.tolist()
train_df, test_df = combine_data(X_train, X_test, y_train, y_test, base_features)

gbr = GradientBoostingRegressor(n_estimators=200,
                                learning_rate=0.17,
                                max_depth=4,
                                min_samples_leaf=1,
                                min_samples_split=2,
                                alpha=0.001,
                                subsample=0.8,
                                max_features=0.2,
                                random_state=21)

gbr.fit(X_train, y_train)
y_pred_train = gbr.predict(X_train)
y_pred_test = gbr.predict(X_test)

fig = pdp_contour_plot(gbr, test_df, base_features, ['O', 'N'],
                       ['O', 'N'], 12, 'matplotlib')
fig.show()
