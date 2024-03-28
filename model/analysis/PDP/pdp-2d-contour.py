from pdp_function import (load_split_data, combine_data,
                          target_plot, predict_plot, pdp_feature_plot,
                          target_2d_plot, predict_2d_plot, pdp_contour_plot)
from xgboost import XGBRegressor

filename = '../../../dataset/carbon_20240326_2.csv'
X_train, X_test, y_train, y_test, base_features = load_split_data(filename, 'Cs', test_size=0.3, random_state=21)
train_df, test_df = combine_data(X_train, X_test, y_train, y_test, base_features)

xgb = XGBRegressor(n_estimators=2000,
                   learning_rate=0.15,
                   max_depth=3,
                   min_child_weight=1,
                   gamma=0.5,
                   subsample=0.2,
                   reg_alpha=0.5,
                   reg_lambda=2,
                   random_state=21)

xgb.fit(X_train, y_train)
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)

fig = pdp_contour_plot(xgb, train_df, base_features, ['Vt', 'SSA'],
                       ['Vt', 'SSA'], 'matplotlib')
fig.show()
