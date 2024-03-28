import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error


def perform_kfold_cv(model, x, y, n_splits=10, random_state=21):
    """
    Perform K-Fold Cross Validation.

    Parameters:
    - model: The regression model to be evaluated.
    - X: Features DataFrame.
    - y: Target Series.
    - n_splits: Number of folds. Default is 10.
    - random_state: Random state for reproducibility.

    Returns:
    - metrics_df: A DataFrame containing the metrics for each fold and the averages.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_df = pd.DataFrame(columns=['Fold', 'R2', 'MAE', 'MAPE', 'RMSE'])
    rows = []

    for fold, (train_index, test_index) in enumerate(kf.split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(x_train, y_train)

        # Predict
        y_pred = model.predict(x_test)

        # Calculate and store metrics
        rows.append({
            'Fold': str(fold),
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'RMSE': root_mean_squared_error(y_test, y_pred)
        })

    # Convert list of rows to DataFrame
    metrics_df = pd.DataFrame(rows)

    # Calculate average metrics and append
    average_metrics = metrics_df.mean(numeric_only=True)
    average_metrics['Fold'] = 'Average'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([average_metrics])], ignore_index=True)

    return metrics_df
