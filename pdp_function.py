import pandas as pd
from pdpbox import info_plots, pdp
from sklearn.model_selection import train_test_split


def load_split_data(filename, target, test_size=0.3, random_state=21):
    """
    Load data and split it into training and testing sets.
    Args:
    - filename: str, the name of the data file to load.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, the seed used by the random number generator.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    - base_features: list, the feature names of the dataset.
    """
    # Assuming load function is defined to load data
    data = pd.read_csv(filename)
    x = data.drop(target, axis=1)
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    base_features = x_train.columns.tolist()

    return x_train, x_test, y_train, y_test, base_features


def combine_data(x_train, x_test, y_train, y_test, base_features):
    """
    Combine the training and testing sets.
    """
    # Create DataFrame of features
    x_train_df = pd.DataFrame(x_train, columns=base_features)
    x_test_df = pd.DataFrame(x_test, columns=base_features)

    # ensure y_train, y_test is pd.Series kind
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]  # choose the 1st column, or specify the column name
    else:
        y_train = pd.Series(y_train, name='Cs')

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]  # choose the 1st column, or specify the column name
    else:
        y_test = pd.Series(y_test, name='Cs')

    # reset the index to fit the index of x_train_df and x_test_df
    y_train.index = x_train_df.index
    y_test.index = x_test_df.index

    # combine the features and target
    train_df = pd.concat([x_train_df, y_train], axis=1)
    test_df = pd.concat([x_test_df, y_test], axis=1)

    return train_df, test_df


def target_plot(df, feature, feature_name, target):
    """
    plot of Target distribution through a single feature
    """
    target_feature = info_plots.TargetPlot(
        df=df,
        feature=feature,
        feature_name=feature_name,
        target=target,
        num_grid_points=10,
        grid_type='percentile',
        show_outliers=False,
        endpoint=True
    )

    fig, axes, summary_df = target_feature.plot(
        show_percentile=True,
        ncols=2,
        engine='plotly',
        template='plotly_white',
    )
    return fig, summary_df


def predict_plot(model, df, model_features, feature, feature_name):
    """
    Check prediction distribution through a single feature
    """
    predict_feature = info_plots.PredictPlot(
        model=model,
        df=df,
        model_features=model_features,
        feature=feature,
        feature_name=feature_name,
    )

    fig, axes, summary_df = predict_feature.plot(
        ncols=2,
        plot_params={"gaps": {"inner_y": 0.05}},
        engine='plotly',
        template='plotly_white',
    )
    return fig, summary_df


def pdp_feature_plot(model, df, model_features, feature, feature_name):
    """
    Partial dependence plot of a single feature
    """
    pdp_feature = pdp.PDPIsolate(
        model=model,
        df=df,
        model_features=model_features,
        feature=feature,
        feature_name=feature_name,
    )

    fig, axes, summary_df = pdp_feature.plot(
        center=True,
        plot_lines=True,
        frac_to_plot=100,
        cluster=False,
        n_cluster_centers=None,
        cluster_method='accurate',
        plot_pts_dist=False,
        to_bins=False,
        show_percentile=True,
        which_classes=None,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params={"pdp_hl": True},
        engine='matplotlib',
        template='plotly_white',
    )
    return fig, summary_df


def target_2d_plot(df, feature, feature_name, target):
    """
    Plot of Target distribution through a single feature.

    Args:
    - df: DataFrame containing the data.
    - feature: List of the feature names (should contain exactly 2 elements).
    - feature_name: List of the feature names for display (should contain exactly 2 elements).
    - target: Name of the target column.

    Returns:
    - fig: The figure object created by plotly.
    - summary_df: A DataFrame summarizing the plot data.
    """

    # Check if both feature and feature_name are lists containing exactly two elements
    if (not isinstance(feature, list) or not isinstance(feature_name, list) or
            len(feature) != 2 or len(feature_name) != 2):
        raise ValueError("Both 'feature' and 'feature_name' must be lists containing exactly 2 elements.")

    target_1_2 = info_plots.InteractTargetPlot(
        df=df,
        features=feature,
        feature_names=feature_name,
        target=target,
        num_grid_points=10,
        grid_types='percentile',
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        show_outliers=False,
        endpoints=True,
    )

    fig, axes, summary_df = target_1_2.plot(
        show_percentile=True,
        figsize=None,
        ncols=2,
        annotate=True,
        plot_params={"subplot_ratio": {"y": [7, 0.8]}, "gaps": {"inner_y": 0.2}},
        engine='plotly',
        template='plotly_white',
    )
    return fig, summary_df


def predict_2d_plot(model, df, model_features, features, feature_names):
    """
    Check prediction distribution through two features

    Args:
    - df: DataFrame containing the data.
    - feature: List of the feature names (should contain exactly 2 elements).
    - feature_name: List of the feature names for display (should contain exactly 2 elements).

    Returns:
    - fig: The figure object created by plotly.
    - summary_df: A DataFrame summarizing the plot data.
    """

    # Check if both feature and feature_name are lists containing exactly two elements
    if (not isinstance(features, list) or not isinstance(feature_names, list) or
            len(features) != 2 or len(feature_names) != 2):
        raise ValueError("Both 'feature' and 'feature_name' must be lists containing exactly 2 elements.")

    predict_1_2 = info_plots.InteractPredictPlot(
        model=model,
        df=df,
        model_features=model_features,
        features=features,
        feature_names=feature_names,
    )

    fig, axes, summary_df = predict_1_2.plot(
        ncols=2,
        plot_params={"gaps": {"inner_y": 0.05}},
        engine='plotly',
        template='plotly_white',
    )
    return fig, summary_df


def pdp_contour_plot(model, df, model_features, features, feature_names):
    """
    Partial dependence plot of two features

    Args:
    - df: DataFrame containing the data.
    - feature: List of the feature names (should contain exactly 2 elements).
    - feature_name: List of the feature names for display (should contain exactly 2 elements).

    Returns:
    - fig: The figure object created by plotly.
    - summary_df: A DataFrame summarizing the plot data.
    """
    if (not isinstance(features, list) or not isinstance(feature_names, list) or
            len(features) != 2 or len(feature_names) != 2):
        raise ValueError("Both 'feature' and 'feature_name' must be lists containing exactly 2 elements.")

    pdp_inter = pdp.PDPInteract(
        model=model,
        df=df,
        model_features=model_features,
        n_classes=0,
        features=features,
        feature_names=feature_names,
    )

    fig, axes = pdp_inter.plot(
        plot_type="contour",
        to_bins=True,
        plot_pdp=True,
        show_percentile=True,
        which_classes=None,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    )
    return fig
