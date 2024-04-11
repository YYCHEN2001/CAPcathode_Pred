import pandas as pd
from pdpbox import info_plots, pdp


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


def target_plot(df, feature, feature_name, target, engine):
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
        figsize=(12, 10),
        dpi=300,
        ncols=2,
        engine=engine,
        template='plotly_white',
    )
    """
    Parameters
        ----------
        center : bool, optional
            If True, the PDP will be centered by deducting the values of `grids[0]`.
            Default is True.
        plot_lines : bool, optional
            If True, ICE lines will be plotted. Default is False.
        frac_to_plot : int or float, optional
            Fraction of ICE lines to plot. Default is 1.
        cluster : bool, optional
            If True, ICE lines will be clustered. Default is False.
        n_cluster_centers : int or None, optional
            Number of cluster centers. Need to provide when `cluster` is True. Default
            is None.
        cluster_method : {'accurate', 'approx'}, optional
            Method for clustering. If 'accurate', use KMeans. If 'approx', use
            MiniBatchKMeans. Default is accurate.
        plot_pts_dist : bool, optional
            If True, distribution of points will be plotted. Default is False.
        to_bins : bool, optional
            If True, the axis will be converted to bins. Only applicable for numeric
            feature. Default is False.
        show_percentile : bool, optional
            If True, percentiles are shown in the plot. Default is False.
        which_classes : list of int, optional
            List of class indices to plot. If None, all classes will be plotted.
            Default is None.
        figsize : tuple or None, optional
            The figure size for matplotlib or plotly figure. If None, the default
            figure size is used. Default is None.
        dpi : int, optional
            The resolution of the plot, measured in dots per inch. Only applicable when
            `engine` is 'matplotlib'. Default is 300.
        ncols : int, optional
            The number of columns of subplots in the figure. Default is 2.
        plot_params : dict or None, optional
            Custom plot parameters that control the style and aesthetics of the plot.
            Default is None.
        engine : {'matplotlib', 'plotly'}, optional
            The plotting engine to use. Default is plotly.
        template : str, optional
            The template to use for plotly plots. Only applicable when `engine` is
            'plotly'. Reference: https://plotly.com/python/templates/ Default is
            plotly_white.
    """
    return fig, summary_df


def predict_plot(model, df, model_features, feature, feature_name, engine):
    """
    Check prediction distribution through a single feature
    """
    predict_feature = info_plots.PredictPlot(
        model=model,
        num_grid_points=10,
        df=df,
        model_features=model_features,
        n_classes=0,
        feature=feature,
        feature_name=feature_name,
    )

    fig, axes, summary_df = predict_feature.plot(
        ncols=2,
        figsize=(25.6, 14.4),
        dpi=300,
        plot_params={"gaps": {"inner_y": 0.05}},
        engine=engine,
        template='plotly_white',
    )
    return fig, summary_df


def pdp_feature_plot(model, df, model_features, feature, feature_name, engine):
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
        figsize=(12, 10),
        dpi=300,
        ncols=2,
        plot_params={"pdp_hl": True},
        engine=engine,
        template='plotly_white',
    )
    return fig, summary_df


def target_2d_plot(df, feature, feature_name, target, engine):
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
        figsize=(12, 10),
        dpi=300,
        ncols=2,
        annotate=True,
        plot_params={"subplot_ratio": {"y": [7, 0.8]}, "gaps": {"inner_y": 0.2}},
        engine=engine,
        template='plotly_white',
    )
    return fig, summary_df


def predict_2d_plot(model, df, model_features, features, feature_names, engine):
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
        figsize=(12, 10),
        dpi=300,
        plot_params={"gaps": {"inner_y": 0.05}},
        engine=engine,
        template='plotly_white',
    )
    return fig, summary_df


def pdp_contour_plot(model, df, model_features, features, feature_names, num_grid_points, engine):
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
        num_grid_points=num_grid_points,
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
        show_percentile=False,
        which_classes=None,
        figsize=(12, 10),
        dpi=300,
        ncols=2,
        plot_params=None,
        engine=engine,
        template="plotly_white",
    )
    return fig
