import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def dataset_load(filename):
    """
    Load and process the dataset without splitting.

    Parameters:
    - filename: str, the path to the dataset.

    Returns:
    - X: Features DataFrame.
    - y: Target Series.
    """
    # Load the cleaned dataset
    df = pd.read_csv(filename)

    # # Drop the 'Current Collector' column
    df_1 = df.drop(['Index', 'Cathode', 'Current collector', 'Active mass loading'], axis=1)

    # One-hot encode the categorical columns 'Elyte'
    df_encoded = pd.get_dummies(df_1, columns=['Elyte'])

    return df_encoded


def dataset_split(df, test_size=0.3, random_state=21, target='Cs'):
    """
    Split the dataset into training and testing sets, using quantile-based stratification for the target variable.

    Parameters:
    - df: DataFrame, the dataset.
    - test_size: float, the proportion of the testing set.
    - random_state: int, the random state.
    - target: str, the name of the target column.

    Returns:
    - x_train: DataFrame containing the features for the training set.
    - x_test: DataFrame containing the features for the testing set.
    - y_train: Series containing the target for the training set.
    - y_test: Series containing the target for the testing set.
    """
    # Rename the target column to 'target' for consistency
    if target != 'target':
        df = df.rename(columns={target: 'target'})

    # Use quantiles to define 10 intervals for the target value
    df['target_class'] = pd.qcut(df['target'], q=10, labels=False)

    # Splitting the dataset
    # Exclude 'target' and 'target_class' from the features
    x = df.drop(['target', 'target_class'], axis=1)
    y = df['target']
    stratify_column = df['target_class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify_column)
    # Remove the 'target_class' column to clean up
    df.drop(['target_class'], axis=1, inplace=True)

    return x_train, x_test, y_train, y_test


def feature_standard(x):
    scaler = StandardScaler()
    x_standard = scaler.fit_transform(x)
    return x_standard


def feature_normalize(x):
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x)
    return x_normalized


def target_normalize(y):
    scaler = MinMaxScaler()
    y_df = y.to_frame()
    y_normalized_array = scaler.fit_transform(y_df)
    y_normalized = pd.Series(y_normalized_array.flatten())
    return y_normalized
