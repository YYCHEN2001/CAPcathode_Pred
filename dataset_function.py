import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_load(filename, target='Cs'):
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

    # Drop the 'Current Collector' column
    df_1 = df.drop('Current collector', axis=1)

    # One-hot encode the categorical columns 'Electrolyte'
    df_encoded = pd.get_dummies(df_1, columns=['Electrolyte'])

    # Fill the missing values in the 'Active mass loading' column
    df_encoded['Active mass loading'] = df_encoded['Active mass loading'].fillna(1.5)

    # Features and Target separation
    x = df_encoded.drop(target, axis=1)
    y = df_encoded[target]
    return x, y


def data_split(x, y, test_size=0.3, random_state=21):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - x: Features DataFrame.
    - y: Target Series.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, the seed used by the random number generator.

    Returns:
    - x_train: Features DataFrame of the training set.
    - x_test: Features DataFrame of the testing set.
    - y_train: Target Series of the training set.
    - y_test: Target Series of the testing set.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def feature_scaling(x_train, x_test):
    """
    Scale the features using StandardScaler.

    Parameters:
    - x_train: Features DataFrame of the training set.
    - x_test: Features DataFrame of the testing set.

    Returns:
    - x_train_scaled: Scaled Features DataFrame of the training set.
    - x_test_scaled: Scaled Features DataFrame of the testing set.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training set
    x_train_scaled = scaler.fit_transform(x_train)

    # Transform the testing set
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled
