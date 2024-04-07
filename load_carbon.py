import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load(filename, target='Cs'):
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

    # One-hot encode the categorical columns 'Electrolyte'
    df_encoded = pd.get_dummies(df, columns=['Electrolyte', 'Current collector'])

    # Features and Target separation
    x = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    return x, y


def split_scale(x, y, scale_data=False, test_size=0.3, random_state=21):
    """
    Split and optionally scale the data.

    Parameters:
    - X: Features DataFrame.
    - y: Target Series.
    - scale_data: bool, whether to scale the training and test data.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, random state for reproducibility.

    Returns:
    - X_train: Training features.
    - X_test: Test features.
    - y_train: Training target.
    - y_test: Test target.
    - X_train_scaled (optional): Scaled training features.
    - X_test_scaled (optional): Scaled test features.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    if scale_data:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled, y_train, y_test
    else:
        return x_train, x_test, y_train, y_test
