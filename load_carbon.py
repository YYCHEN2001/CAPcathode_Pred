import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load(filename):
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
    df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

    # Features and Target separation
    X = df_encoded.drop('Cs', axis=1)
    y = df_encoded['Cs']

    return X, y

def split_scale(X, y, scale_data=False, test_size=0.3, random_state=21):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        return X_train, X_test, y_train, y_test
