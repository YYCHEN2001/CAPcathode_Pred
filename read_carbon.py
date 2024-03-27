import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_process_data(filename):
    # Load the cleaned dataset
    df = pd.read_csv(filename)

    # One-hot encode the categorical columns 'Electrolyte' and 'Current collector'
    df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

    # Features and Target separation
    X = df_encoded.drop('Cs', axis=1)
    y = df_encoded['Cs']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

    # Data standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test
