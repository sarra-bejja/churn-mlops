"""Pipeline ML : prédiction du churn (Churn_Modelling.csv)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, nettoie, encode et sépare en train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine aléatoire.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()

    # Encodage de Gender
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=["Surname", "Geography", "RowNumber", "CustomerId"])

    x = df.drop(columns=["Exited"])
    y = df["Exited"]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et le retourne."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle : accuracy, matrice de confusion, rapport de classification.

    Returns:
        float: accuracy
    """
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc * 100:.2f}%")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    return acc


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(path)
    print(f"Modèle chargé depuis {path}")
    return model
