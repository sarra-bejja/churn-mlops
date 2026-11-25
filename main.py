"""Point d'entrée : exécution des étapes du pipeline via la ligne de commande."""

import argparse
import os

import pandas as pd

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

MODEL_PATH = "classifier.joblib"
SAMPLE_CLIENT = {
    "CreditScore": 650,
    "Gender": 1,
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0,
}


def main():
    """Lit les arguments CLI et exécute les étapes demandées du pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument("--data", default="Churn_Modelling.csv", help="Chemin du CSV")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder le modèle (l'entraîne si besoin)",
    )
    parser.add_argument(
        "--load", action="store_true", help="Charger le modèle sauvegardé"
    )
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument(
        "--predict", action="store_true", help="Prédire pour un client exemple"
    )
    parser.add_argument("--all", action="store_true", help="Exécuter tout le pipeline")
    args = parser.parse_args()

    # --all active toutes les étapes
    if args.all:
        args.prepare = args.train = args.save = args.load = args.evaluate = True

    # Aucune option : afficher l'aide
    if not (
        args.prepare
        or args.train
        or args.save
        or args.load
        or args.evaluate
        or args.predict
    ):
        parser.print_help()
        return

    x_train = x_test = y_train = y_test = None
    model = None

    # 1. Préparation (aussi nécessaire pour train, save et evaluate)
    if args.prepare or args.train or args.save or args.evaluate:
        print("=== Préparation des données ===")
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        print(f"Données prêtes : train={x_train.shape}, test={x_test.shape}")

    # 2. Entraînement (aussi nécessaire pour save)
    if args.train or args.save:
        print("\n=== Entraînement du modèle ===")
        model = train_model(x_train, y_train)
        print("Modèle entraîné")

    # 3. Sauvegarde
    if args.save:
        print("\n=== Sauvegarde du modèle ===")
        save_model(model, MODEL_PATH)

    # 4. Chargement (aussi nécessaire pour evaluate si aucun modèle en mémoire)
    if args.load or ((args.evaluate or args.predict) and model is None):
        print("\n=== Chargement du modèle ===")
        if not os.path.exists(MODEL_PATH):
            print(
                f"Erreur : {MODEL_PATH} introuvable. Lancez d'abord : python main.py --save"
            )
            raise SystemExit(1)
        model = load_model(MODEL_PATH)

    # 5. Évaluation
    if args.evaluate:
        print("\n=== Évaluation du modèle ===")
        evaluate_model(model, x_test, y_test)

    # 6. Prédiction sur un client exemple
    if args.predict:
        print("\n=== Prédiction ===")
        client = pd.DataFrame([SAMPLE_CLIENT])
        resultat = model.predict(client)[0]
        print("Client exemple :", SAMPLE_CLIENT)
        print(
            "Résultat :",
            "Le client va PARTIR (1)" if resultat == 1 else "Le client va RESTER (0)",
        )


if __name__ == "__main__":
    main()
