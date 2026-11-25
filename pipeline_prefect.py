"""Pipeline Prefect : automatisation des étapes du projet ML (churn)."""

import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
PYTHON_BIN = VENV_DIR / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path(sys.executable)
MAIN_PY = BASE_DIR / "main.py"

# Les étapes de qualité et de sécurité ne visent que nos 3 fichiers
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd):
    """Exécute une commande dans le dossier du projet (échoue si erreur)."""
    subprocess.run([str(c) for c in cmd], cwd=BASE_DIR, check=True)


# ---------------------------------------------------------------- TASKS
@task(name="install-dependencies")
def install_dependencies() -> None:
    """Installe les dépendances du fichier requirements.txt."""
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "pip", "install", "-r", "requirements.txt"])
    logger.info("Dependencies installed.")


@task(name="format-code")
def format_code() -> None:
    """Formate le code avec isort et black."""
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "isort", "--profile", "black", *CODE_FILES])
    _run([PYTHON_BIN, "-m", "black", *CODE_FILES])
    logger.info("Code formatting complete (isort + black).")


@task(name="check-quality")
def check_quality() -> None:
    """Vérifie la qualité du code avec flake8 et pylint."""
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "flake8", "--max-line-length=100", *CODE_FILES])
    _run([PYTHON_BIN, "-m", "pylint", *CODE_FILES])
    logger.info("Code quality OK (flake8 + pylint).")


@task(name="check-security")
def check_security() -> None:
    """Analyse la sécurité du code avec bandit."""
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "bandit", "-q", "-ll", *CODE_FILES])
    logger.info("Code security OK (bandit).")


@task(name="unit-tests")
def run_tests() -> None:
    """Exécute les tests unitaires avec pytest."""
    logger = get_run_logger()
    _run([PYTHON_BIN, "-m", "pytest", "-v", "test_model_pipeline.py"])
    logger.info("Unit tests passed.")


@task(name="prepare-data")
def prepare_data() -> None:
    """Prépare les données (main.py --prepare)."""
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--prepare"])
    logger.info("Data preparation complete.")


@task(name="train-model")
def train_model() -> None:
    """Entraîne le modèle (main.py --train)."""
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--train"])
    logger.info("Model training complete.")


@task(name="save-model")
def save_model() -> None:
    """Entraîne puis sauvegarde le modèle (main.py --save)."""
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--save"])
    logger.info("Model saved.")


@task(name="load-model")
def load_model() -> None:
    """Charge le modèle sauvegardé (main.py --load)."""
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--load"])
    logger.info("Model loaded.")


@task(name="evaluate-model")
def evaluate_model() -> None:
    """Évalue le modèle sauvegardé (main.py --evaluate)."""
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--evaluate"])
    logger.info("Model evaluation complete.")


@task(name="predict")
def predict() -> None:
    """Prédit pour un client exemple (main.py --predict)."""
    logger = get_run_logger()
    _run([PYTHON_BIN, MAIN_PY, "--predict"])
    logger.info("Prediction complete.")


# ---------------------------------------------------------------- FLOWS
@flow(name="install", log_prints=True)
def flow_install() -> None:
    """Flow : installation des dépendances."""
    install_dependencies()


@flow(name="code", log_prints=True)
def flow_code() -> None:
    """Flow : formatage, qualité, sécurité et tests unitaires."""
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train", log_prints=True)
def flow_train() -> None:
    """Flow : préparation des données et entraînement du modèle."""
    prepare_data()
    train_model()


@flow(name="evaluate", log_prints=True)
def flow_evaluate() -> None:
    """Flow : chargement et évaluation du modèle."""
    load_model()
    evaluate_model()


@flow(name="predict", log_prints=True)
def flow_predict() -> None:
    """Flow : prédiction."""
    predict()


@flow(name="all", log_prints=True)
def flow_all() -> None:
    """Flow : pipeline complet (code, données, modèle)."""
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()
    prepare_data()
    train_model()
    save_model()
    evaluate_model()


FLOWS = {
    "install": flow_install,
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "predict": flow_predict,
    "all": flow_all,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect - Churn")
    parser.add_argument("--flow", required=True, choices=FLOWS.keys())
    args = parser.parse_args()
    FLOWS[args.flow]()
