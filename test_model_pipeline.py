"""Tests unitaires de model_pipeline.py."""

from model_pipeline import (
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def test_prepare_data_shapes():
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert x_train.shape[1] == 9
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) > len(x_test)


def test_train_model_predicts_binary():
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(x_train, y_train, n_estimators=10)
    predictions = model.predict(x_test)
    assert set(predictions) <= {0, 1}


def test_save_and_load_model(tmp_path):
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(x_train, y_train, n_estimators=10)
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert (loaded.predict(x_test) == model.predict(x_test)).all()
