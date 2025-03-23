import json
import pytest
import pandas as pd
import mlflow


@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    mlflow.set_tracking_uri(config["tracking_uri"])
    model_name = config["model_name"]
    model_version = config["model_version"]
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )


def test_model_out(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 30,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0


def test_model_dir(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 9999,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1


def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'Pregnancies': 0,
        'Glucose': 30,
        'BloodPressure': 88,
        'SkinThickness': 60,
        'Insulin': 110,
        'BMI': 20.0,
        'DiabetesPedigreeFunction': 0.962,
        'Age': 20
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )