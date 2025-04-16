import json
import pytest
import pandas as pd
import mlflow

@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    
    model_name = config["model_name"]
    model_version = config["model_version"]

    mlflow.set_tracking_uri("http://localhost:5000")
    #mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}") 

    # Load the registered model specified in the configuration
    model = f"models:/{config['model_name']}@{config['model_version']}"
    print(f"Modelo carregado com sucesso: {model}")   

    print (f"Modelo carregado com sucesso: models:/{model_name}@{model_version}")
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )

def test_model_out_false(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 80000.0,
        'SEX': 2,       
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 34,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': -1,
        'PAY_6': -1,
        'BILL_AMT1': 55933.0,
        'BILL_AMT2': 11865.0,
        'BILL_AMT3': 4602.0,
        'BILL_AMT4': 34197.0,
        'BILL_AMT5': 27398.0,
        'BILL_AMT6': 28646.0,
        'PAY_AMT1': 4000.0,
        'PAY_AMT2': 2333.0,
        'PAY_AMT3': 3032.0,
        'PAY_AMT4': 28298.0,
        'PAY_AMT5': 2000.0,
        'PAY_AMT6': 2000.0        
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 0

def test_model_out_true(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 30000.0,
        'SEX': 2,       
        'EDUCATION': 1,
        'MARRIAGE': 2,
        'AGE': 23,
        'PAY_0': 2,
        'PAY_2': 2,
        'PAY_3': 2,
        'PAY_4': 2,
        'PAY_5': 2,
        'PAY_6': 2,
        'BILL_AMT1': 35932.0,
        'BILL_AMT2': 31864.0,
        'BILL_AMT3': 28635.0,
        'BILL_AMT4': 30127.0,
        'BILL_AMT5': 30525.0,
        'BILL_AMT6': 29793.0,
        'PAY_AMT1': 1800.0,
        'PAY_AMT2': 150.0,
        'PAY_AMT3': 2250.0,
        'PAY_AMT4': 1000.0,
        'PAY_AMT5': 0.0,
        'PAY_AMT6': 700.0
    }])
    prediction = model.predict(data=input)
    assert prediction[0] == 1

def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 80000.0,
        'SEX': 2,       
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 34,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': -1,
        'PAY_6': -1,
        'BILL_AMT1': 55933.0,
        'BILL_AMT2': 11865.0,
        'BILL_AMT3': 4602.0,
        'BILL_AMT4': 34197.0,
        'BILL_AMT5': 27398.0,
        'BILL_AMT6': 28646.0,
        'PAY_AMT1': 4000.0,
        'PAY_AMT2': 2333.0,
        'PAY_AMT3': 3032.0,
        'PAY_AMT4': 28298.0,
        'PAY_AMT5': 2000.0,
        'PAY_AMT6': 2000.0        
    }])
    prediction = model.predict(data=input)
    assert prediction.shape == (1, )