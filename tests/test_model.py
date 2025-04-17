import json
import pytest
import pandas as pd
import mlflow
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.fixture(scope="module")
def model() -> mlflow.pyfunc.PyFuncModel:
    with open('./config/app.json') as f:
        config = json.load(f)
    
    # URI do serviço MLFlow
    mlflow.set_tracking_uri(f"http://localhost:{config['tracking_port']}")    
    
    # Carregar o modelo especificado na configuração
    model_name = config["model_name"]
    model_version = config["model_version"]    
    print (f"Modelo carregado com sucesso: models:/{model_name}@{model_version}")
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_version}"
    )

# Garantir que para o input específico, o modelo retorna 0 (False) 
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

# Garantir que para o input específico, o modelo retorna 1 (True) 
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

# Confirmar se o modelo dá apenas uma previsão
def test_model_out_shape(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 80000.0,
        'SEX': 2,       
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 54,
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

# Assegurar que a saída retorna um valor esperado: 0 ou 1
def test_model_out_range(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 60000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 90,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 10000.0,
        'BILL_AMT2': 11000.0,
        'BILL_AMT3': 12000.0,
        'BILL_AMT4': 13000.0,
        'BILL_AMT5': 14000.0,
        'BILL_AMT6': 15000.0,
        'PAY_AMT1': 1000.0,
        'PAY_AMT2': 1000.0,
        'PAY_AMT3': 1000.0,
        'PAY_AMT4': 1000.0,
        'PAY_AMT5': 1000.0,
        'PAY_AMT6': 1000.0        
    }])
    prediction = model.predict(data=input)
    pred = prediction[0]

    assert pred in [0, 1], f"Previsão inválida: {pred}"

# Verificar consistência da previsão - Para o mesmo input ambas as previsões têm que ser iguais
def test_previsao_deterministica(model: mlflow.pyfunc.PyFuncModel):
    input = pd.DataFrame.from_records([{
        'LIMIT_BAL': 50000.0,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 35,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 5000.0,
        'BILL_AMT2': 6000.0,
        'BILL_AMT3': 7000.0,
        'BILL_AMT4': 8000.0,
        'BILL_AMT5': 9000.0,
        'BILL_AMT6': 10000.0,
        'PAY_AMT1': 1500.0,
        'PAY_AMT2': 1200.0,
        'PAY_AMT3': 1000.0,
        'PAY_AMT4': 1500.0,
        'PAY_AMT5': 2000.0,
        'PAY_AMT6': 2500.0        
    }])

    # Previsão 1
    pred1 = model.predict(data=input)

    # Previsão 2 (mesmo input, deve dar o mesmo valor)
    pred2 = model.predict(data=input)

    # Verificar que ambas previsões são iguais
    assert pred1 == pred2, f"Previsões diferentes: {pred1} != {pred2}"
