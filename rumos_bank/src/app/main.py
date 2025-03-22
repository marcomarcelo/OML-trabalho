import pandas as pd
import json
import mlflow
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, conint

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)

# Define the inputs expected in the request body as JSON
class Request(BaseModel):
    """
    Request model for the API, defining the input structure.

    Attributes:
        LIMIT_BAL (int): Amount of the given credit.
        SEX (int): Gender (1 = male, 2 = female).
        EDUCATION (int): Education (1 = graduate school, 2 = university, 3 = high school, 4 = others).
        MARRIAGE (int): Marital status (1 = married, 2 = single, 3 = divorced).
        AGE (int): Age (years).        
        PAY_0 (float): Repayment status in September, 2005.
        PAY_2 (float): Repayment status in August, 2005.                
        PAY_3 (float): Repayment status in July, 2005.                
        PAY_4 (float): Repayment status in June, 2005.                        
        PAY_5 (float): Repayment status in May, 2005.                        
        PAY_6 (float): Repayment status in April, 2005.                        
        BILL_AMT1 (float): Amount of bill statement in September, 2005.                        
        BILL_AMT2 (float): Amount of bill statement in August, 2005.                        
        BILL_AMT3 (float): Amount of bill statement in July, 2005.                        
        BILL_AMT4 (float): Amount of bill statement in June, 2005.                        
        BILL_AMT5 (float): Amount of bill statement in May, 2005.                        
        BILL_AMT6 (float): Amount of bill statement in April, 2005.                        
        PAY_AMT1 (float): Amount of previous payment in September, 2005.                        
        PAY_AMT2 (float): Amount of previous payment in August, 2005.                        
        PAY_AMT3 (float): Amount of previous payment in July, 2005.                        
        PAY_AMT4 (float): Amount of previous payment in June, 2005.                        
        PAY_AMT5 (float): Amount of previous payment in May, 2005.                        
        PAY_AMT6 (float): Amount of previous payment in April, 2005.            
    """
    LIMIT_BAL: conint(ge=0) = 0 # type: ignore
    SEX: int = 118
    EDUCATION: int = 84
    MARRIAGE: int = 47
    AGE: int = 230
    PAY_0: float = 45.8
    PAY_2: float = 45.8
    PAY_3: float = 45.8
    PAY_4: float = 45.8
    PAY_5: float = 45.8
    PAY_6: float = 45.8
    BILL_AMT1: float = 45.8
    BILL_AMT2: float = 45.8
    BILL_AMT3: float = 45.8
    BILL_AMT4: float = 45.8
    BILL_AMT5: float = 45.8
    BILL_AMT6: float = 45.8
    PAY_AMT1: float = 45.8
    PAY_AMT2: float = 45.8
    PAY_AMT3: float = 45.8
    PAY_AMT4: float = 45.8
    PAY_AMT5: float = 45.8
    PAY_AMT6: float = 45.8


# Variável global para armazenar o modelo carregado
model = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("A aplicação está a iniciar...")  

    mlflow.set_tracking_uri(config['tracking_uri'])

    # Load the registered model specified in the configuration
    model_uri = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri = model_uri)
    
    print(f"LModelo carregado com sucesso: {model_uri}")

    # Carregar o modelo do MLflow apenas uma vez
    #model_uri = "models:/logistic_regression/latest"  # Ajusta conforme o nome do teu modelo
    #model = mlflow.pyfunc.load_model(model_uri)

    yield  # Mantém a aplicação em execução

    print("A aplicação está a encerrar...")  # Este print também deve aparecer



# Criar a aplicação com lifespan
app = FastAPI(lifespan=lifespan)

# Adicionar CORS Middleware depois do app ser criado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (para testes)
    allow_methods=["*"],   # Permite todos os métodos HTTP
    allow_headers=["*"],   # Permite todos os headers
)

@app.post("/default_payment")
async def predict(input: Request):  
    """
    Prediction endpoint that processes input data and returns a model prediction.

    Parameters:
        input (Request): Request body containing input values for the model.

    Returns:
        dict: A dictionary with the model prediction under the key "prediction".
    """

    # Build a DataFrame from the request data
    input_df = pd.DataFrame.from_dict({k: [v] for k, v in input.model_dump().items()})

    # Predict using the model and retrieve the first item in the prediction list
    prediction = app.model.predict(input_df)

    # Return the prediction result as a JSON response
    return {"prediction": prediction.tolist()[0]}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Run the app on port 5003
uvicorn.run(app=app, port=config["service_port"])
