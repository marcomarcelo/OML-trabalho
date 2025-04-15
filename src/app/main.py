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
        LIMIT_BAL (float): Amount of the given credit.
        SEX (int): Gender (1 = male, 2 = female).
        EDUCATION (int): Education (1 = graduate school, 2 = university, 3 = high school, 4 = others).
        MARRIAGE (int): Marital status (1 = married, 2 = single, 3 = divorced).
        AGE (int): Age (years).        
        PAY_0 (int): Repayment status in September, 2005.
        PAY_2 (int): Repayment status in August, 2005.                
        PAY_3 (int): Repayment status in July, 2005.                
        PAY_4 (int): Repayment status in June, 2005.                        
        PAY_5 (int): Repayment status in May, 2005.                        
        PAY_6 (int): Repayment status in April, 2005.                        
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
    #LIMIT_BAL: conint(ge=0) = 20000 # type: ignore
    LIMIT_BAL: float = 80000.0	
    SEX: int = 2
    EDUCATION: int = 2
    MARRIAGE: int = 1
    AGE: int = 34
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = -1
    PAY_6: int = -1
    BILL_AMT1: float = 55933.0
    BILL_AMT2: float = 11865.0
    BILL_AMT3: float = 4602.0
    BILL_AMT4: float = 34197.0
    BILL_AMT5: float = 27398.0
    BILL_AMT6: float = 28646.0
    PAY_AMT1: float = 4000.0
    PAY_AMT2: float = 2333.0
    PAY_AMT3: float = 3032.0
    PAY_AMT4: float = 28298.0
    PAY_AMT5: float = 2000.0
    PAY_AMT6: float = 2000.0

# Variável global para armazenar o modelo carregado
model = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("A aplicação está a iniciar...")  

    # Carregar o modelo do MLflow
    mlflow.set_tracking_uri(f"{config['tracking_base_url']}:{config['tracking_port']}")

    # Load the registered model specified in the configuration
    model = f"models:/{config['model_name']}@{config['model_version']}"    
    app.model = mlflow.pyfunc.load_model(model_uri = model)  
    print(f"Modelo carregado com sucesso: {model}")   

    yield  # Mantém a aplicação em execução

    print("A aplicação está a encerrar...")  

# # Criar a aplicação com lifespan
app = FastAPI(lifespan=lifespan)

# Adicionar CORS Middleware depois do app ser criado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Permite todas as origens (para testes)
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

# Run the app on port 5002
uvicorn.run(app=app, host="0.0.0.0", port=config["service_port"])
