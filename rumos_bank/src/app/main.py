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
        Pregnancies (int): Number of pregnancies.
        Glucose (int): Plasma glucose concentration.
        BloodPressure (int): Diastolic blood pressure.
        SkinThickness (int): Skin thickness in mm.
        Insulin (int): 2-Hour serum insulin.
        BMI (float): Body Mass Index.
        DiabetesPedigreeFunction (float): Diabetes pedigree function.
        Age (int): Age of the individual.
    """
    Pregnancies: conint(ge=0) = 0 # type: ignore
    Glucose: int = 118
    BloodPressure: int = 84
    SkinThickness: int = 47
    Insulin: int = 230
    BMI: float = 45.8
    DiabetesPedigreeFunction: float = 0.551
    Age: int = 31


# Variável global para armazenar o modelo carregado
model = None  





# Variável global para armazenar o modelo carregado
model = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("A aplicação está a iniciar...")  # Este print deve aparecer

    # Carregar o modelo do MLflow apenas uma vez
    model_uri = "models:/logistic_regression/latest"  # Ajusta conforme o nome do teu modelo
    model = mlflow.pyfunc.load_model(model_uri)
    print("Modelo carregado com sucesso!")

    yield  # Mantém a aplicação rodando

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

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Run the app on port 5003
uvicorn.run(app=app, port=config["service_port"])
