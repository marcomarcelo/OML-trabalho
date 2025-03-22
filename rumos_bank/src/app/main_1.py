from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, conint

import pandas as pd
import json
import mlflow
import uvicorn

# Load the application configuration
with open('./config/app.json') as f:
    config = json.load(f)


# Variável global para armazenar o modelo carregado
model = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Set up actions to perform when the app starts.

    Configures the tracking URI for MLflow to locate the model metadata
    in the local mlruns directory.
    """

    print("A aplicação está a iniciar...")  # Este print deve aparecer

    mlflow.set_tracking_uri(config['tracking_uri'])

    # Load the registered model specified in the configuration
    model = f"models:/{config['model_name']}@{config['model_version']}"
    app.model = mlflow.pyfunc.load_model(model_uri = model)
    
    print(f"Loaded model ")

    yield  # Mantém a aplicação rodando

    print("A aplicação está a encerrar...")  # Este print também deve aparecer

# Criar a aplicação com lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow all origins, methods, and headers for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")  # A rota precisa de estar definida
async def root():
    return {"message": "API está a funcionar"}

@app.get("/api")
async def my_endpoint():
    return {"message": "API está a funcionar"}

@app.post("/has_diabetes")
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

# Run the app on port 5003
uvicorn.run(app=app, port=config["port"])