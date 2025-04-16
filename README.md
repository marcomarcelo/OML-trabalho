
# Operacionalização de Machine Learning 

Este repositório descreve o processo de treino, implementação e operacionalização de modelos de Machine Learning para prever quais os clientes que não irão cumprir os prazos do próximo pagamento. Contém instruções detalhadas sobre como utilizar o MLFlow e a FastAPI para servir os modelos, bem como informações sobre testes e ferramentas auxiliares.

Foi proposto o desafio para evitar que o "Rumos Bank" deixe de perder tanto dinheiro devido à quantidade de créditos que fornece e que não são pagos dentro do prazo. Pretende-se prever quais os clientes que não irão cumprir os prazos e passar o modelo para produção de uma forma rápida e eficiente:
* Definição do ambiente do projecto
* Modelos registados e versionados num Model Registry
* Testes de serviço e do modelo
* Serviços containerizados: 
    - MLFlow-Tracking
    - Default-payment-prediction-service
* Container do serviço é built, testado e enviado para um container registry num pipeline de CICD

# Índice
- [Prever clientes maus pagadores](#prever-clientes-maus-pagadores)
    - [Modelos](#modelos)
    - [Webservice](#webservice)    
    - [Tests](#tests)
- [Comandos Úteis](#comandos-úteis)
    - [Anaconda](#tests)
    - [Docker](#docker)
    - [Docker Compose](#docker-compose)
    - [GitHub - Criar Package](#github---criar-package)
    - [CI-CD](#ci-cd)

# Prever clientes maus pagadores

## Modelos

Os modelos foram treinados no notebook ***rumos_bank_lending_prediction.ipynb*** e trackeados e registados através do MLFlow.

## Webservice

O modelo foi disponibilizado através de uma API, utilizando a framework FastAPI.

Esta API expõe o endpoint `/default_payment` na qual espera receber as features de input do modelo (em formato json, no body do pedido) e retorna a previsão dada pelo modelo.   
Para testar a API basta correr o notebook `test_requests.ipynb`, na secção de `mlflow serve`.

### Com a FastAPI

No python script `src\app\main.py` foi desenvolvida uma aplicação simples com a fastapi.

O nome e a versão do modelo registado a ser utilizado na app tem que ser especificado no ficheiro de configuração presente na diretoria `config` no ficheiro `app.json`.

Para correr a app: com o ambiente deste projeto ativo, na raiz do projeto, executar o comando abaixo:

```
python ./src/app/main.py
```



## Tests

Para testarmos o nosso modelo registado, utilizou-se a framework de Python `pytest`.

Os testes estão presentes nos scripts de Python:
* Para testar operacionalidade do serviço:
`tests\test_service.py`
* Para testar o modelo: 
`tests\test_model.py`
    - **test_model_out_false**: testa o output do modelo e verifica se coincide com o output esperado: prediction = false
    - **test_model_out_true**: testa o output do modelo e verifica se coincide com o output esperado: prediction = true
    - **test_model_out_shape**: testa se a shape do output do modelo coincide com a shape esperada

Para correr estes testes: com o ambiente deste projeto ativo, na raiz do projeto, executar o comando abaixo:

```
python -m pytest tests
```

# Comandos Úteis

## Anaconda

Instalar: https://docs.anaconda.com/miniconda/miniconda-install/

Nota: Em Windows, para o comando `conda` funcionar corretamente em qualquer janela, devem abrir o Anaconda Prompt e correr `conda init`. A partir desse momento, podem fechar o Anaconda Prompt, reiniciar todos os terminais e usar o terminal normal do Windows para correr comandos do `conda`.

Nota 2: Em Windows, poderá ser necessário executar o comando `Set-ExecutionPolicy -ExecutionPolicy Unrestricted` na PowerShell caso não estejam autorizados a correr comandos na consola.

`conda create -n <env-name> python=<python-version>`: comando utilizado para criar um novo ambinete de anaconda com a versão `<python-version>` (que deverá ser substituido pela versão do Python que queremos usar) do Python e com o nome `<env-name>`(que deverá ser substituido pelo nome que queremos dar ao ambiente). [Link para a documentação do conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

`conda activate <env-name>`: activa o ambiente `<env-name>` do Anaconda

`conda deactivate`: desactiva o ambiente atualmente activo do Anaconda

`conda env export --file conda.yaml`: comando utilizado para exportar o ambiente atual do anaconda para um ficheiro yaml. [Link para a documentação do conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file)

`conda env create -f conda.yaml`: comando utilizado para criar um ambiente do Anaconda a partir de um ficheiro que contenha a especificação de um ambiente do Anaconda. Este novo ambiente ficará com o nome que está especificado no ficheiro. Para usar um outro nome basta adicionar ao comando `-n <env-name>` (substituindo `<env-name>` pelo nome qu querem que fique). [Link para a documentação do conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)   

*Configuração*: **conda.yaml**


## Docker

### Serviço MLFlow

Inicializar o serviço MLFlow UI num container do docker:

```
docker run -p 5000:5000 -v ./mlruns:/mlruns ghcr.io/mlflow/mlflow mlflow ui --port 5000 --host 0.0.0.0 --backend-store-uri ./mlruns --artifacts-destination ./mlruns
```

Endereço de acesso ao MLFLow: 
http://localhost:5000


### Serviço do modelo 

Criar a imagem no docker do serviço de predição (FastAPI).   
*Configuração*: **Dockerfile.service**
```
docker build -t service -f Dockerfile.Service .
```
Endereço de acesso à documentação do serviço: 
http://localhost:5002/docs


## Docker Compose
Inicializar os dois serviços em simultâneo.   
*Configuração*: **docker-compose.yaml**
```
docker compose up
```

## GitHub - Criar Package
Enviar a imagem do serviço para o GitHub.
```
docker push ghcr.io/marcomarcelo/default-payment-prediction-service
```

https://github.com/marcomarcelo


## CI-CD
Integrar e passar serviço para produção (deploy) através da construção de um pipeline.
*Configuração*: **.github/workflows/cicd.yaml**

Acionado com:
```
git push
```

Ver o log dos workflows:   
https://github.com/marcomarcelo/OML-trabalho/actions
