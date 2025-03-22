# Operacionalização de Machine Learning

Este repositório descreve o processo de treino, implementação e operacionalização de modelos de Machine Learning para prever a probabilidade de um indivíduo ter diabetes. Contém instruções detalhadas sobre como utilizar o MLFlow e a FastAPI para servir os modelos, bem como informações sobre testes e ferramentas auxiliares.

Estes ficheiros e pastas são utilizados no âmbido do curso Operacionalização de Machine Learning. Os slides podem ser obtidos [daqui](https://docs.google.com/presentation/d/1nJI_xrpfoA2itFLAVaF9O4SjYUerUkByfPSLog04lEo/edit?usp=sharing).


# Índice
- [Prever Diabetes](#prever-diabetes)
    - [Modelos](#modelos)
    - [Webservice](#webservice)
    - [UI](#ui)
    - [Tests](#tests)
- [Comandos Úteis](#comandos-úteis)

# Prever Diabetes

## Modelos

Os modelos foram treinados através da pasta notebooks.
Estes modelos foram trackeados e registados através do MLFlow.

## Webservice

Para expormos o nosso modelo registado numa API podemos utilizar ou a funcionalidade de models serve do mlflow ou a framework FastAPI.

### Com o MlFlow

Para expor uma api utilizando a funcionalidade de serving do mlflow basta correr, na raiz do projeto (pasta rumos), os seguintes comandos:

Windows (psh):
```
$env:MLFLOW_TRACKING_URI="./mlruns"
mlflow models serve -m models:/logistic_reg/1 --port=5001 --no-conda
```

Linux:
```
export MLFLOW_TRACKING_URI=./mlruns
mlflow models serve -m models:/logistic_reg/1 --port=5001 --no-conda
```

O nome do modelo (logistic_reg) e a versão do modelo (1) devem ser substituídos pelo nome com que o modelo foi registado e a versão do mesmo a ser utilizada, respetivamente.

Esta funcionalidade do mlflow utiliza a especificação de ambiente criada automaticamente pelo mlflow para o modelo, para criar um ambiente de conda para o modelo e servir o modelo isolado nesse ambiente virtual de conda.

Esta API expõe o endpoint `/invocations` na qual espera receber as features de input do modelo e retorna a previsão dada pelo modelo. Para testar a API basta correr o notebook `test_requests.ipynb`, na secção de `mlflow serve`.

### Com a FastAPI

No python script `src\app\main.py` foi desenvolvida uma aplicação simples com a fastapi.

O nome e a versão do modelo registado a ser utilizado na app tem que ser especificado no ficheiro de configuração presente na diretoria `config` no ficheiro `app.json`.

Esta app expõe o endpoint `/predict` na qual espera receber as features de input do modelo (em formato json, no body do pedido) e retorna a previsão dada pelo modelo.

Para correr a app: com o ambiente deste projeto ativo, na raiz do projeto (pasta rumos), executar o comando abaixo:

```
python ./src/app/main.py
```

Para testar se o modelo ficou corretamente exposto na app, temos 2 opções:
- http://127.0.0.1:5002/docs
- Correr a secção `FastAPI` do notebook `test_requests.ipynb` ou então podemos utilizar a página html presente na diretoria `frontend` deste projeto e realizar um pedido (e ver a resposta) através desse frontend.

## UI

Para adicionar uma UI ao nosso webservice da FastApi, apenas necessitamos de abrir o ficheiro `frontend/form.html` num browser.

## Tests

Para testarmos o nosso modelo registado utilizou-se a framework de Python `pytest`.

Os testes estão presentes no script de Python `tests\random_forest\test_rf_out.py`:

* `test_model_out`: testa o output do modelo e verifica se coincide com o output esperado
* `test_model_out_shape`: testa se a shape do output do modelo coincide com a shape esperada

Para correr estes testes: com o ambiente deste projeto ativo, na raiz do projeto (pasta rumos), executar o comando abaixo:

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

`conda env list`: comando utilizar para listar os ambientes que temos do Anaconda. Útil também para verificarmos onde estão instalados os ambientes do Anaconda

`conda env export --file conda.yaml`: comando utilizado para exportar o ambiente atual do anaconda para um ficheiro yaml. [Link para a documentação do conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file)

`conda env export --from-history --file conda.yml`: comando utilizado para exportar o ambiente atual do anaconda para um ficheiro yaml, **incluindo apenas pacotes explicitamente pedidos** graças à flag `--from-history`. Esta flag é normalmente utilizada para tentar não incluir dependências que não funcionam cross platform, como é referido [na documentação do conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-an-environment-file-across-platforms) 

`conda env create -f conda.yaml`: comando utilizado para criar um ambiente do Anaconda a partir de um ficheiro que contenha a especificação de um ambiente do Anaconda. Este novo ambiente ficará com o nome que está especificado no ficheiro. Para usar um outro nome basta adicionar ao comando `-n <env-name>` (substituindo `<env-name>` pelo nome qu querem que fique). [Link para a documentação do conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)


## Jupyter

Para instalar e correr, numa consola correr:

```
conda install jupyter
jupyter lab
```

Para registar o venv, correr numa consola:

```
python -m ipykernel install --user --name=OML
```

## MLFlow

Inicializar a UI:

```
mlflow ui --backend-store-uri ./mlruns
```

## Windows

Por padrão, o Windows bloqueia a execução de scripts não assinados no PowerShell. Pode-se alterar essa configuração com:

```
Set-ExecutionPolicy Unrestricted -Scope LocalMachine -Force
```