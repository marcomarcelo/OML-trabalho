# Reprodutibilidade

## Pré passos

* Instalar miniconda: https://docs.anaconda.com/free/miniconda/miniconda-install/

## Criar um ambiente reproduzível

1. Criar um novo ambiente virtual usando o conda, chamado OML 
```
conda create -n OML_trabalho python=3.12
```
2. Ativar o ambiente acabado de criar
```
conda activate OML_trabalho
```
3. Tentar executar o ficheiro `python scripts/reprodutibilidade/split_data.py` . Irá falhar porque este ambiente não tem as dependências necessárias para o ficheiro correr. Assim, devemos instalar as dependências necessárias:
```
conda install pandas numpy scikit-learn
```

4. Voltar a correr o script. Deveria correr sem nenhum issue.

5. De forma a tornar a minha experiência reproduzível, preciso de exportar o meu ambiende para um ficheiro conda.yaml:
```
conda env export --file conda.yaml
```

5. Agora que temos o ficheiro conda.yml, conseguimos facilmente recriar o nosso ambiente em qualquer altura. Por isso vamos desativar e eliminar o ambiente que acabamos de criar:
```
conda deactivate
conda env remove --name OML_tabalho
```

6. Podemos verificar que foi mesmo eliminado com
```
conda env list
```

7. E agora vamos criar um novo ambiente a partir do conda.yml, e ativa-lo:
```
conda env create -f conda.yaml
conda activate OML
```

8. Vamos agora voltar a correr o script com este novo ambiente criado. Tudo deveria funcionar.

9. Como podemos ver, ao adicionar o ficheiro conda.yml ao nosso projeto, podemos passar o nosso código a qualquer pessoa, que vai poder muito rapidamente reproduzir as nossas experiências, sem qualquer conhecimento prévio necessário.

## Tornar a amostra reproduzível

Se corrermos várias vezes o script, vamos perceber que os datasets de treino e teste obtidos mudam aleatoriamente. Isto não é algo desejável, visto que tipicamente queremos treinar e testar diferentes modelos com os mesmos dados, de forma a serem comparáveis.

1. Adicionar a linha `np.random.seed(42)` antes de fazer o split
2. Desta forma, independentemente do número de vezes que corrermos o script, o split será sempre o mesmo.
