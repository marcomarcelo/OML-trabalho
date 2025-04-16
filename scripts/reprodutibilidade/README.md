# Reprodutibilidade

Para garantir a **reprodutibilidade** deste projeto foram efetuados os seguintes passos:

## Pré passos

* Instalação do miniconda: https://docs.anaconda.com/free/miniconda/miniconda-install/

## Criação de ambiente reproduzível

1. Criação de um novo ambiente virtual usando o conda, chamado OML.
```
conda create -n OML python=3.10
```
2. Ativação do ambiente acabado de criar
```
conda activate OML
```
3. Instalação das dependências necessárias:
```
conda install pandas numpy scikit-learn
```

4. De forma a tornar a minha experiência reproduzível é necessário exportar o meu ambiente para um ficheiro conda.yaml:
```
conda env export --file conda.yaml
```

5. Como podemos ver, ao adicionar o ficheiro ***conda.yaml*** ao nosso projeto, podemos passar o nosso código a qualquer pessoa, que vai poder muito rapidamente reproduzir as nossas experiências, sem qualquer conhecimento prévio necessário.

## Tornar a amostra reproduzível
Para treinar e testar diferentes modelos com os mesmos dados, de forma a que sejam comparáveis, é necessário evitar a aleatoriedade:

1. Adicionar a linha `np.random.seed(42)` antes de fazer o split
2. Desta forma, independentemente do número de vezes que corrermos o script, o split será sempre o mesmo.
