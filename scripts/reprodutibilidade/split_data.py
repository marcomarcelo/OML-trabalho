import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Criar alguns dados de exemplo
data = {'Nome': ['Manel', 'Joana', 'Francisco', 'David', 'Catarina'],
        'Idade': [25, 30, 35, 40, 65],
        'Nota': [85, 90, 88, 92, 99]}

# Criar um DataFrame
df = pd.DataFrame(data)
print("DataFrame Original:")
print(df)
print()

# Definir a random seed
np.random.seed(42)

# Criar datasets para treino e test de modelos
train, test = train_test_split(df, test_size=0.2)
# Imprimir os datasets
print("Train:")
print(train)
print()
print("Test:")
print(test)
print()
