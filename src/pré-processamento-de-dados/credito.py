import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


base_credit = pd.read_csv("assets/credit_data.csv")

# retorna o número de linhas e colunas do dataframe
print(base_credit.head(5))

# Descrição estatística do dataframe
print(base_credit.describe())

print(base_credit[base_credit['income'] > 69995])

print(np.unique(base_credit['default'], return_counts=True))

sns.countplot(x=base_credit['default'])
plt.hist(x = base_credit['age'])
plt.hist(x = base_credit['income'])
plt.hist(x = base_credit['loan'])
plt.show()

# grafico com mais informações 
grafico = px.scatter_matrix(base_credit, dimensions=["age", "income", "loan"], color="default",)
grafico.show()

# ----- tratamento de dados -----

base_credit.loc[base_credit['age'] < 0, 'age']

# APAGAR A COLUNA INTEIRA
base_credit2 = base_credit.drop('age', axis=1)


# APAGAR somente os valores INCORRETOS
base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index, axis=0)
# print(base_credit3.loc[base_credit3['age'] < 0])

# preencher os valores incorretos com a média da coluna
base_credit4 = base_credit.copy()

# Calcular a média da coluna 'age' ignorando os valores negativos
base_credit['age'][base_credit['age'] > 0].mean()

# Preencher os valores negativos com a média calculada
base_credit.loc[base_credit['age' ] < 0, 'age'] = 40.92


# _______ tratamento de valores faltantes _____


# Preencher os valores faltantes com a média da coluna 'income'
base_credit.isnull().sum() # verificar se tem valores nulos

base_credit.loc[pd.isnull(base_credit["age"])]  # retorna os valores nulos da coluna age

base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)  # preenche os valores nulos com a média da coluna age



# _______ Divisao entre previsores e classes _______

X_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

print(X_credit)
print(y_credit)

# _____________Escalonamento de dados _____________

X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()
X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()


# Padronizacao serve para colocar os dados em uma mesma escala, para que o modelo de machine learning consiga aprender melhor, indicado para dados que tem escalas muito diferentes 

from sklearn.preprocessing import StandardScaler

scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()
X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()

# _____________________ dIvisao entre treino e teste ______________________
from sklearn.model_selection import train_test_split

X_credit_treino, X_credit_teste, y_credit_treino, y_credit_teste = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)


#  ____________________ Salvar Base de Dados ______________________
import pickle

with open('assets/variaveis/credit.pkl', 'wb') as f:
    pickle.dump([X_credit_treino, X_credit_teste, y_credit_treino, y_credit_teste], f)