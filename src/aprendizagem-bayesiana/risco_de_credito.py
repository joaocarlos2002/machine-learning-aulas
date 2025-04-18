import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_risco_credito = pd.read_csv('assets/risco_credito.csv')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                               #   
#                              PRÉ-PROCESSAMENTO                                #
#                                                                               # 
#                                                                               #
#                                                                               #
#   1 -> Separar as variáveis (previsores e classe)                             #
#   2 -> Aplicar o labelEncode nas variáveis categoricas                        #
#   3 -> Salvar Variaveis                                                       #
#                                                                               #
#                                                                               #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# 1. Separar as variáveis (previsores e classe)

X_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

# 2. Aplicar o labelEncode nas variáveis categoricas
from sklearn.preprocessing import LabelEncoder

label_encodedr_historia = LabelEncoder()
label_encodedr_divida = LabelEncoder()
label_encodedr_garantia = LabelEncoder()
label_encodedr_renda = LabelEncoder()

# Aplicando o labelEncoder nas variáveis categoricas para transformar em numéricas
X_risco_credito[:, 0] = label_encodedr_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encodedr_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encodedr_garantia.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encodedr_renda.fit_transform(X_risco_credito[:, 3])


# 3. Salvar Variaveis
import pickle

with open('assets/variaveis/risco_credito.pkl', 'wb') as f:
    pickle.dump((X_risco_credito, y_risco_credito), f)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                               #   
#                              Algorítimo Naive Bayes                           #
#                                                                               # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from sklearn.naive_bayes import GaussianNB


naive_risco_credito = GaussianNB()

naive_risco_credito.fit(X_risco_credito, y_risco_credito)

naive_risco_credito.classes__ # retorna as classes do modelo
naive_risco_credito.class_count__ # retorna a quantidade de cada classe
naive_risco_credito.class_prior_ # retorna a probabilidade de cada classe
naive_risco_credito.feature_count_ # retorna a quantidade de cada classe para cada variavel
naive_risco_credito.theta_ # retorna a media de cada classe para cada variavel
naive_risco_credito.sigma_ # retorna a variancia de cada classe para cada variavel

previsao = naive_risco_credito.predict([[0,0,1,2], [0,1,0,2], [1,0,1,2], [1,1,0,2]])
# print(previsao)