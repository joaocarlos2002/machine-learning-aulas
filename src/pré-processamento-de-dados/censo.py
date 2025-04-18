import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



#  +++++ BASE DE DADOS DO CENSO ++++++

# CARREGAR A BASE DE DADOS DO CENSO

# 15 colunas x rows
base_censo = pd.read_csv("assets/census.csv")

esta = base_censo.describe()  # Descrição estatística do dataframe

# Não é necessário fazer o tratamento de dados, pois a base já está limpa.
valores_nulos = base_censo.isnull().sum()  # verificar se tem valores nulos

# print(valores_nulos)  
# print(esta)

#  +++++++ VISUALIZAÇÃO ++++++
income = np.unique(base_censo['income'], return_counts=True)  # retorna os valores únicos da coluna income e a quantidade de vezes que aparece

# sns.countplot(x=base_censo['income'])  # gráfico de barras da coluna income

# plt.hist(x = base_censo['age'])  # histograma da coluna age
# plt.hist(x = base_censo['education-num'])  # histograma da coluna education-num
# plt.hist(x = base_censo['hour-per-week'])  # histograma da coluna hours-per-week

plt.show()  

# grafico = px.treemap(base_censo, path=["workclass", "age"])
# grafico = px.treemap(base_censo, path=["occupation", "relationship"],)
# grafico = px.parallel_categories(base_censo, dimensions=['occupation', 'relationship', 'income'], color='age',
#                               color_continuous_scale=px.colors.sequential.Inferno,)
# grafico.show()

# ++++++ DIVISÃO DE PREVISORES E CLASSE ++++++
X_censo = base_censo.iloc[:, 0:14].values
y_censo = base_censo.iloc[:, 14].values

# +++++++ TRATAMENTO DE ATRIBUTOS CATEGÓRICOS ++++++

# Label Encoding -> transforma os atributos categóricos em numéricos

from sklearn.preprocessing import LabelEncoder

label_enconder_teste = LabelEncoder()
teste = label_enconder_teste.fit_transform(X_censo[:, 1])  # transforma a coluna 1 em numérico

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()



X_censo[:, 1] = label_encoder_workclass.fit_transform(X_censo[:, 1])  # transforma a coluna 1 em numérico
X_censo[:, 3] = label_encoder_education.fit_transform(X_censo[:, 3])  # transforma a coluna 3 em numérico
X_censo[:, 5] = label_encoder_marital.fit_transform(X_censo[:, 5])  # transforma a coluna 5 em numérico
X_censo[:, 6] = label_encoder_occupation.fit_transform(X_censo[:, 6])  # transforma a coluna 6 em numérico
X_censo[:, 7] = label_encoder_relationship.fit_transform(X_censo[:, 7])  # transforma a coluna 7 em numérico
X_censo[:, 8] = label_encoder_race.fit_transform(X_censo[:, 8])  # transforma a coluna 8 em numérico
X_censo[:, 9] = label_encoder_sex.fit_transform(X_censo[:, 9])  # transforma a coluna 9 em numérico
X_censo[:, 13] = label_encoder_country.fit_transform(X_censo[:, 13])  # transforma a coluna 13 em numérico

# One Hot Encoding -> transforma os atributos categóricos em numéricos, mas cria novas colunas para cada valor único da coluna original

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

len(np.unique(base_censo['occupation']))  

len(np.unique(base_censo['workclass']))  # 1 0 0 0 0 0 0 0, 0 0 0 0 1 0 0 0 0

one_hot_encoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9,13])], remainder='passthrough') 

X_censo = one_hot_encoder_census.fit_transform(X_censo).toarray()  # transforma os atributos categóricos em numéricos e cria novas colunas para cada valor único da coluna original



#  +++++ ESCALONAMENTO DE VALORES ++++++

from sklearn.preprocessing import StandardScaler

# padronizacao -> transforma os dados para que tenham média 0 e desvio padrão 1

scale_censo = StandardScaler()
X_censo = scale_censo.fit_transform(X_censo)  # transforma os dados para que tenham média 0 e desvio padrão 1

#  ++++++ DIVISÃO DE TREINO E TESTE ++++++
from sklearn.model_selection import train_test_split

# divisão em treino e teste, 80% treino e 20% teste
X_censo_treino, X_censo_teste, y_censo_treino, y_censo_teste = train_test_split(X_censo, y_censo, test_size=0.15, random_state=0)

#  +++++ SALVAR A BASE DE DADOS ++++++
import pickle

with open('assets/variaveis/censo.pkl', 'wb') as f:
    pickle.dump((X_censo_treino, X_censo_teste, y_censo_treino, y_censo_teste), f)