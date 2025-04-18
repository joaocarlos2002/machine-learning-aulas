from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                               #   
#                         Algorítimo árvore de decisão                          #
#                                                                               # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Carregar o dataset de risco de crédito
with open('assets/variaveis/risco_credito.pkl', 'rb') as f:
    X_risco_credito_treinamento, Y_risco_credito_treinamento = pickle.load(f)

# Criar o classificador de árvore de decisão
arvore_decisao = DecisionTreeClassifier(criterion="entropy")

# Treinar o classificador com os dados de treinamento
arvore_decisao.fit(X_risco_credito_treinamento, Y_risco_credito_treinamento)


# Atributos mais importantes
atributos_mais_importantes = arvore_decisao.feature_importances_
# print("Atributos mais importantes:" + str(atributos_mais_importantes))


# Prever os resultados para o conjunto de treinamento
plt.figure(figsize=(12, 8))
tree.plot_tree(arvore_decisao, feature_names=["historia", "divida", "garantias", "renda"], class_names=arvore_decisao.classes_, filled=True)
# plt.show()

# previsoes da árvore de decisão
previsoes_arvore = arvore_decisao.predict([[0,0,1,2], [2,0,0,0]])
print("Previsões da árvore de decisão: " + (previsoes_arvore))
