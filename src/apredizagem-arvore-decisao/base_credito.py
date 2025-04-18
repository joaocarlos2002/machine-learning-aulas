from sklearn import tree
from yellowbrick.classifier import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                               #   
#                         Algorítimo árvore de decisão                          #
#                                                                               # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Carregar o dataset de crédito
with open('assets/variaveis/credit.pkl', 'rb') as f:
    X_credito_treinamento, Y_credito_treinamento, X_credito_teste, Y_credito_teste = pickle.load(f)

# Criar o classificador de árvore de decisão
arvore_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)

# Treinar o classificador com os dados de treinamento
arvore_credit.fit(X_credito_treinamento, Y_credito_treinamento)

# Previsões com os dados de teste
previsao_credit = arvore_credit.predict(X_credito_teste)

accuracy = accuracy_score(Y_credito_teste, previsao_credit)

cm = ConfusionMatrix(arvore_credit)
cm.fit(X_credito_treinamento, Y_credito_treinamento)
cm.score(X_credito_teste, Y_credito_teste)
# cm.show()

# print(classification_report(Y_credito_teste, previsao_credit))


previsores = ["income", "age", "loan"]
fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(20, 20))
tree.plot_tree(arvore_credit, feature_names=previsores, filled=True);
plt.savefig("assets/variaveis/arvore_credito.png")