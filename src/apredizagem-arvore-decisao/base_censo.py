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

# Carregar o dataset de censo
with open('assets/variaveis/census.pkl', 'rb') as f:
    X_censo_treinamento, Y_censo_treinamento, X_censo_teste, Y_censo_teste = pickle.load(f)

# Criar o classificador de árvore de decisão
arvore_censo = DecisionTreeClassifier(criterion="entropy", random_state=0)

# Treinar o classificador com os dados de treinamento
arvore_censo.fit(X_censo_treinamento, Y_censo_treinamento)

# Previsões com os dados de teste
previsao_censo = arvore_censo.predict(X_censo_teste)

accuracy = accuracy_score(Y_censo_teste, previsao_censo)

cm = ConfusionMatrix(arvore_censo)
cm.fit(X_censo_treinamento, Y_censo_treinamento)
cm.score(X_censo_teste, Y_censo_teste)
cm.show()


print(classification_report(Y_censo_teste, previsao_censo))   

