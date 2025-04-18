from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from yellowbrick.classifier import ConfusionMatrix


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                               #   
#                              Algorítimo Naive Bayes                           #
#                                                                               # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Carregar credit.pkl

with open('assets/variaveis/creditv2.pkl', 'rb') as f:
    X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

# Criar o modelo Naive Bayes
naive_credit_credito = GaussianNB()
naive_credit_credito.fit(X_credit_treinamento, Y_credit_treinamento)

# Fazer previsões
previsoes = naive_credit_credito.predict(X_credit_teste)
print(accuracy_score(Y_credit_teste, previsoes))  
print(confusion_matrix(Y_credit_teste, previsoes))

cm = ConfusionMatrix(naive_credit_credito)
cm.fit(X_credit_treinamento, Y_credit_treinamento)
cm.score(X_credit_teste, Y_credit_teste)

cm.show()

print(classification_report(Y_credit_teste, previsoes))