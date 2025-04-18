from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from yellowbrick.classifier import ConfusionMatrix


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                                               #   
#                              Algorítimo Naive Bayes                           #
#                                                                               # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Carregar census.pkl

with open('assets/variaveis/census.pkl', 'rb') as f:
    X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

# Criar o modelo Naive Bayes
naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, Y_census_treinamento)

# Fazer previsões
previsoes = naive_census.predict(X_census_teste)
print(accuracy_score(Y_census_teste, previsoes))
print(confusion_matrix(Y_census_teste, previsoes))

cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, Y_census_treinamento)
cm.score(X_census_teste, Y_census_teste)
cm.show()


print(classification_report(Y_census_teste, previsoes))
