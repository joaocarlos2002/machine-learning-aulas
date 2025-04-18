from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
import pickle
from sklearn.metrics import accuracy_score, classification_report




# Carregar o dataset de risco de crédito
with open('assets/variaveis/credit.pkl', 'rb') as f:
    X_credito_treinamento, Y_credito_treinamento, X_credito_teste, Y_credito_teste = pickle.load(f)



random_forest = RandomForestClassifier(n_estimators= 40, criterion="entropy", random_state=0)

random_forest.fit(X_credito_treinamento, Y_credito_treinamento)

previsoes_risco_credito = random_forest.predict(X_credito_teste)

accuracy = accuracy_score(Y_credito_teste, previsoes_risco_credito)

print(classification_report(Y_credito_teste, previsoes_risco_credito))