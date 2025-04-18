from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
import pickle
from sklearn.metrics import accuracy_score, classification_report


# Carregar o dataset de censo
with open('assets/variaveis/census.pkl', 'rb') as f:
    X_censo_treinamento, Y_censo_treinamento, X_censo_teste, Y_censo_teste = pickle.load(f)

# Criar o classificador de árvore de decisão
random_forest = RandomForestClassifier(n_estimators= 4000, criterion="entropy", random_state=0)

# Treinar o classificador com os dados de treinamento
random_forest.fit(X_censo_treinamento, Y_censo_treinamento)

previsores = random_forest.predict(X_censo_teste)

accuracy = accuracy_score(Y_censo_teste, previsores)
print("Acurácia:", accuracy)

print(classification_report(Y_censo_teste, previsores))
