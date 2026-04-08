from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from experimentos import read_dados
import pandas as pd 
from sklearn.linear_model import LogisticRegression



def run_model_vanilla(treino_path, teste_path):
    # Lê os dados (usando uma função fictícia para simular os dados)
    #X_train, X_test, y_train, y_test = read_dados(treino_path, teste_path)
    df_train = pd.read_csv(treino_path)
    df_test = pd.read_csv(teste_path)

    # Remover colunas de vazamento
    colunas_vazamento = [
        'placar_casa', 'placar_visitante', 'resultado',
        'data', 'ano', 'equipe_casa', 'equipe_visitante'
    ]
    X_train = df_train.drop(columns=colunas_vazamento, errors='ignore')
    X_test = df_test.drop(columns=colunas_vazamento, errors='ignore')

    y_train = df_train['resultado']
    y_test = df_test['resultado']

    # Modelo baseline linear
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, []