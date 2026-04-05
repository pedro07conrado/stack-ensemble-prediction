from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def get_hyper_params_regressao_logistica(X_train, y_train):
    param_grid = {
        'C':        [0.01, 0.1, 1, 10],          # Regularização inversa
        'solver':   ['lbfgs', 'liblinear'],        # Algoritmo de otimização
        'max_iter': [500, 1000]                    # Máximo de iterações
    }

    modelo = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=2,
        scoring='f1_weighted',
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def run_model_regressao_logistica(treino_path, teste_path, useGridSearch=True):
    df_train = pd.read_csv(treino_path)
    df_test  = pd.read_csv(teste_path)

    # Remover colunas de vazamento
    colunas_vazamento = [
        'placar_casa', 'placar_visitante', 'resultado',
        'data', 'ano', 'equipe_casa', 'equipe_visitante'
    ]
    X_train = df_train.drop(columns=colunas_vazamento, errors='ignore')
    X_test  = df_test.drop(columns=colunas_vazamento,  errors='ignore')
    y_train = df_train['resultado']
    y_test  = df_test['resultado']

    # Normalização — importante para convergência da Regressão Logística
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if useGridSearch:
        best_params = get_hyper_params_regressao_logistica(X_train, y_train)
        if best_params is None:
            return None, None, None

        model = LogisticRegression(
            C=best_params['C'],
            solver=best_params['solver'],
            max_iter=best_params['max_iter'],
            random_state=42
        )
    else:
        model       = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, random_state=42)
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params