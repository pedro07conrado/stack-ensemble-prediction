from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def get_hyper_params_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],   # numero de vizinhos
        'weights':     ['uniform', 'distance'],  # peso dos vizinhos
        'metric':      ['euclidean', 'manhattan']  # metrica de distância
    }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=2,
        scoring='f1_weighted',
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def run_model_knn(treino_path, teste_path, useGridSearch=True):
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

    # KNN é sensível a escala — normalização obrigatoria
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if useGridSearch:
        best_params = get_hyper_params_knn(X_train, y_train)
        if best_params is None:
            return None, None, None

        model = KNeighborsClassifier(
            n_neighbors=best_params['n_neighbors'],
            weights=best_params['weights'],
            metric=best_params['metric']
        )
    else:
        model       = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params