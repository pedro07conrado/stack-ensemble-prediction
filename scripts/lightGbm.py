from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def get_hyper_params_lightgbm(X_train, y_train):
    param_grid = {
        'n_estimators':  [100, 200],
        'max_depth':     [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'num_leaves':    [15, 31],
    }

    modelo = LGBMClassifier(random_state=42, verbose=-1)

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=2,
        scoring='f1_weighted',
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def run_model_lightgbm(treino_path, teste_path, useGridSearch=True):
    df_train = pd.read_csv(treino_path)
    df_test  = pd.read_csv(teste_path)

    colunas_vazamento = [
        'placar_casa', 'placar_visitante', 'resultado',
        'data', 'ano', 'equipe_casa', 'equipe_visitante'
    ]
    X_train = df_train.drop(columns=colunas_vazamento, errors='ignore')
    X_test  = df_test.drop(columns=colunas_vazamento,  errors='ignore')
    y_train = df_train['resultado']
    y_test  = df_test['resultado']

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if useGridSearch:
        best_params = get_hyper_params_lightgbm(X_train, y_train)
        if best_params is None:
            return None, None, None

        model = LGBMClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            num_leaves=best_params['num_leaves'],
            random_state=42,
            verbose=-1
        )
    else:
        model       = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                     num_leaves=31, random_state=42, verbose=-1)
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params