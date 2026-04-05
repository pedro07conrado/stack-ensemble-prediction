from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def get_hyper_params_rf(X_train, y_train):
    param_grid = {
        'n_estimators':     [50, 100, 200],
        'max_depth':        [None, 10, 20, 50],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap':        [True, False]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=2,
        scoring='f1_weighted',
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def run_model_random_forest(treino_path, teste_path, useGridSearch=True):
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

    if useGridSearch:
        best_params = get_hyper_params_rf(X_train, y_train)
        if best_params is None:
            return None, None, None

        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            bootstrap=best_params['bootstrap'],
            random_state=42
        )
    else:
        model       = RandomForestClassifier(n_estimators=100, random_state=42)
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params