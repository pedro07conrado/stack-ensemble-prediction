from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def get_hyper_params_extra_trees(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 5, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2'],
    }

    modelo = ExtraTreesClassifier(random_state=42)

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


def run_model_extra_trees(treino_path, teste_path, useGridSearch=True):
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

    from sklearn.preprocessing import MinMaxScaler
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if useGridSearch:
        best_params = get_hyper_params_extra_trees(X_train, y_train)
        if best_params is None:
            return None, None, None

        model = ExtraTreesClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            max_features=best_params['max_features'],
            random_state=42,
            n_jobs=-1
        )
    else:
        model       = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        best_params = []

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='weighted')

    return accuracy, f1, best_params