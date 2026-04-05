import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_dados(treino_path, teste_path):
    treino_df = pd.read_csv(treino_path)
    teste_df  = pd.read_csv(teste_path)

    treino_df['victory_casa'] = treino_df['placar_casa'] > treino_df['placar_visitante']
    teste_df['victory_casa']  = teste_df['placar_casa']  > teste_df['placar_visitante']

    X_train = treino_df.drop(['victory_casa'], axis=1)
    y_train = treino_df['victory_casa']

    X_test = teste_df.drop(['victory_casa'], axis=1)
    y_test = teste_df['victory_casa']

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train = X_train[numeric_features]
    X_test  = X_test[numeric_features]

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def save_results_csv(path, results):
    results_df = pd.DataFrame(results)
    results_df.to_csv(path, index=False)
    print(f'Resultados salvos em {path}')