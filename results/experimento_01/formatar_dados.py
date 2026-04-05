"""
Este script lê os resultados de experimentos de diferentes modelos de machine learning a partir de arquivos CSV,
processa os dados e os armazena em um dicionário. Em seguida, converte o dicionário em um DataFrame do pandas,
formata os resultados de acurácia e f1-score, e salva o DataFrame resultante em um novo arquivo CSV.

"""

import csv
import os
import pandas as pd

models = ["svm", "random_forest", "naive_bayes", "rede_neural", "vanilla", "xgboost"]
models_no_underline = ["svm", "random-forest", "naive-bayes", "rede-neural", "vanilla", "xgboost"]
quant_jogos = 15

data_dict = {}

# Seu código aqui
for key, model in enumerate(models):

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    arquivo_path = os.path.join(base_path, f"{model}_experimento_01_{quant_jogos}.csv")

    # Abre o arquivo CSV
    with open(arquivo_path) as file:

        data_results = []

        # Lê os dados do arquivo CSV
        reader = csv.reader(file)
        for row in reader:
            # Processa cada linha de dados
            if row[1] == '-':
                data_results.append(row)

    data_dict[f"{models_no_underline[key]}_{quant_jogos}"] = data_results

print(data_dict)

def dict_to_dataframe(data_dict):
    # Create an empty dataframe
    df = pd.DataFrame()

    # Iterate over the dictionary items
    for key, values in data_dict.items():
        # Extract model and quant_jogos from the key
        model, quant_jogos = key.split("_")

        # Create a new dataframe for each model and quant_jogos
        new_df = pd.DataFrame(columns=[f"{model}_{quant_jogos}_acuracia", f"{model}_{quant_jogos}_f1score"])

        # Iterate over the dictionary items
        for key, values in data_dict.items():
            # Extract model and quant_jogos from the key
            model, quant_jogos = key.split("_")

            # Create a temporary dataframe for each model and quant_jogos
            temp_df = pd.DataFrame(values, columns=["porcentagem_treino", "temporada", "acuracia", "acuracia_desvio", "f1_score", "f1_score_desvio"])

            # Add additional columns for model and quant_jogos to the new dataframe
            new_df[f"{model}_{quant_jogos}_acuracia"] = (temp_df['acuracia'].astype(str).str.replace('.', ',', regex=False) + ' ± ' + temp_df['acuracia_desvio'].astype(str).str.replace('.', ',', regex=False))
            new_df[f"{model}_{quant_jogos}_f1score"] = (temp_df['f1_score'].astype(str).str.replace('.', ',', regex=False) + ' ± ' + temp_df['f1_score_desvio'].astype(str).str.replace('.', ',', regex=False))
                
        return new_df


df = dict_to_dataframe(data_dict)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
arquivo_path = os.path.join(base_path, f"resultados_experimento_01_{quant_jogos}.csv")

df.to_csv(arquivo_path, index=False)
