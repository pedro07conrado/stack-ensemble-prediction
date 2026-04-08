import os
import time
import re

import numpy as np

from rede_neural import run_model_rede_neural
from svm import run_model_svm 
from experimentos import save_results_csv
from xgboost_model import run_model_xgboost

modelo = 'xgb'

bests_params = []

def extract_number(file_name):
    # Extrai o número do nome do arquivo.
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def run_models(data_dir):
    # Obter a lista de arquivos no diretório de treino e teste
    files = os.listdir(data_dir)

    # Filtrar e ordenar os arquivos de treino e teste com base nos números extraídos
    treino_files = sorted([f for f in files if f.startswith('treino_') and f.endswith('.csv')], key=extract_number)
    teste_files = sorted([f for f in files if f.startswith('teste_') and f.endswith('.csv')], key=extract_number)

    # Garantir que o número de arquivos de treino e teste é o mesmo
    if len(treino_files) != len(teste_files):
        print("Número de arquivos de treino e teste não coincide.")
        return

    acuracias_from_temporada = []
    f1_scores_from_temporada = []

    # Loop pelos arquivos de treino e teste
    for treino_file, teste_file in zip(treino_files, teste_files):

        treino_path = os.path.join(data_dir, treino_file)
        teste_path = os.path.join(data_dir, teste_file)

        acuracia, f1, best_params = run_model_xgboost(treino_path, teste_path, True)

        if acuracia is None:
            continue
        else:
            bests_params.append(best_params)
            acuracias_from_temporada.append(acuracia)
            f1_scores_from_temporada.append(f1)
    
    media_acuracia = np.mean(acuracias_from_temporada)
    media_f1_score = np.mean(f1_scores_from_temporada)

    return media_acuracia, media_f1_score

if __name__ == '__main__':
    # Início do temporizador
    start_time_all = time.time()

    # Array para guardar os resultados de cada temporada / janela
    results = []

    temporadas = ['2008-2009', '2009-2010', '2011-2012',
                  '2012-2013', '2013-2014', '2014-2015',
                  '2015-2016', '2016-2017', '2018-2019', '2019-2020',
                  '2020-2021', '2021-2022', '2022-2023',
                  '2023-2024'
                 ]
    
    numeros_jogos_treino = [128]
    numeros_jogos_teste = [4,6,8]

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for numero_jogos_treino in numeros_jogos_treino:
        for numero_jogos_teste in numeros_jogos_teste:

            f1_scores_temporadas = []
            acuracias_temporadas = []

            for temporada in temporadas:    
                data_dir = os.path.join(base_path, "data", "experimento_02", temporada, f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}')

                # Rodando modelos para uma temporada
                print(f'Rodando modelo para temporada {temporada}, com janela {numero_jogos_treino} - {numero_jogos_teste}')

                acuracia_from_temporada, f1_score_from_temporada = run_models(data_dir)
                
                acuracias_temporadas.append(acuracia_from_temporada)
                f1_scores_temporadas.append(f1_score_from_temporada)

                results.append({
                    'Janela Flutuante': f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}',
                    'Temporada': temporada,
                    'Acurácia': f'{acuracia_from_temporada:.2f}',
                    'Desvio Padrão Acurácia': '-',
                    'F1-Score': f'{f1_score_from_temporada:.2f}',
                    'Desvio Padrão F1-Score': '-',
                })

                output_dir = os.path.join(base_path, 'results', 'experimento_02')
                output_path = os.path.join(output_dir, f'{modelo}_experimento_02_treino_{numero_jogos_treino}_parcial__2.csv')
                save_results_csv(output_path, results)
            
            # Calcular e adicionar a média e o desvio daquela janela para todas as temporadas
            media_acuracia = np.mean(acuracias_temporadas)
            media_f1_score = np.mean(f1_scores_temporadas)

            desvio_padrao_acuracia = np.std(acuracias_temporadas)
            desvio_padrao_f1_score = np.std(f1_scores_temporadas)

            results.append({
                'Janela Flutuante': f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}',
                'Temporada': '-',
                'Acurácia': f'{media_acuracia:.2f}',
                'Desvio Padrão Acurácia': f'{desvio_padrao_acuracia:.2f}',
                'F1-Score': f'{media_f1_score:.2f}',
                'Desvio Padrão F1-Score': f'{desvio_padrao_f1_score:.2f}',
            })

            output_dir = os.path.join(base_path, 'results', 'experimento_02')
            output_path = os.path.join(output_dir, f'{modelo}_experimento_02_treino_{numero_jogos_treino}_{numero_jogos_teste}__2.csv')
            save_results_csv(output_path, results)

    # Fim do temporizador
    end_time_all = time.time()
    print(f"Tempo total de execução: {end_time_all - start_time_all:.2f} segundos")



