import os
import re
import time

import numpy as np
import pandas as pd

from experimentos import save_results_csv
from svm import run_model_svm
from knn import run_model_knn
from extraTress import run_model_extra_trees
from regressao_logistica import run_model_regressao_logistica
from lightGbm import run_model_lightgbm


TEMPORADAS = [
    '2008-2009', '2009-2010', '2011-2012',
    '2012-2013', '2013-2014', '2014-2015',
    '2015-2016', '2016-2017', '2018-2019', '2019-2020',
    '2020-2021', '2021-2022', '2022-2023', '2023-2024'
]

JANELAS = [5, 10, 15]

MODELOS = {
    'svm': run_model_svm,
    'knn': run_model_knn,
    'extra_trees': run_model_extra_trees,
    'regressao_logistica': run_model_regressao_logistica,
    'lightgbm': run_model_lightgbm,
}


def numeric_sort_key(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_sorted_files(directory, prefix):
    files = [name for name in os.listdir(directory) if name.startswith(prefix)]
    return sorted(files, key=numeric_sort_key)


def list_sorted_dirs(directory):
    dirs = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    return sorted(dirs, key=numeric_sort_key)


def build_prediction_rows(modelo, janela, temporada, subpasta, treino_file, teste_file, teste_df, details):
    rows = []
    y_true = details['y_true']
    y_pred = details['y_pred']
    prob_1 = details.get('prob_1')

    for index in range(len(y_true)):
        row = {
            'Modelo': modelo,
            'Janela Incremental': janela,
            'Temporada': temporada,
            'Subpasta': subpasta,
            'Treino Arquivo': treino_file,
            'Teste Arquivo': teste_file,
            'Posicao no Teste': index + 1,
            'Data Jogo': teste_df.iloc[index]['data'] if 'data' in teste_df.columns else None,
            'Round': teste_df.iloc[index]['round'] if 'round' in teste_df.columns else None,
            'Equipe Casa': teste_df.iloc[index]['equipe_casa'] if 'equipe_casa' in teste_df.columns else None,
            'Equipe Visitante': teste_df.iloc[index]['equipe_visitante'] if 'equipe_visitante' in teste_df.columns else None,
            'Resultado Real': int(y_true[index]),
            'Previsao': int(y_pred[index]),
            'Acertou': int(y_true[index] == y_pred[index]),
            'Probabilidade Classe 1': float(prob_1[index]) if prob_1 is not None else None,
        }
        rows.append(row)

    return rows


def run_model_for_directory(modelo, janela, temporada, data_dir, runner):
    treino_files = list_sorted_files(data_dir, 'treino_')
    teste_files = list_sorted_files(data_dir, 'teste_')

    detailed_rows = []

    for treino_file, teste_file in zip(treino_files, teste_files):
        treino_path = os.path.join(data_dir, treino_file)
        teste_path = os.path.join(data_dir, teste_file)
        teste_df = pd.read_csv(teste_path)

        accuracy, f1, best_params, details = runner(
            treino_path,
            teste_path,
            True,
            return_details=True
        )

        if accuracy is None:
            continue

        subpasta = os.path.basename(data_dir)
        detailed_rows.extend(
            build_prediction_rows(
                modelo=modelo,
                janela=janela,
                temporada=temporada,
                subpasta=subpasta,
                treino_file=treino_file,
                teste_file=teste_file,
                teste_df=teste_df,
                details=details
            )
        )

    return detailed_rows


def add_running_metrics(predictions_df):
    predictions_df = predictions_df.copy()
    predictions_df['Ordem Jogo Temporada'] = (
        predictions_df
        .groupby(['Modelo', 'Janela Incremental', 'Temporada'])
        .cumcount() + 1
    )

    predictions_df['Acuracia Acumulada'] = (
        predictions_df
        .groupby(['Modelo', 'Janela Incremental', 'Temporada'])['Acertou']
        .expanding()
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )

    return predictions_df


def build_summary(predictions_df):
    grouped = (
        predictions_df
        .groupby(['Modelo', 'Janela Incremental', 'Temporada'], as_index=False)
        .agg(
            Jogos=('Acertou', 'size'),
            Acertos=('Acertou', 'sum'),
            Acuracia=('Acertou', 'mean')
        )
    )
    grouped['F1-Score'] = np.nan
    return grouped


if __name__ == '__main__':
    start_time = time.time()
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    detailed_output_dir = os.path.join(base_path, 'results', 'experimento_03_detalhado')
    os.makedirs(detailed_output_dir, exist_ok=True)

    for modelo, runner in MODELOS.items():
        print(f'\n{"=" * 70}')
        print(f'Executando modo detalhado para: {modelo}')
        print(f'{"=" * 70}')

        all_rows = []

        for janela in JANELAS:
            for temporada in TEMPORADAS:
                temporada_dir = os.path.join(
                    base_path, 'data', 'experimento_03', str(janela), temporada
                )

                if not os.path.exists(temporada_dir):
                    print(f'[AVISO] Pasta nÃ£o encontrada: {temporada_dir}')
                    continue

                subpastas = list_sorted_dirs(temporada_dir)
                season_rows = []

                for subpasta in subpastas:
                    data_dir = os.path.join(temporada_dir, subpasta)
                    season_rows.extend(
                        run_model_for_directory(
                            modelo=modelo,
                            janela=janela,
                            temporada=temporada,
                            data_dir=data_dir,
                            runner=runner
                        )
                    )

                all_rows.extend(season_rows)

                if season_rows:
                    season_accuracy = np.mean([row['Acertou'] for row in season_rows])
                    print(
                        f'[{modelo}] K={janela} | {temporada} | '
                        f'Jogos avaliados: {len(season_rows)} | '
                        f'AcurÃ¡cia: {season_accuracy:.4f}'
                    )

        if not all_rows:
            print(f'[AVISO] Nenhuma previsÃ£o detalhada foi gerada para {modelo}.')
            continue

        predictions_df = pd.DataFrame(all_rows)
        predictions_df = add_running_metrics(predictions_df)

        summary_df = build_summary(predictions_df)

        detailed_path = os.path.join(
            detailed_output_dir,
            f'{modelo}_experimento_03_predicoes_detalhadas.csv'
        )
        summary_path = os.path.join(
            detailed_output_dir,
            f'{modelo}_experimento_03_resumo_temporada.csv'
        )

        save_results_csv(detailed_path, predictions_df.to_dict('records'))
        save_results_csv(summary_path, summary_df.to_dict('records'))

    end_time = time.time()
    print(f'\nTempo total do modo detalhado: {end_time - start_time:.2f} segundos')
