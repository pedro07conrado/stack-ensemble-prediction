"""
Experimento 03 — SVM com rastreamento da curva de aprendizado incremental.

Salva, para cada predição individual:
  - jogo_index: qual iteração da janela (ex: 16, 17, 18...)
  - y_true:     resultado real do jogo
  - y_pred:     predição do modelo
  - acertou:    1 se acertou, 0 se errou

Isso permite plotar a curva de acurácia ao longo do tempo e verificar
se o modelo melhora conforme acumula mais dados de treino.
"""

import os
import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from experimentos import save_results_csv


modelo = 'svm'


# ---------------------------------------------------------------------------
# Treina e prediz usando os melhores hiperparâmetros (GridSearch)
# Retorna y_true e y_pred ao invés dos scores já calculados
# ---------------------------------------------------------------------------
def run_model_svm_raw(treino_path, teste_path, useGridSearch=True):
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

    if X_train.empty or X_test.empty:
        return None, None

    if useGridSearch:
        param_grid = {
            'C':      [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma':  ['scale', 'auto'],
            'degree': [3, 4]
        }
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            estimator=svm, param_grid=param_grid,
            cv=2, scoring='f1_weighted', verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        model = SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            degree=best_params.get('degree', 3),
            random_state=42
        )
    else:
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test.values, y_pred


# ---------------------------------------------------------------------------
# Processa uma subpasta e retorna lista de (jogo_global_idx, y_true, y_pred)
# ---------------------------------------------------------------------------
def run_models_com_historico(data_dir, jogo_offset=0):
    files        = os.listdir(data_dir)
    treino_files = sorted([f for f in files if f.startswith('treino_')])
    teste_files  = sorted([f for f in files if f.startswith('teste_')])

    registros = []

    for idx, (treino_file, teste_file) in enumerate(zip(treino_files, teste_files)):
        treino_path = os.path.join(data_dir, treino_file)
        teste_path  = os.path.join(data_dir, teste_file)

        y_true_arr, y_pred_arr = run_model_svm_raw(treino_path, teste_path, True)

        if y_true_arr is None:
            continue

        for y_true, y_pred in zip(y_true_arr, y_pred_arr):
            registros.append({
                'jogo_index': jogo_offset + idx + 1,
                'y_true':     int(y_true),
                'y_pred':     int(y_pred),
                'acertou':    int(y_true == y_pred)
            })

    return registros


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    temporadas = [
        '2008-2009', '2009-2010', '2011-2012',
        '2012-2013', '2013-2014', '2014-2015',
        '2015-2016', '2016-2017', '2018-2019', '2019-2020',
        '2020-2021', '2021-2022', '2022-2023', '2023-2024'
    ]

    for jogos_base in [5, 10, 15]:
        print(f'\n{"="*55}')
        print(f'  Janela K={jogos_base}')
        print(f'{"="*55}')

        resultados_resumo   = []      # CSV principal (igual ao script original)
        resultados_detalhes = []      # CSV detalhado por predição

        for temporada in temporadas:
            temporada_dir = os.path.join(
                base_path, "data", "experimento_03", str(jogos_base), temporada
            )

            if not os.path.exists(temporada_dir):
                print(f'  [SKIP] {temporada} — pasta não encontrada')
                continue

            subpastas = sorted([
                p for p in os.listdir(temporada_dir)
                if os.path.isdir(os.path.join(temporada_dir, p))
            ])

            # ----------------------------------------------------------------
            # Coleta todos os y_true / y_pred com índice de jogo
            # ----------------------------------------------------------------
            todas_y_true = []
            todas_y_pred = []
            jogo_offset  = 0

            for pasta in subpastas:
                data_dir  = os.path.join(temporada_dir, pasta)
                registros = run_models_com_historico(data_dir, jogo_offset)

                for r in registros:
                    todas_y_true.append(r['y_true'])
                    todas_y_pred.append(r['y_pred'])
                    resultados_detalhes.append({
                        'Janela Incremental': jogos_base,
                        'Temporada':          temporada,
                        **r
                    })

                jogo_offset += len(registros)

            if not todas_y_true:
                continue

            # ----------------------------------------------------------------
            # Métricas AGREGADAS (y_true e y_pred do conjunto completo)
            # ----------------------------------------------------------------
            accuracy_agregada = accuracy_score(todas_y_true, todas_y_pred)
            f1_agregado       = f1_score(todas_y_true, todas_y_pred, average='weighted')

            resultados_resumo.append({
                'Janela Incremental':  jogos_base,
                'Temporada':           temporada,
                'N_Predicoes':         len(todas_y_true),
                'Acurácia Agregada':   f'{accuracy_agregada:.4f}',
                'F1 Agregado':         f'{f1_agregado:.4f}',
            })

            print(
                f'  [SVM] K={jogos_base} | {temporada} | '
                f'N={len(todas_y_true):4d} | '
                f'Acc={accuracy_agregada:.4f} | F1={f1_agregado:.4f}'
            )

        # --------------------------------------------------------------------
        # Salva CSV resumo (métricas agregadas por temporada)
        # --------------------------------------------------------------------
        output_dir = os.path.join(base_path, 'results', 'experimento_03_curva')
        os.makedirs(output_dir, exist_ok=True)

        resumo_path = os.path.join(
            output_dir, f'{modelo}_experimento_03_janela{jogos_base}_agregado.csv'
        )
        pd.DataFrame(resultados_resumo).to_csv(resumo_path, index=False)
        print(f'\n  Resumo salvo: {resumo_path}')

        # --------------------------------------------------------------------
        # Salva CSV detalhado (1 linha por predição — para plotar a curva)
        # --------------------------------------------------------------------
        detalhe_path = os.path.join(
            output_dir, f'{modelo}_experimento_03_janela{jogos_base}_detalhe.csv'
        )
        pd.DataFrame(resultados_detalhes).to_csv(detalhe_path, index=False)
        print(f'  Detalhe salvo:  {detalhe_path}')

    end_time = time.time()
    print(f'\nTempo total: {end_time - start_time:.2f} segundos')
