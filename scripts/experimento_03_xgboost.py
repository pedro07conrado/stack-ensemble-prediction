import os
import time
import numpy as np
from experimentos import save_results_csv
from xgboost_model import run_model_xgboost

modelo = 'xgboost'


def run_models(data_dir):
    files        = os.listdir(data_dir)
    treino_files = sorted([f for f in files if f.startswith('treino_')])
    teste_files  = sorted([f for f in files if f.startswith('teste_')])

    acuracias = []
    f1_scores = []

    for treino_file, teste_file in zip(treino_files, teste_files):
        treino_path = os.path.join(data_dir, treino_file)
        teste_path  = os.path.join(data_dir, teste_file)

        acuracia, f1, best_params = run_model_xgboost(treino_path, teste_path, True)

        if acuracia is not None:
            acuracias.append(acuracia)
            f1_scores.append(f1)

    return acuracias, f1_scores


if __name__ == '__main__':
    start_time = time.time()

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    temporadas = [
       # '2008-2009' ,
       #  '2009-2010', '2011-2012',
        #'2012-2013', '2013-2014', '2014-2015',
        #'2015-2016', '2016-2017',
         '2018-2019', '2019-2020',
        #'2020-2021', '2021-2022', '2022-2023', '2023-2024'
    ]

    for jogos_base in [5, 10, 15]:
        results = []
        for temporada in temporadas:
            temporada_dir = os.path.join(
                base_path, "data", "experimento_03", str(jogos_base), temporada
            )

            if not os.path.exists(temporada_dir):
                print(f'Pasta não encontrada: {temporada_dir} — pulando.')
                continue

            subpastas = sorted([
                p for p in os.listdir(temporada_dir)
                if os.path.isdir(os.path.join(temporada_dir, p))
            ])

            todas_acuracias = []
            todas_f1        = []

            for pasta in subpastas:
                data_dir             = os.path.join(temporada_dir, pasta)
                acuracias, f1_scores = run_models(data_dir)
                todas_acuracias.extend(acuracias)
                todas_f1.extend(f1_scores)

            media_acuracia = np.mean(todas_acuracias) if todas_acuracias else 0
            media_f1       = np.mean(todas_f1)        if todas_f1        else 0

            results.append({
                'Janela Incremental': jogos_base,
                'Temporada':          temporada,
                'Acurácia':           f'{media_acuracia:.4f}',
                'F1-Score':           f'{media_f1:.4f}'
            })

            print(f'[XGBoost] K={jogos_base} | {temporada} → Acurácia: {media_acuracia:.4f}')

        output_dir  = os.path.join(base_path, 'results', 'experimento_03_xgboost')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f'{modelo}_experimento_03_janela{jogos_base}_2018-2019.csv'
        )
        save_results_csv(output_path, results)

    end_time = time.time()
    print(f'\nTempo total: {end_time - start_time:.2f} segundos')