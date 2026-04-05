import os
import re
import pandas as pd

def extract_number(file_name):
    # Extrai o número do nome do arquivo.
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def calcular_media_temporada(data_dir):
    # Obter a lista de arquivos no diretório de treino e teste
    files = os.listdir(data_dir)

    # Filtrar e ordenar os arquivos de treino e teste com base nos números extraídos
    treino_files = sorted([f for f in files if f.startswith('treino_') and f.endswith('.csv')], key=extract_number)
    teste_files = sorted([f for f in files if f.startswith('teste_') and f.endswith('.csv')], key=extract_number)

    # Garantir que o número de arquivos de treino e teste é o mesmo
    if len(treino_files) != len(teste_files):
        print("Número de arquivos de treino e teste não coincide.")
        return None

    medias_vitorias = []

    # Loop pelos arquivos de treino e teste
    for treino_file, teste_file in zip(treino_files, teste_files):
        treino_path = os.path.join(data_dir, treino_file)
        teste_path = os.path.join(data_dir, teste_file)

        # Ler os dados
        treino_df = pd.read_csv(treino_path)
        teste_df = pd.read_csv(teste_path)

        # Calcular médias de vitórias dos times da casa
        media_treino = (treino_df['placar_casa'] > treino_df['placar_visitante']).mean()
        media_teste = (teste_df['placar_casa'] > teste_df['placar_visitante']).mean()

        medias_vitorias.append(media_treino)
        medias_vitorias.append(media_teste)

    # Retornar a média geral da temporada
    return round(sum(medias_vitorias) / len(medias_vitorias), 2)

def salvar_medias_csv(medias_por_temporada, output_path):
    # Criar um DataFrame com as médias por temporada
    df = pd.DataFrame(medias_por_temporada)
    # Salvar como CSV
    df.to_csv(output_path, index=False, float_format='%.2f')

if __name__ == '__main__':
    # Configurar o diretório base
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    temporadas = ['2008-2009', '2009-2010', '2011-2012',
                  '2012-2013', '2013-2014', '2014-2015',
                  '2015-2016', '2016-2017', '2018-2019', '2019-2020',
                  '2020-2021', '2021-2022', '2022-2023',
                  '2023-2024'
                 ]
    numeros_jogos_treino = [128]
    numeros_jogos_teste = [1]

    medias_por_temporada = []

    for numero_jogos_treino in numeros_jogos_treino:
        for numero_jogos_teste in numeros_jogos_teste:
            for temporada in temporadas:
                data_dir = os.path.join(base_path, "data", "experimento_02", temporada, f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}')
                print(f'Calculando média para temporada {temporada}, janela {numero_jogos_treino}-{numero_jogos_teste}')

                media_temporada = calcular_media_temporada(data_dir)
                if media_temporada is not None:
                    medias_por_temporada.append({
                        'Temporada': temporada,
                        'Média Vitórias Casa': media_temporada
                    })

    # Salvar resultados em um arquivo CSV
    output_path = os.path.join(base_path, "results", "medias_vitorias_temporada.csv")
    salvar_medias_csv(medias_por_temporada, output_path)

    print(f'Resultados salvos em {output_path}')
