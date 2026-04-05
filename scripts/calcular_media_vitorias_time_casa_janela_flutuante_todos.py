import os
import re
import pandas as pd

def extract_number(file_name):
    # Extrai o número do nome do arquivo.
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

def calcular_media_vitorias_casa(data_dir, temporada, numero_jogos_treino, numero_jogos_teste):
    # Obter a lista de arquivos no diretório de treino e teste
    files = os.listdir(data_dir)

    # Filtrar e ordenar os arquivos de treino e teste com base nos números extraídos
    treino_files = sorted([f for f in files if f.startswith('treino_') and f.endswith('.csv')], key=extract_number)
    teste_files = sorted([f for f in files if f.startswith('teste_') and f.endswith('.csv')], key=extract_number)

    # Garantir que o número de arquivos de treino e teste é o mesmo
    if len(treino_files) != len(teste_files):
        print("Número de arquivos de treino e teste não coincide.")
        return []

    medias_vitorias_casa = []

    # Loop pelos arquivos de treino e teste
    for treino_file, teste_file in zip(treino_files, teste_files):
        treino_path = os.path.join(data_dir, treino_file)
        teste_path = os.path.join(data_dir, teste_file)

        # Ler os dados
        treino_df = pd.read_csv(treino_path)
        teste_df = pd.read_csv(teste_path)

        # Calcular média de vitórias dos times da casa
        media_treino = (treino_df['placar_casa'] > treino_df['placar_visitante']).mean()
        media_teste = (teste_df['placar_casa'] > teste_df['placar_visitante']).mean()

        # Guardar as médias
        medias_vitorias_casa.append({
            'Treino': f"{temporada}_{treino_file}_{numero_jogos_treino}-{numero_jogos_teste}",
            'Média': round(media_treino, 2),
        })

    return medias_vitorias_casa

if __name__ == '__main__':
    # Configurar o diretório base
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    temporadas = ['2008-2009', '2009-2010', '2011-2012',
                  '2012-2013', '2013-2014', '2014-2015',
                  '2015-2016', '2016-2017', '2018-2019', '2019-2020',
                  '2020-2021', '2021-2022', '2022-2023',
                  '2023-2024'
                 ]
    numeros_jogos_treino = [8,16,32,64,128]
    numeros_jogos_teste = [1]
    
    # Caminho para salvar o CSV
    output_dir = os.path.join(base_path, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    resultados_gerais = []

    for numero_jogos_treino in numeros_jogos_treino:
        for numero_jogos_teste in numeros_jogos_teste:
            for temporada in temporadas:
                data_dir = os.path.join(base_path, "data", "experimento_02", temporada, f'{numero_jogos_treino}'+'-'+f'{numero_jogos_teste}')
                print(f'Calculando médias para temporada {temporada}, janela {numero_jogos_treino}-{numero_jogos_teste}')

                medias = calcular_media_vitorias_casa(data_dir, temporada, numero_jogos_treino, numero_jogos_teste)
                resultados_gerais.extend(medias)

            output_path = os.path.join(output_dir, "medias_vitorias_treino", f'medias_vitorias_todos_{numero_jogos_treino}-{numero_jogos_teste}.csv')
            # Salvar resultados em um arquivo CSV
            if resultados_gerais:
                df_resultados = pd.DataFrame(resultados_gerais)
                df_resultados.to_csv(output_path, index=False, float_format='%.2f')
                print(f'Resultados salvos em: {output_path}')
            else:
                print("Nenhum resultado encontrado.")
            
            resultados_gerais = []
            
