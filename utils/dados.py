import os
from dotenv import load_dotenv
import mysql.connector
import json
import time

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração de conexão com o banco de dados usando variáveis de ambiente
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE')
}

def get_all_jogos():
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos ORDER BY data"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe(equipe):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_casa = %s OR equipe_visitante = %s ORDER BY data"
        cursor.execute(query, (equipe, equipe))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe_casa(equipe):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_casa = %s ORDER BY data"
        cursor.execute(query, (equipe,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def get_jogos_equipe_visitante(equipe):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE equipe_visitante = %s ORDER BY data"
        cursor.execute(query, (equipe,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media=8):
    # Limita os jogos aos últimos num_jogos_passados_media jogos, mas só se houver mais jogos que o limite
    if len(jogos) > num_jogos_passados_media:
        jogos = jogos[-num_jogos_passados_media:]
    
    total_estatisticas = None
    for jogo in jogos:
        # Verifica se a equipe é a equipe da casa ou visitante
        if jogo['equipe_casa'] == equipe:
            estatisticas = json.loads(jogo['estatisticas_casa'])
        else:
            estatisticas = json.loads(jogo['estatisticas_visitantes'])
            
        # Inicializa o total_estatisticas se for a primeira iteração
        if total_estatisticas is None:
            total_estatisticas = estatisticas
        else:
            # Acumula os valores das estatísticas
            for key, value in estatisticas.items():
                total_estatisticas[key] += value
    
    # Calcula a média
    num_jogos = len(jogos)
    if num_jogos > 0:
        for key in total_estatisticas:
            total_estatisticas[key] /= num_jogos
    
    return total_estatisticas

def get_media_estatisticas_time_teste(equipe, data_jogo, num_jogos_passados_media=8):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        # Seleciona todos os jogos anteriores ao jogo especificado
        query = """
            SELECT equipe_casa, equipe_visitante, estatisticas_casa, estatisticas_visitantes FROM jogos
            WHERE (equipe_casa = %s OR equipe_visitante = %s) AND data < %s
            ORDER BY data
        """
        cursor.execute(query, (equipe, equipe, data_jogo))
        jogos = cursor.fetchall()
        
        if jogos:
            return calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media)
        else:
            return {}
    finally:
        cursor.close()
        conn.close()

def get_media_estatisticas_time_treino(equipe, data_jogo, num_jogos_passados_media=8):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)
    try:
        cursor = conn.cursor(dictionary=True)
        # Seleciona todos os jogos anteriores e o jogo especificado
        query = """
            SELECT equipe_casa, equipe_visitante, estatisticas_casa, estatisticas_visitantes FROM jogos
            WHERE (equipe_casa = %s OR equipe_visitante = %s) AND data <= %s
            ORDER BY data
        """
        cursor.execute(query, (equipe, equipe, data_jogo))
        jogos = cursor.fetchall()
        
        if jogos:
            return calcular_media_estatisticas(jogos, equipe, num_jogos_passados_media)
        else:
            return {}
    finally:
        cursor.close()
        conn.close()

def get_jogos_temporada(ano):
    time.sleep(0.005)
    conn = mysql.connector.connect(**db_config)    
    try:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM jogos WHERE ano = %s ORDER BY data"
        cursor.execute(query, (ano,))
        results = cursor.fetchall()
        return results
    finally:
        cursor.close()
        conn.close()

def formatar_medias(jogos, isTreino, num_jogos_passados_media=10):
    jogos_nova = jogos
    for jogo in jogos_nova:
        equipe_casa = jogo['equipe_casa']
        equipe_visitante = jogo['equipe_visitante']
        data_jogo = jogo['data']
        
        if isTreino:
            media_casa = get_media_estatisticas_time_treino(equipe_casa, data_jogo, num_jogos_passados_media)
            media_visitante = get_media_estatisticas_time_treino(equipe_visitante, data_jogo, num_jogos_passados_media)
        else:
            media_casa = get_media_estatisticas_time_teste(equipe_casa, data_jogo, num_jogos_passados_media)
            media_visitante = get_media_estatisticas_time_teste(equipe_visitante, data_jogo, num_jogos_passados_media)
        
        # Substitui as estatísticas brutas pelas médias calculadas
        jogo['estatisticas_casa'] = media_casa
        jogo['estatisticas_visitantes'] = media_visitante
        
    return jogos_nova

# Exemplo de uso
if __name__ == "__main__":
    temporada = '2008-2009'
    #jogos = get_jogos_temporada(temporada)
    #print(jogos)
    # jogos = get_jogos_equipe("Minas")
 
    # for jogo in jogos:
    #     print("\n", jogo)
    