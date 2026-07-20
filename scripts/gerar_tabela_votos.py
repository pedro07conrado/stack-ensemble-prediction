"""
Gera a Tabela de Decisão por Jogo (Janela 15)
==============================================
Para a janela de média móvel = 15, monta uma tabela (uma linha por jogo)
com a previsão e a confiança de cada um dos 5 modelos, o voto final
(majoritário) e o resultado real (target). Uma tabela por temporada.

Colunas de saída:
  Rodada, Data,
  SVM_prob(%), SVM_pred,
  KNN_prob(%), KNN_pred,
  ExtraTrees_prob(%), ExtraTrees_pred,
  LightGBM_prob(%), LightGBM_pred,
  Regressao_prob(%), Regressao_pred,
  Voto_final, Target
"""

import pandas as pd
import os

# ─────────────────────────────────────────────
# 1. CONFIGURAÇÃO
# ─────────────────────────────────────────────

PASTA_ENTRADA = r'C:\Users\pedro.negreiro\Documents\stack-ensemble-prediction\results\experimento_03_detalhado'
PASTA_SAIDA   = r'C:\Users\pedro.negreiro\Documents\stack-ensemble-prediction\results\tabela_votos_por_jogo\janela_15'
ARQUIVO_EXCEL = os.path.join(PASTA_SAIDA, 'tabela_votos_janela_15.xlsx')

JANELA = 15

# nome_exibicao -> arquivo de predições detalhadas
MODELOS = {
    "SVM":         "svm_experimento_03_predicoes_detalhadas.csv",
    "KNN":         "knn_experimento_03_predicoes_detalhadas.csv",
    "ExtraTrees":  "extra_trees_experimento_03_predicoes_detalhadas.csv",
    "LightGBM":    "lightgbm_experimento_03_predicoes_detalhadas.csv",
    "Regressao":   "regressao_logistica_experimento_03_predicoes_detalhadas.csv",
}

# Chave que identifica um jogo único dentro de uma mesma temporada/janela.
# (Não usar Data Jogo + Round + Posicao no Teste: como cada arquivo de
#  teste da janela flutuante tem 1 jogo só, "Posicao no Teste" é sempre 1,
#  e jogos diferentes no mesmo round/data colidiriam nessa chave.)
CHAVE_JOGO = ["Temporada", "Ordem Jogo Temporada"]

COLUNAS_BASE = ["Temporada", "Round", "Data Jogo", "Resultado Real", "Ordem Jogo Temporada"]

# ─────────────────────────────────────────────
# 2. LEITURA DE CADA MODELO (filtrado na janela escolhida)
# ─────────────────────────────────────────────

tabela = None

for nome, arquivo in MODELOS.items():
    caminho = os.path.join(PASTA_ENTRADA, arquivo)
    df = pd.read_csv(caminho)
    df = df[df["Janela Incremental"] == JANELA].copy()

    # confiança = probabilidade da classe que o modelo efetivamente previu
    prob_classe1 = df["Probabilidade Classe 1"]
    previsao = df["Previsao"]
    confianca_pct = (prob_classe1.where(previsao == 1, 1 - prob_classe1) * 100).round(1)

    df_modelo = df[COLUNAS_BASE].copy()
    df_modelo[f"{nome}_prob(%)"] = confianca_pct
    df_modelo[f"{nome}_pred"] = previsao

    if tabela is None:
        tabela = df_modelo
    else:
        # junta apenas as colunas novas do modelo; base (Round/Data/Resultado Real)
        # já vem do primeiro modelo
        novas_colunas = CHAVE_JOGO + [f"{nome}_prob(%)", f"{nome}_pred"]
        tabela = tabela.merge(
            df_modelo[novas_colunas],
            on=CHAVE_JOGO,
            how="inner",
            validate="one_to_one",
        )

print(f"Total de jogos combinados (janela {JANELA}, todos os modelos): {len(tabela)}")

# ─────────────────────────────────────────────
# 3. VOTO FINAL (majoritário; 5 modelos = nunca empata)
# ─────────────────────────────────────────────

colunas_pred = [f"{nome}_pred" for nome in MODELOS]
tabela["Voto_final"] = (tabela[colunas_pred].sum(axis=1) >= 3).astype(int)
tabela["Target"] = tabela["Resultado Real"]

# ─────────────────────────────────────────────
# 4. ORGANIZAÇÃO DE COLUNAS E ORDENAÇÃO
# ─────────────────────────────────────────────

tabela = tabela.rename(columns={"Round": "Rodada", "Data Jogo": "Data"})
tabela = tabela.sort_values(["Temporada", "Ordem Jogo Temporada"])

colunas_modelos = []
for nome in MODELOS:
    colunas_modelos += [f"{nome}_prob(%)", f"{nome}_pred"]

colunas_finais = ["Rodada", "Data"] + colunas_modelos + ["Voto_final", "Target"]

# ─────────────────────────────────────────────
# 5. SALVA UM ÚNICO .XLSX, UMA ABA POR TEMPORADA
# ─────────────────────────────────────────────
# (usar .xlsx em vez de .csv evita o problema do Excel em locale pt-BR
#  não separar as colunas por vírgula ao abrir o arquivo)

os.makedirs(PASTA_SAIDA, exist_ok=True)

with pd.ExcelWriter(ARQUIVO_EXCEL, engine="openpyxl") as writer:
    for temporada, grupo in tabela.groupby("Temporada"):
        saida = grupo[colunas_finais].reset_index(drop=True)
        saida.index += 1
        saida.to_excel(writer, sheet_name=temporada, index_label="Jogo")

        acertos = (saida["Voto_final"] == saida["Target"]).mean()
        print(f"  {temporada}: {len(saida)} jogos | acurácia voto majoritário: {acertos:.4f}")

print(f"\nArquivo salvo em: {ARQUIVO_EXCEL}")
