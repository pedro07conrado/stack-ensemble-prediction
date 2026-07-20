"""
Gera a Tabela de Erros Brutos (Janela 15)
==========================================
A partir da tabela de votos por jogo (tabela_votos_janela_15.xlsx), filtra
apenas os jogos em que PELO MENOS UM modelo "errou feio": previu uma classe
com confiança >= LIMITE e o resultado real foi a classe contrária.

Ex: modelo disse 80% de confiança para 0, resultado real foi 1 -> erro bruto.

Adiciona duas colunas:
  Erros_Brutos     -> nomes dos modelos que erraram feio nesse jogo
  Qtd_Erros_Brutos -> quantos modelos erraram feio nesse jogo
"""

import pandas as pd
import os

# ─────────────────────────────────────────────
# 1. CONFIGURAÇÃO
# ─────────────────────────────────────────────

PASTA = r'C:\Users\pedro.negreiro\Documents\stack-ensemble-prediction\results\tabela_votos_por_jogo\janela_15'
ARQUIVO_ENTRADA = os.path.join(PASTA, 'tabela_votos_janela_15.xlsx')
ARQUIVO_SAIDA   = os.path.join(PASTA, 'tabela_erros_brutos_janela_15.xlsx')

LIMITE_CONFIANCA = 70  # em % — confiança mínima para considerar "errou feio"

MODELOS = ["SVM", "KNN", "ExtraTrees", "LightGBM", "Regressao"]

# ─────────────────────────────────────────────
# 2. PROCESSA CADA TEMPORADA (ABA) E FILTRA
# ─────────────────────────────────────────────

xls = pd.ExcelFile(ARQUIVO_ENTRADA)

contagem_por_modelo = {m: 0 for m in MODELOS}
total_jogos = 0
total_filtrados = 0

with pd.ExcelWriter(ARQUIVO_SAIDA, engine="openpyxl") as writer:
    for temporada in xls.sheet_names:
        df = xls.parse(temporada)

        erro_bruto = pd.DataFrame(index=df.index)
        for m in MODELOS:
            erro_bruto[m] = (df[f"{m}_pred"] != df["Target"]) & (df[f"{m}_prob(%)"] >= LIMITE_CONFIANCA)
            contagem_por_modelo[m] += erro_bruto[m].sum()

        df["Qtd_Erros_Brutos"] = erro_bruto.sum(axis=1)
        df["Erros_Brutos"] = erro_bruto.apply(lambda linha: ", ".join(m for m in MODELOS if linha[m]), axis=1)

        total_jogos += len(df)
        filtrado = df[df["Qtd_Erros_Brutos"] >= 1].reset_index(drop=True)
        total_filtrados += len(filtrado)

        # a coluna "Jogo" (posição original na temporada) já identifica a linha;
        # não escreve o índice do pandas para não duplicar essa informação
        filtrado.to_excel(writer, sheet_name=temporada, index=False)

        print(f"  {temporada}: {len(df)} jogos -> {len(filtrado)} com erro bruto (>= {LIMITE_CONFIANCA}%)")

print(f"\nTotal geral: {total_filtrados} / {total_jogos} jogos com pelo menos 1 erro bruto")
print("\nErros brutos por modelo (todas as temporadas):")
for m, qtd in sorted(contagem_por_modelo.items(), key=lambda x: -x[1]):
    print(f"  {m:12s}: {qtd}")

print(f"\nArquivo salvo em: {ARQUIVO_SAIDA}")
