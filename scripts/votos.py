"""
Análise de Combinações de Modelos (Ensemble Voting)
=====================================================
Junta os 5 CSVs de predição, testa todas as combinações possíveis
de modelos e calcula acurácia por 3 estratégias de votação:
  1. Voto Majoritário
  2. Média das Probabilidades (Classe 1)
  3. Voto Ponderado pela Confiança
"""

import pandas as pd
import numpy as np
from itertools import combinations
import os

# ─────────────────────────────────────────────
# 1. CONFIGURAÇÃO — ajuste o caminho se precisar
# ─────────────────────────────────────────────

PASTA = r'C:\Users\pedro.negreiro\Documents\stack-ensemble-prediction\results\experimento_03_detalhado'

ARQUIVOS = {
    "extra_trees":        "extra_trees_experimento_03_predicoes_detalhadas.csv",
    "knn":                "knn_experimento_03_predicoes_detalhadas.csv",
    "lightgbm":           "lightgbm_experimento_03_predicoes_detalhadas.csv",
    "regressao_logistica":"regressao_logistica_experimento_03_predicoes_detalhadas.csv",
    "svm":                "svm_experimento_03_predicoes_detalhadas.csv",
}

# Coluna(s) que identificam um jogo único entre os arquivos
# Aqui usamos Temporada + Data Jogo + Round + Posicao no Teste
CHAVE_JOGO = ["Temporada", "Data Jogo", "Round", "Posicao no Teste"]

# ─────────────────────────────────────────────
# 2. LEITURA E EMPILHAMENTO
# ─────────────────────────────────────────────

dfs = []
for modelo, arquivo in ARQUIVOS.items():
    caminho = os.path.join(PASTA, arquivo)
    df = pd.read_csv(caminho)
    df["_modelo"] = modelo          # garante o nome correto mesmo se a coluna Modelo diferir
    dfs.append(df)

dados = pd.concat(dfs, ignore_index=True)

print(f"Total de linhas carregadas: {len(dados)}")
print(f"Modelos encontrados: {dados['_modelo'].unique()}")

# ─────────────────────────────────────────────
# 3. PIVOT — uma linha por jogo, colunas por modelo
# ─────────────────────────────────────────────

# Verifica duplicatas por jogo+modelo (não deveria ter, mas por garantia)
dup = dados.groupby(CHAVE_JOGO + ["_modelo"]).size()
if (dup > 1).any():
    print("⚠️  Atenção: existem linhas duplicadas por jogo+modelo. Mantendo a primeira.")
    dados = dados.drop_duplicates(subset=CHAVE_JOGO + ["_modelo"], keep="first")

pivot_prev  = dados.pivot(index=CHAVE_JOGO, columns="_modelo", values="Previsao")
pivot_prob  = dados.pivot(index=CHAVE_JOGO, columns="_modelo", values="Probabilidade Classe 1")
pivot_conf  = dados.pivot(index=CHAVE_JOGO, columns="_modelo", values="Confianca da Previsao (0-1)")
pivot_real  = dados.drop_duplicates(subset=CHAVE_JOGO).set_index(CHAVE_JOGO)["Resultado Real"]

# Alinha tudo no mesmo índice
idx_comum = pivot_prev.dropna().index   # jogos com predição de TODOS os modelos
n_jogos_total   = len(pivot_real)
n_jogos_comuns  = len(idx_comum)
print(f"\nJogos totais: {n_jogos_total} | Jogos com todos os modelos: {n_jogos_comuns}")

pivot_prev  = pivot_prev.loc[idx_comum]
pivot_prob  = pivot_prob.loc[idx_comum]
pivot_conf  = pivot_conf.loc[idx_comum]
y_real      = pivot_real.loc[idx_comum]

modelos = list(ARQUIVOS.keys())

# ─────────────────────────────────────────────
# 4. FUNÇÃO DE AVALIAÇÃO DE UMA COMBINAÇÃO
# ─────────────────────────────────────────────

def avaliar_combinacao(lista_modelos):
    """Retorna acurácia pelas 3 estratégias para um subconjunto de modelos."""

    prev = pivot_prev[lista_modelos]
    prob = pivot_prob[lista_modelos]
    conf = pivot_conf[lista_modelos]

    # --- Estratégia 1: Voto Majoritário ---
    soma_votos = prev.sum(axis=1)
    n = len(lista_modelos)
    # empate → classe 1 (mandante vence, padrão do domínio)
    voto_maj = (soma_votos >= n / 2).astype(int)
    acc_maj = (voto_maj == y_real).mean()

    # --- Estratégia 2: Média das Probabilidades ---
    media_prob = prob.mean(axis=1)
    voto_prob = (media_prob >= 0.5).astype(int)
    acc_prob = (voto_prob == y_real).mean()

    # --- Estratégia 3: Voto Ponderado pela Confiança ---
    # para cada modelo: se previu 1 → peso = +confiança; se previu 0 → peso = -confiança
    sinal = prev.copy()
    sinal[sinal == 0] = -1
    score_pond = (sinal.values * conf.values).sum(axis=1)
    voto_pond = (score_pond >= 0).astype(int)
    acc_pond = (voto_pond == y_real).mean()

    return acc_maj, acc_prob, acc_pond


# ─────────────────────────────────────────────
# 5. MODELOS INDIVIDUAIS (baseline de comparação)
# ─────────────────────────────────────────────

print("\n── Acurácia individual ──")
resultados_ind = []
for m in modelos:
    acc = (pivot_prev[m] == y_real).mean()
    print(f"  {m:25s}: {acc:.4f}")
    resultados_ind.append({
        "Combinacao":      m,
        "N_Modelos":       1,
        "Acc_Majoritario": acc,
        "Acc_Media_Prob":  acc,   # com 1 modelo as 3 estratégias são iguais
        "Acc_Ponderado":   acc,
    })

# ─────────────────────────────────────────────
# 6. TODAS AS COMBINAÇÕES (tamanho 2 a 5)
# ─────────────────────────────────────────────

print("\n── Calculando combinações ──")
resultados_comb = []

for tamanho in range(2, len(modelos) + 1):
    for combo in combinations(modelos, tamanho):
        acc_maj, acc_prob, acc_pond = avaliar_combinacao(list(combo))
        resultados_comb.append({
            "Combinacao":      " + ".join(combo),
            "N_Modelos":       tamanho,
            "Acc_Majoritario": round(acc_maj,  4),
            "Acc_Media_Prob":  round(acc_prob, 4),
            "Acc_Ponderado":   round(acc_pond, 4),
        })
        print(f"  {' + '.join(combo)}")
        print(f"    Majoritário: {acc_maj:.4f} | Média Prob: {acc_prob:.4f} | Ponderado: {acc_pond:.4f}")

# ─────────────────────────────────────────────
# 7. CONSOLIDAÇÃO E RANKING
# ─────────────────────────────────────────────

df_resultado = pd.DataFrame(resultados_ind + resultados_comb)

# Melhor acurácia entre as 3 estratégias
df_resultado["Melhor_Acc"] = df_resultado[
    ["Acc_Majoritario", "Acc_Media_Prob", "Acc_Ponderado"]
].max(axis=1)

df_resultado["Melhor_Estrategia"] = df_resultado[
    ["Acc_Majoritario", "Acc_Media_Prob", "Acc_Ponderado"]
].idxmax(axis=1).map({
    "Acc_Majoritario": "Voto Majoritário",
    "Acc_Media_Prob":  "Média Probabilidades",
    "Acc_Ponderado":   "Ponderado Confiança",
})

df_resultado = df_resultado.sort_values("Melhor_Acc", ascending=False).reset_index(drop=True)
df_resultado.index += 1   # ranking começa em 1

# ─────────────────────────────────────────────
# 8. SAÍDA
# ─────────────────────────────────────────────

print("\n\n══════════════ TOP 10 COMBINAÇÕES ══════════════")
print(df_resultado[["Combinacao","N_Modelos","Acc_Majoritario",
                     "Acc_Media_Prob","Acc_Ponderado","Melhor_Acc",
                     "Melhor_Estrategia"]].head(10).to_string())

saida = "resultado_combinacoes_ensemble.csv"
df_resultado.to_csv(saida, index_label="Rank")
print(f"\n✅ Resultado salvo em: {saida}")
print(f"   {len(df_resultado)} combinações avaliadas (individuais + grupos de 2 a 5 modelos)")