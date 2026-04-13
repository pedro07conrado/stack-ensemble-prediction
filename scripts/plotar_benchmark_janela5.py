"""
Benchmark visual — Experimento 03 | Janela K=5

Compara a acurácia de todos os modelos em 6 temporadas selecionadas.
Cada temporada é um painel independente com um gráfico de barras.

Uso:
    cd scripts
    python plotar_benchmark_janela5.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────
BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC    = os.path.join(BASE, 'results', 'experimento_03')
OUT    = os.path.join(BASE, 'results', 'plots')
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# Modelos disponíveis para janela 5
# ─────────────────────────────────────────────
MODELOS = {
    'SVM':                'svm_experimento_03_janela5.csv',
    'KNN':                'knn_experimento_03_janela5.csv',
    'LightGBM':           'lightgbm_experimento_03_janela5.csv',
    'Extra Trees':        'extra_trees_experimento_03_janela5.csv',
    'Reg. Logística':     'regressao_logistica_experimento_03_janela5.csv',
}

# ─────────────────────────────────────────────
# 6 temporadas para exibir (as mais recentes)
# ─────────────────────────────────────────────
TEMPORADAS = [
    '2018-2019',
    '2019-2020',
    '2020-2021',
    '2021-2022',
    '2022-2023',
    '2023-2024',
]

# ─────────────────────────────────────────────
# Cores — uma por modelo
# ─────────────────────────────────────────────
CORES = {
    'SVM':            '#4C72B0',
    'KNN':            '#DD8452',
    'LightGBM':       '#55A868',
    'Extra Trees':    '#C44E52',
    'Reg. Logística': '#8172B2',
}

# ─────────────────────────────────────────────
# Carrega dados
# ─────────────────────────────────────────────
def carregar_dados():
    registros = {}
    for nome, arquivo in MODELOS.items():
        caminho = os.path.join(SRC, arquivo)
        if not os.path.exists(caminho):
            print(f'[AVISO] Arquivo não encontrado: {arquivo}')
            continue
        df = pd.read_csv(caminho)
        # Normaliza nome da coluna (pode ter acento)
        col_acc = [c for c in df.columns if 'cur' in c.lower() or 'acc' in c.lower() or 'Acu' in c][0]
        for _, row in df.iterrows():
            temporada = str(row['Temporada'])
            if temporada not in registros:
                registros[temporada] = {}
            registros[temporada][nome] = float(row[col_acc])
    return registros

# ─────────────────────────────────────────────
# Plot principal
# ─────────────────────────────────────────────
def plotar():
    dados = carregar_dados()

    nomes_modelos = list(MODELOS.keys())
    x = np.arange(len(nomes_modelos))
    largura = 0.55

    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(18, 10),
        sharey=True
    )
    axes = axes.flatten()

    fig.patch.set_facecolor('#F7F7F7')
    fig.suptitle(
        'Benchmark de Modelos — Experimento 03 | Janela Incremental K = 5',
        fontsize=18, fontweight='bold', y=1.02, color='#1a1a2e'
    )

    for i, temporada in enumerate(TEMPORADAS):
        ax = axes[i]
        ax.set_facecolor('#FFFFFF')

        acuracias = [dados.get(temporada, {}).get(m, 0) for m in nomes_modelos]
        melhor    = max(acuracias) if any(a > 0 for a in acuracias) else 0

        barras = ax.bar(
            x, acuracias,
            width=largura,
            color=[CORES[m] for m in nomes_modelos],
            edgecolor='white',
            linewidth=1.2,
            zorder=3
        )

        # Linha de referência 50%
        ax.axhline(0.5, color='#999999', linestyle='--', linewidth=1,
                   zorder=2, label='_nolegend_')
        ax.text(len(nomes_modelos) - 0.5, 0.505, '50%',
                fontsize=8, color='#999999', va='bottom', ha='right')

        # Valor em cima de cada barra
        for barra, acc in zip(barras, acuracias):
            if acc == 0:
                continue
            destaque = (acc == melhor)
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                acc + 0.008,
                f'{acc:.1%}',
                ha='center', va='bottom',
                fontsize=10,
                fontweight='bold' if destaque else 'normal',
                color='#222222' if not destaque else CORES[nomes_modelos[acuracias.index(melhor)]]
            )

        # Contorno do melhor modelo
        idx_melhor = acuracias.index(melhor)
        barras[idx_melhor].set_edgecolor('#FFD700')
        barras[idx_melhor].set_linewidth(3)

        ax.set_title(f'🗓  {temporada}', fontsize=13, fontweight='bold',
                     color='#1a1a2e', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(nomes_modelos, fontsize=10, rotation=15, ha='right')
        ax.set_ylim(0.4, 0.95)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylabel('Acurácia' if i % 3 == 0 else '', fontsize=10)
        ax.grid(axis='y', alpha=0.4, zorder=1)
        ax.spines[['top', 'right']].set_visible(False)

    # Legenda global (ícone colorido = modelo)
    patches = [
        mpatches.Patch(color=CORES[m], label=m)
        for m in nomes_modelos
    ]
    fig.legend(
        handles=patches,
        title='Modelos',
        title_fontsize=11,
        fontsize=10,
        loc='lower center',
        ncol=len(nomes_modelos),
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        edgecolor='#cccccc'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    saida = os.path.join(OUT, 'benchmark_janela5_6temporadas.png')
    plt.savefig(saida, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\nGráfico salvo em:\n  {saida}')
    plt.show()


if __name__ == '__main__':
    plotar()
