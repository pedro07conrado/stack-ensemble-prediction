"""
Plota a curva de aprendizado incremental do experimento 03.

Lê o CSV detalhado gerado por experimento_03_svm_curva.py e plota:
  1. Acurácia acumulada ao longo do índice de jogo (por temporada ou global)
  2. Acurácia em janela deslizante (média móvel) para suavizar ruído

Uso:
    python plotar_curva_aprendizado.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ============================================================
# Configuração
# ============================================================
BASE_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR  = os.path.join(BASE_PATH, 'results', 'experimento_03_curva')
PLOTS_DIR    = os.path.join(BASE_PATH, 'results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELO       = 'svm'
JANELAS      = [5, 10, 15]
JANELA_MEDIA = 15   # janela para a média móvel (suavização)

plt.style.use('seaborn-v0_8-darkgrid')
CORES = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ============================================================
# Função: calcula acurácia acumulada jogo a jogo
# ============================================================
def acuracia_acumulada(df_sorted):
    acertos_cum = df_sorted['acertou'].cumsum()
    indices     = np.arange(1, len(df_sorted) + 1)
    return indices, acertos_cum / indices


# ============================================================
# Função: média móvel
# ============================================================
def media_movel(serie, janela):
    return serie.rolling(window=janela, min_periods=1).mean()


# ============================================================
# Plota curva por temporada (uma linha por temporada)
# ============================================================
def plotar_por_temporada(jogos_base):
    path = os.path.join(
        RESULTS_DIR,
        f'{MODELO}_experimento_03_janela{jogos_base}_detalhe.csv'
    )
    if not os.path.exists(path):
        print(f'[SKIP] Arquivo não encontrado: {path}')
        return

    df         = pd.read_csv(path)
    temporadas = df['Temporada'].unique()

    fig, axes = plt.subplots(
        nrows=len(temporadas), ncols=1,
        figsize=(14, 3.5 * len(temporadas)),
        sharex=False
    )
    if len(temporadas) == 1:
        axes = [axes]

    fig.suptitle(
        f'Curva de Aprendizado Incremental — {MODELO.upper()} | K={jogos_base}',
        fontsize=16, fontweight='bold', y=1.01
    )

    for ax, temporada in zip(axes, temporadas):
        df_t = df[df['Temporada'] == temporada].reset_index(drop=True)

        indices, acc_cum = acuracia_acumulada(df_t)
        acc_mm           = media_movel(df_t['acertou'], JANELA_MEDIA)

        ax.plot(indices, acc_cum,
                label='Acurácia Acumulada',
                color='steelblue', linewidth=1.5, alpha=0.8)
        ax.plot(indices, acc_mm,
                label=f'Média Móvel ({JANELA_MEDIA} jogos)',
                color='tomato', linewidth=2, linestyle='--')

        # Linha de referência (50%)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='50% (chute aleatório)')

        n_total   = len(df_t)
        acc_final = acc_cum.iloc[-1]
        ax.set_title(
            f'{temporada}  |  {n_total} predições  |  Acurácia final: {acc_final:.2%}',
            fontsize=11
        )
        ax.set_ylabel('Acurácia', fontsize=10)
        ax.set_xlabel('Índice do Jogo (iteração da janela)', fontsize=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'{MODELO}_curva_por_temporada_K{jogos_base}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Salvo: {out}')
    plt.close()


# ============================================================
# Plota curva global (junta todas as temporadas numa linha)
# ============================================================
def plotar_global(jogos_base):
    path = os.path.join(
        RESULTS_DIR,
        f'{MODELO}_experimento_03_janela{jogos_base}_detalhe.csv'
    )
    if not os.path.exists(path):
        print(f'[SKIP] Arquivo não encontrado: {path}')
        return

    df = pd.read_csv(path)
    # Reindexa globalmente (como se fosse uma sequência contínua)
    df = df.reset_index(drop=True)
    df['jogo_global'] = df.index + 1

    indices, acc_cum = acuracia_acumulada(df)
    acc_mm           = media_movel(df['acertou'], JANELA_MEDIA)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(indices, acc_cum,
            label='Acurácia Acumulada (todas as temporadas)',
            color='steelblue', linewidth=1.5, alpha=0.7)
    ax.plot(indices, acc_mm,
            label=f'Média Móvel ({JANELA_MEDIA} jogos)',
            color='tomato', linewidth=2.5, linestyle='--')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='50% (chute aleatório)')

    # Marca a separação entre temporadas
    temporadas = df['Temporada'].unique()
    offset = 0
    for i, t in enumerate(temporadas):
        n = (df['Temporada'] == t).sum()
        offset += n
        if i < len(temporadas) - 1:
            ax.axvline(offset, color='black', alpha=0.2, linewidth=0.8)
            ax.text(offset, 0.33, t, rotation=90, fontsize=7, color='gray',
                    ha='right', va='bottom')

    acc_final = acc_cum.iloc[-1]
    ax.set_title(
        f'Curva de Aprendizado Global — {MODELO.upper()} | K={jogos_base} | '
        f'{len(df)} predições totais | Acurácia final: {acc_final:.2%}',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Índice Global do Jogo', fontsize=11)
    ax.set_ylabel('Acurácia', fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'{MODELO}_curva_global_K{jogos_base}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Salvo: {out}')
    plt.close()


# ============================================================
# Plota comparação entre janelas (K=5, 10, 15) num único gráfico
# ============================================================
def plotar_comparacao_janelas():
    fig, ax = plt.subplots(figsize=(16, 6))

    for i, jogos_base in enumerate(JANELAS):
        path = os.path.join(
            RESULTS_DIR,
            f'{MODELO}_experimento_03_janela{jogos_base}_detalhe.csv'
        )
        if not os.path.exists(path):
            continue

        df               = pd.read_csv(path).reset_index(drop=True)
        acc_mm           = media_movel(df['acertou'], JANELA_MEDIA)
        indices          = np.arange(1, len(df) + 1)

        ax.plot(indices, acc_mm,
                label=f'K={jogos_base} (média móvel {JANELA_MEDIA}j)',
                color=CORES[i], linewidth=2)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='50% (chute aleatório)')
    ax.set_title(
        f'Comparação de Janelas — {MODELO.upper()} | Média Móvel {JANELA_MEDIA} jogos',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Índice Global do Jogo', fontsize=11)
    ax.set_ylabel('Acurácia (Média Móvel)', fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'{MODELO}_comparacao_janelas.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Salvo: {out}')
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    for jogos_base in JANELAS:
        print(f'\n--- Janela K={jogos_base} ---')
        plotar_por_temporada(jogos_base)
        plotar_global(jogos_base)

    print('\n--- Comparação entre janelas ---')
    plotar_comparacao_janelas()

    print('\nTodos os gráficos salvos em:', PLOTS_DIR)
