"""
Módulo de Análise Exploratória de Dados Pokémon
Gera visualizações e estatísticas descritivas dos dados processados.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pointbiserialr
from pathlib import Path

from src.utils.config import PROCESSED_DATA_PATH


# Paleta de cores profissional para fundo branco
COLORS = {
    'primary': '#2E86AB',      # Azul profissional
    'secondary': '#A23B72',    # Roxo/rosa
    'success': '#06A77D',      # Verde
    'warning': '#F18F01',      # Laranja
    'danger': '#C73E1D',       # Vermelho
    'info': '#6C757D',         # Cinza
    'accent': '#DDA15E',       # Bege/dourado
    'palette': ['#2E86AB', '#A23B72', '#06A77D', '#F18F01', '#C73E1D', '#6C757D', '#DDA15E', '#BC6C25']
}


# =============================================================================
# FUNÇÕES DE ANÁLISE ESTATÍSTICA
# =============================================================================

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula correlações entre features e a variável alvo."""
    
    features = ["stats_diff", "speed_diff", "attack_diff", "defense_diff"]
    
    # Cria a variável alvo se não existir
    if "winner_1" not in df.columns:
        df["winner_1"] = (df["winner_name"] == df["pokemon_1"]).astype(int)
    
    corr_series = df[features + ["winner_1"]].corr()["winner_1"]
    
    # Remove a correlação da variável consigo mesma e ordena
    corr_df = corr_series[corr_series.index != "winner_1"].sort_values(ascending=False)
    
    return pd.DataFrame({
        "feature": corr_df.index.tolist(),
        "correlation": corr_df.values.tolist()
    })


def analyze_type_advantage(df_combats: pd.DataFrame) -> dict:
    """Analisa o impacto da vantagem de tipo nos combates."""
    
    df = df_combats.copy()
    
    # Cria a variável alvo se não existir
    if "winner_1" not in df.columns:
        df["winner_1"] = (df["winner_name"] == df["pokemon_1"]).astype(int)
    
    # Mapeia vantagem do ponto de vista do pokemon_1
    df['vantagem_p1'] = df['type_advantage'].map({1: 1, 0: 0, 2: -1})
    
    # Classifica resultado da vantagem
    def classificar_vantagem(row):
        if row['type_advantage'] == 0:
            return 'Neutro'
        elif (row['type_advantage'] == 1 and row['winner_1'] == 1) or \
             (row['type_advantage'] == 2 and row['winner_1'] == 0):
            return 'Vencedor com vantagem'
        else:
            return 'Perdedor com vantagem'
    
    df['resultado_vantagem'] = df.apply(classificar_vantagem, axis=1)
    
    # Calcula estatísticas
    contagem = df['resultado_vantagem'].value_counts()
    
    # Correlação ponto-bisserial
    corr_vantagem, p_value = pointbiserialr(df['vantagem_p1'], df['winner_1'])
    
    # Taxas de vitória por cenário
    vantagem_p1 = df[df['vantagem_p1'] == 1]
    vantagem_p2 = df[df['vantagem_p1'] == -1]
    neutro = df[df['vantagem_p1'] == 0]
    
    # Vencedor tinha vantagem
    df['vencedor_tinha_vantagem'] = (
        ((df['type_advantage'] == 1) & (df['winner_1'] == 1)) |
        ((df['type_advantage'] == 2) & (df['winner_1'] == 0))
    ).astype(int)
    
    combates_com_vantagem = df[df['type_advantage'] != 0]
    perc_vantagem_vencedor = combates_com_vantagem['vencedor_tinha_vantagem'].mean()
    
    return {
        'df_analise': df,
        'contagem': contagem,
        'correlacao': corr_vantagem,
        'p_value': p_value,
        'taxa_vitoria_vantagem_p1': vantagem_p1['winner_1'].mean(),
        'taxa_vitoria_vantagem_p2': 1 - vantagem_p2['winner_1'].mean(),
        'taxa_vitoria_neutro': neutro['winner_1'].mean(),
        'perc_vantagem_vencedor': perc_vantagem_vencedor
    }


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def plot_feature_correlations(corr_df: pd.DataFrame) -> go.Figure:
    """Plota correlações entre features e variável alvo."""
    
    fig = px.bar(
        corr_df,
        x="feature",
        y="correlation",
        text="correlation",
        title="<b>Correlação das Features com a Vitória</b>",
        labels={"feature": "Feature", "correlation": "Correlação"},
        color="correlation",
        color_continuous_scale=[[0, COLORS['danger']], [0.5, '#E8E8E8'], [1, COLORS['primary']]],
        range_color=[-1, 1]
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#E8E8E8')
    )
    
    return fig


def plot_speed_advantage(df: pd.DataFrame) -> go.Figure:
    """Plota taxa de vitória por vantagem de velocidade."""
    
    df_copy = df.copy()
    
    # Cria a variável alvo se não existir
    if "winner_1" not in df_copy.columns:
        df_copy["winner_1"] = (df_copy["winner_name"] == df_copy["pokemon_1"]).astype(int)
    
    df_copy["faster"] = (df_copy["speed_diff"] > 0).astype(int)
    
    speed_group = df_copy.groupby("faster")["winner_1"].mean().reset_index()
    speed_group["faster"] = speed_group["faster"].map({1: "Mais rápido", 0: "Mais lento"})
    speed_group = speed_group.sort_values("winner_1", ascending=False)
    
    fig = px.bar(
        speed_group,
        y="faster",
        x="winner_1",
        text="winner_1",
        title="<b>Taxa de Vitória por Vantagem de Velocidade</b>",
        labels={"winner_1": "Probabilidade de Vitória", "faster": ""},
        color_discrete_sequence=[COLORS['primary']],
        orientation='h',
        category_orders={"faster": speed_group["faster"].tolist()}
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        xaxis=dict(showgrid=True, gridcolor='#E8E8E8', tickformat='.0%'),
        yaxis=dict(showgrid=False)
    )
    
    return fig


def plot_pokemon_stats_distribution(df_pokemon: pd.DataFrame) -> go.Figure:
    """Plota distribuição das estatísticas base dos Pokémon."""
    
    variaveis = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
    colors_dist = [COLORS['primary'], COLORS['success'], COLORS['warning'], 
                   COLORS['secondary'], COLORS['accent'], COLORS['info']]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"<b>{var.upper().replace('_', ' ')}</b>" for var in variaveis],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    for i, var in enumerate(variaveis):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(
                x=df_pokemon[var], 
                nbinsx=30, 
                name=var.upper(), 
                marker_color=colors_dist[i],
                marker_line_color='white',
                marker_line_width=0.5
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text=var.upper().replace('_', ' '), row=row, col=col, showgrid=False)
        fig.update_yaxes(title_text="Frequência", row=row, col=col, showgrid=True, gridcolor='#E8E8E8')
    
    fig.update_layout(
        title_text="<b>Distribuição das Estatísticas Base dos Pokémon</b>",
        title_font_size=16,
        showlegend=False,
        height=600,
        width=1200,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    return fig


def plot_type_advantage_distribution(df_analise: pd.DataFrame) -> go.Figure:
    """Plota distribuição dos cenários de vantagem de tipo."""
    
    counts = df_analise['resultado_vantagem'].value_counts()
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        labels={'x': 'Cenário', 'y': 'Quantidade'},
        title='<b>Distribuição dos Combates por Vantagem de Tipo</b>',
        color_discrete_map={
            'Neutro': COLORS['info'],
            'Vencedor com vantagem': COLORS['success'],
            'Perdedor com vantagem': COLORS['danger']
        }
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#E8E8E8')
    )
    fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    
    return fig


def plot_advantage_winner_distribution(perc_vantagem: float) -> go.Figure:
    """Plota percentual de vitórias quando há vantagem de tipo."""
    
    labels = ['Vencedor com vantagem', 'Perdedor com vantagem']
    values = [perc_vantagem, 1 - perc_vantagem]
    colors_pie = [COLORS['success'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.4,
        marker_colors=colors_pie,
        textfont_size=14
    )])
    
    fig.update_layout(
        title=f'<b>Vitórias em Combates com Vantagem de Tipo</b><br><sup>O vencedor tinha vantagem em {perc_vantagem:.1%} dos casos</sup>',
        title_font_size=16,
        font=dict(family="Arial, sans-serif", size=12),
        annotations=[dict(
            text=f'<b>{perc_vantagem:.1%}</b>', 
            x=0.5, y=0.5, 
            font_size=24, 
            showarrow=False,
            font_color=COLORS['success']
        )]
    )
    
    return fig


# =============================================================================
# PIPELINE DE ANÁLISE
# =============================================================================

def generate_all_visualizations(
    df_pokemon: pd.DataFrame,
    df_combats: pd.DataFrame,
    show_plots: bool = False  # Agora padrão é False para evitar abertura automática
) -> dict:
    """
    Gera todas as visualizações e análises.
    
    Args:
        df_pokemon: DataFrame com dados dos Pokémon
        df_combats: DataFrame com dados dos combates
        show_plots: Se True, exibe os gráficos (padrão False)
    
    Returns:
        Dicionário com todas as figuras e análises
    """
    
    # Cálculos (sem prints de andamento)
    corr_df = calculate_correlations(df_combats)
    type_analysis = analyze_type_advantage(df_combats)
    
    # Geração das figuras
    fig_corr = plot_feature_correlations(corr_df)
    fig_speed = plot_speed_advantage(df_combats)
    fig_stats_dist = plot_pokemon_stats_distribution(df_pokemon)
    fig_type_dist = plot_type_advantage_distribution(type_analysis['df_analise'])
    fig_advantage_winner = plot_advantage_winner_distribution(type_analysis['perc_vantagem_vencedor'])
    
    if show_plots:
        fig_corr.show()
        fig_speed.show()
        fig_stats_dist.show()
        fig_type_dist.show()
        fig_advantage_winner.show()
    
    # Exibe resumo estatístico (considerado essencial)
    print("\n" + "="*80)
    print("RESUMO DA ANÁLISE DE VANTAGEM DE TIPO")
    print("="*80)
    print(f"\nCorrelação vantagem-vitória: {type_analysis['correlacao']:.3f} (p-value: {type_analysis['p_value']:.3e})")
    print(f"Taxa de vitória com vantagem (P1): {type_analysis['taxa_vitoria_vantagem_p1']:.2%}")
    print(f"Taxa de vitória com vantagem (P2): {type_analysis['taxa_vitoria_vantagem_p2']:.2%}")
    print(f"Taxa de vitória em combates neutros: {type_analysis['taxa_vitoria_neutro']:.2%}")
    print(f"\nPercentual de vitórias com vantagem: {type_analysis['perc_vantagem_vencedor']:.2%}")
    print("="*80 + "\n")
    
    return {
        'correlations': corr_df,
        'type_analysis': type_analysis,
        'figures': {
            'correlations': fig_corr,
            'speed_advantage': fig_speed,
            'stats_distribution': fig_stats_dist,
            'type_distribution': fig_type_dist,
            'advantage_winner': fig_advantage_winner
        }
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run():
    """
    Executa a análise exploratória completa dos dados processados.
    """
    # Carrega dados processados
    pokemon_path = PROCESSED_DATA_PATH / "pokemon_processed.parquet"
    combats_path = PROCESSED_DATA_PATH / "combats_processed.parquet"
    
    if not pokemon_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pokemon_path}")
    if not combats_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {combats_path}")
    
    df_pokemon = pd.read_parquet(pokemon_path)
    df_combats = pd.read_parquet(combats_path)
    
    # Gera análises e visualizações (sem abrir figuras)
    results = generate_all_visualizations(df_pokemon, df_combats, show_plots=False)
    
    # Mensagem final simples
    print("Análise exploratória concluída!")
    
    return results


if __name__ == "__main__":
    run()