"""
Comparação de Modelos de ML para Predição de Batalhas Pokémon
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from src.utils.config import PROCESSED_DATA_PATH
from src.models.logistic_regression import PokemonLogisticRegression
from src.models.catboost_model import PokemonCatBoost


# Paleta de cores
COLORS = {
    'logistic': '#2E86AB',
    'catboost': '#06A77D',
    'background': 'white'
}


def compare_models(
    logistic_metrics: dict,
    catboost_metrics: dict
) -> pd.DataFrame:
    """
    Compara métricas de dois modelos.
    
    Args:
        logistic_metrics: Métricas do modelo de regressão logística
        catboost_metrics: Métricas do modelo CatBoost
    
    Returns:
        DataFrame com comparação das métricas
    """
    
    comparison = pd.DataFrame({
        'Modelo': ['Regressão Logística', 'CatBoost'],
        'Acurácia': [logistic_metrics['accuracy'], catboost_metrics['accuracy']],
        'AUC-ROC': [logistic_metrics['roc_auc'], catboost_metrics['roc_auc']],
        'Precisão': [
            logistic_metrics['classification_report']['1']['precision'],
            catboost_metrics['classification_report']['1']['precision']
        ],
        'Recall': [
            logistic_metrics['classification_report']['1']['recall'],
            catboost_metrics['classification_report']['1']['recall']
        ],
        'F1-Score': [
            logistic_metrics['classification_report']['1']['f1-score'],
            catboost_metrics['classification_report']['1']['f1-score']
        ]
    })
    
    return comparison


def plot_metrics_comparison(comparison: pd.DataFrame) -> go.Figure:
    """Plota comparação de métricas entre modelos."""
    
    metrics = ['Acurácia', 'AUC-ROC', 'Precisão', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    # Regressão Logística
    fig.add_trace(go.Bar(
        name='Regressão Logística',
        x=metrics,
        y=comparison.iloc[0][metrics].values,
        marker_color=COLORS['logistic'],
        text=comparison.iloc[0][metrics].values,
        texttemplate='%{text:.4f}',
        textposition='outside'
    ))
    
    # CatBoost
    fig.add_trace(go.Bar(
        name='CatBoost',
        x=metrics,
        y=comparison.iloc[1][metrics].values,
        marker_color=COLORS['catboost'],
        text=comparison.iloc[1][metrics].values,
        texttemplate='%{text:.4f}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='<b>Comparação de Métricas entre Modelos</b>',
        title_font_size=16,
        xaxis_title='Métrica',
        yaxis_title='Valor',
        barmode='group',
        plot_bgcolor=COLORS['background'],
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor='#E8E8E8', range=[0, 1.05])
    )
    
    return fig


def plot_confusion_matrices(
    logistic_cm: np.ndarray,
    catboost_cm: np.ndarray
) -> go.Figure:
    """Plota matrizes de confusão lado a lado."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['<b>Regressão Logística</b>', '<b>CatBoost</b>'],
        horizontal_spacing=0.15
    )
    
    # Normalizar para percentuais
    logistic_cm_pct = logistic_cm / logistic_cm.sum() * 100
    catboost_cm_pct = catboost_cm / catboost_cm.sum() * 100
    
    labels = ['Pokémon 2 vence', 'Pokémon 1 vence']
    
    # Regressão Logística
    fig.add_trace(
        go.Heatmap(
            z=logistic_cm_pct,
            x=labels,
            y=labels,
            text=logistic_cm,
            texttemplate='%{text}<br>(%{z:.1f}%)',
            colorscale='Blues',
            showscale=False
        ),
        row=1, col=1
    )
    
    # CatBoost
    fig.add_trace(
        go.Heatmap(
            z=catboost_cm_pct,
            x=labels,
            y=labels,
            text=catboost_cm,
            texttemplate='%{text}<br>(%{z:.1f}%)',
            colorscale='Greens',
            showscale=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Predito", row=1, col=1)
    fig.update_xaxes(title_text="Predito", row=1, col=2)
    fig.update_yaxes(title_text="Real", row=1, col=1)
    fig.update_yaxes(title_text="Real", row=1, col=2)
    
    fig.update_layout(
        title='<b>Matrizes de Confusão</b>',
        title_font_size=16,
        height=500,
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig


def plot_feature_importance_comparison(
    logistic_importance: pd.DataFrame,
    catboost_importance: pd.DataFrame
) -> go.Figure:
    """Compara importância das features entre modelos."""
    
    # Top 10 features do CatBoost
    top_features = catboost_importance.head(10)['feature'].tolist()
    
    # Filtrar regressão logística para as mesmas features (se existirem)
    logistic_filtered = logistic_importance[
        logistic_importance['feature'].isin(top_features)
    ].set_index('feature')
    
    catboost_filtered = catboost_importance[
        catboost_importance['feature'].isin(top_features)
    ].set_index('feature')
    
    fig = go.Figure()
    
    # Regressão Logística (coeficientes)
    if not logistic_filtered.empty:
        fig.add_trace(go.Bar(
            name='Regressão Logística (|coef|)',
            y=logistic_filtered.index,
            x=abs(logistic_filtered['coefficient']),
            orientation='h',
            marker_color=COLORS['logistic']
        ))
    
    # CatBoost
    fig.add_trace(go.Bar(
        name='CatBoost (importance)',
        y=catboost_filtered.index,
        x=catboost_filtered['importance'],
        orientation='h',
        marker_color=COLORS['catboost']
    ))
    
    fig.update_layout(
        title='<b>Importância das Features (Top 10)</b>',
        title_font_size=16,
        xaxis_title='Importância',
        yaxis_title='Feature',
        barmode='group',
        plot_bgcolor=COLORS['background'],
        font=dict(family="Arial, sans-serif", size=12),
        height=600,
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=True, gridcolor='#E8E8E8')
    )
    
    return fig


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run():
    """Executa comparação completa entre os modelos."""
    
    # Carregar dados
    combats_path = PROCESSED_DATA_PATH / "combats_processed.parquet"
    
    if not combats_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {combats_path}")
    
    df_combats = pd.read_parquet(combats_path)
    
    # Treinar Regressão Logística
    logistic_model = PokemonLogisticRegression()
    logistic_metrics = logistic_model.train(df_combats)
    logistic_model.save()
    
    # Treinar CatBoost
    catboost_model = PokemonCatBoost()
    catboost_metrics = catboost_model.train(df_combats)
    catboost_model.save()
    
    # Comparar métricas
    comparison = compare_models(logistic_metrics, catboost_metrics)
    print("Comparação de Métricas:")
    print(comparison.to_string(index=False))
    
    # Exemplos de predição
    print("\nExemplos de Predição:")
    examples = [
        ('Pikachu', 'Bulbasaur'),
        ('Mewtwo', 'Mew'),
        ('Arceus', 'Togepi'),
        ('Kyogre', 'Groudon')
    ]
    
    for p1, p2 in examples:
        try:
            log_result = logistic_model.predict(p1, p2)
            cat_result = catboost_model.predict(p1, p2)
            
            print(f"\n{p1} vs {p2}:")
            print(f"  Regressão Logística: {log_result['vencedor_previsto']} "
                  f"(P1: {log_result['probabilidade_vitoria_p1']:.2%})")
            print(f"  CatBoost:            {cat_result['vencedor_previsto']} "
                  f"(P1: {cat_result['probabilidade_vitoria_p1']:.2%})")
            
            if log_result['vencedor_previsto'] != cat_result['vencedor_previsto']:
                print(f"  ⚠️  DIVERGÊNCIA entre modelos!")
        except ValueError as e:
            print(f"\n{p1} vs {p2}: {e}")
    
    # As figuras não são exibidas automaticamente (removido o .show())
    # Para visualizá‑las, chame as funções explicitamente.
    
    return {
        'logistic_model': logistic_model,
        'catboost_model': catboost_model,
        'logistic_metrics': logistic_metrics,
        'catboost_metrics': catboost_metrics,
        'comparison': comparison
    }


if __name__ == "__main__":
    run()