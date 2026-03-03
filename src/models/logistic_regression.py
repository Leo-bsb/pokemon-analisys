"""
Modelo de Regressão Logística para Predição de Batalhas Pokémon
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import PROCESSED_DATA_PATH, MODELS_PATH
from src.models.model_utils import (
    create_pokemon_stats_dict,
    prepare_features_for_prediction
)


class PokemonLogisticRegression:
    """Modelo de Regressão Logística para predição de batalhas."""
    
    def __init__(self, C: float = 1.0, class_weight: str = 'balanced'):
        """
        Inicializa o modelo.
        
        Args:
            C: Parâmetro de regularização
            class_weight: Peso das classes ('balanced' ou None)
        """
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=42,
            max_iter=1000
        )
        self.pokemon_stats = None
        self.features = ['stats_diff', 'speed_diff', 'attack_diff', 'defense_diff', 'vantagem_p1']
        self.is_trained = False
    
    def prepare_data(self, df_combats: pd.DataFrame) -> tuple:
        """Prepara dados para treinamento."""
        
        df = df_combats.copy()
        
        # Criar variável alvo
        df['target'] = (df['winner_name'] == df['pokemon_1']).astype(int)
        
        # Criar feature de vantagem
        df['vantagem_p1'] = (df['type_advantage'] == 1).astype(int)
        
        # Selecionar features
        X = df[self.features]
        y = df['target']
        
        return X, y
    
    def train(self, df_combats: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Treina o modelo.
        
        Args:
            df_combats: DataFrame com dados de combates
            test_size: Proporção do conjunto de teste
        
        Returns:
            Dicionário com métricas de avaliação
        """
        
        print("Preparando dados...")
        X, y = self.prepare_data(df_combats)
        
        # Criar dicionário de estatísticas dos Pokémon
        self.pokemon_stats = create_pokemon_stats_dict(df_combats)
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("Treinando modelo de Regressão Logística...")
        self.model.fit(X_train, y_train)
        
        # Avaliar
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': pd.DataFrame({
                'feature': self.features,
                'coefficient': self.model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
        }
        
        self.is_trained = True
        
        print(f"\n{'='*80}")
        print("RESULTADOS - REGRESSÃO LOGÍSTICA")
        print(f"{'='*80}")
        print(f"Acurácia no teste: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC no teste: {metrics['roc_auc']:.4f}")
        print(f"\nCoeficientes do modelo:")
        print(metrics['feature_importance'].to_string(index=False))
        print(f"{'='*80}\n")
        
        return metrics
    
    def predict(self, nome1: str, nome2: str) -> dict:
        """
        Prediz o resultado de uma batalha.
        
        Args:
            nome1: Nome do primeiro Pokémon
            nome2: Nome do segundo Pokémon
        
        Returns:
            Dicionário com predição e probabilidade
        """
        
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Preparar features
        features_dict = prepare_features_for_prediction(
            self.pokemon_stats, nome1, nome2, include_categorical=False
        )
        
        # Converter para DataFrame
        X_pred = pd.DataFrame([features_dict])[self.features]
        
        # Predizer
        prob = self.model.predict_proba(X_pred)[0, 1]
        vencedor = nome1 if prob >= 0.5 else nome2
        
        return {
            'pokemon1': nome1,
            'pokemon2': nome2,
            'probabilidade_vitoria_p1': prob,
            'probabilidade_vitoria_p2': 1 - prob,
            'vencedor_previsto': vencedor,
            'confianca': max(prob, 1 - prob)
        }
    
    def save(self, filename: str = 'logistic_regression_model.pkl'):
        """Salva o modelo treinado."""
        
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        filepath = MODELS_PATH / filename
        
        model_data = {
            'model': self.model,
            'pokemon_stats': self.pokemon_stats,
            'features': self.features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    @classmethod
    def load(cls, filename: str = 'logistic_regression_model.pkl'):
        """Carrega um modelo treinado."""
        
        filepath = MODELS_PATH / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.model = model_data['model']
        instance.pokemon_stats = model_data['pokemon_stats']
        instance.features = model_data['features']
        instance.is_trained = True
        
        print(f"Modelo carregado de: {filepath}")
        
        return instance


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run():
    """Executa o treinamento do modelo de Regressão Logística."""
    
    print("="*80)
    print("TREINAMENTO - REGRESSÃO LOGÍSTICA")
    print("="*80 + "\n")
    
    # Carregar dados
    combats_path = PROCESSED_DATA_PATH / "combats_processed.parquet"
    
    if not combats_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {combats_path}")
    
    print("Carregando dados processados...")
    df_combats = pd.read_parquet(combats_path)
    
    # Criar e treinar modelo
    model = PokemonLogisticRegression()
    metrics = model.train(df_combats)
    
    # Salvar modelo
    model.save()
    
    # Exemplos de predição
    print("Exemplos de predição:")
    print("-" * 80)
    
    examples = [
        ('Pikachu', 'Bulbasaur'),
        ('Charizard', 'Blastoise'),
        ('Mewtwo', 'Mew'),
        ('Arceus', 'Togepi')
    ]
    
    for p1, p2 in examples:
        try:
            result = model.predict(p1, p2)
            print(f"\n{p1} vs {p2}:")
            print(f"  Vencedor previsto: {result['vencedor_previsto']}")
            print(f"  Probabilidade {p1}: {result['probabilidade_vitoria_p1']:.2%}")
            print(f"  Confiança: {result['confianca']:.2%}")
        except ValueError as e:
            print(f"\n{p1} vs {p2}: {e}")
    
    print("\n" + "="*80)
    print("TREINAMENTO CONCLUÍDO")
    print("="*80 + "\n")
    
    return model, metrics


if __name__ == "__main__":
    run()
