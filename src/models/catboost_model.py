"""
Modelo CatBoost para Predição de Batalhas Pokémon
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import PROCESSED_DATA_PATH, MODELS_PATH
from src.models.model_utils import (
    create_pokemon_stats_dict,
    prepare_features_for_prediction
)


class PokemonCatBoost:
    """Modelo CatBoost para predição de batalhas."""
    
    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        verbose: int = 100
    ):
        """
        Inicializa o modelo.
        
        Args:
            iterations: Número de árvores
            learning_rate: Taxa de aprendizado
            depth: Profundidade das árvores
            verbose: Frequência de print durante treinamento
        """
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            od_type='Iter',
            od_wait=100,
            verbose=verbose,
            use_best_model=True
        )
        self.pokemon_stats = None
        self.num_features = ['stats_diff', 'speed_diff', 'attack_diff', 'defense_diff', 'vantagem_p1']
        self.cat_features = ['generation_1', 'generation_2', 'legendary_1', 'legendary_2',
                             'type1_1', 'type1_2', 'type2_1', 'type2_2']
        self.all_features = self.num_features + self.cat_features
        self.is_trained = False
    
    def prepare_data(self, df_combats: pd.DataFrame) -> tuple:
        """Prepara dados para treinamento."""
        
        df = df_combats.copy()
        
        # Criar variável alvo
        df['target'] = (df['winner_name'] == df['pokemon_1']).astype(int)
        
        # Criar feature de vantagem
        df['vantagem_p1'] = (df['type_advantage'] == 1).astype(int)
        
        # Converter categóricas para string
        for col in self.cat_features:
            df[col] = df[col].astype(str).fillna('None')
        
        # Selecionar features
        X = df[self.all_features].copy()
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
        
        # Criar Pools do CatBoost
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        test_pool = Pool(X_test, y_test, cat_features=self.cat_features)
        
        print("Treinando modelo CatBoost...")
        self.model.fit(train_pool, eval_set=test_pool, plot=False)
        
        # Avaliar
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Importância das features
        feature_importance = pd.DataFrame({
            'feature': self.all_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
        
        self.is_trained = True
        
        print(f"\n{'='*80}")
        print("RESULTADOS - CATBOOST")
        print(f"{'='*80}")
        print(f"Acurácia no teste: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC no teste: {metrics['roc_auc']:.4f}")
        print(f"\nImportância das features (Top 10):")
        print(feature_importance.head(10).to_string(index=False))
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
            self.pokemon_stats, nome1, nome2, include_categorical=True
        )
        
        # Converter para DataFrame
        X_pred = pd.DataFrame([features_dict])[self.all_features]
        
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
    
    def save(self, filename: str = 'catboost_model.pkl'):
        """Salva o modelo treinado."""
        
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        filepath = MODELS_PATH / filename
        
        model_data = {
            'model': self.model,
            'pokemon_stats': self.pokemon_stats,
            'num_features': self.num_features,
            'cat_features': self.cat_features,
            'all_features': self.all_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    @classmethod
    def load(cls, filename: str = 'catboost_model.pkl'):
        """Carrega um modelo treinado."""
        
        filepath = MODELS_PATH / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.model = model_data['model']
        instance.pokemon_stats = model_data['pokemon_stats']
        instance.num_features = model_data['num_features']
        instance.cat_features = model_data['cat_features']
        instance.all_features = model_data['all_features']
        instance.is_trained = True
        
        print(f"Modelo carregado de: {filepath}")
        
        return instance


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run():
    """Executa o treinamento do modelo CatBoost."""
    
    print("="*80)
    print("TREINAMENTO - CATBOOST")
    print("="*80 + "\n")
    
    # Carregar dados
    combats_path = PROCESSED_DATA_PATH / "combats_processed.parquet"
    
    if not combats_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {combats_path}")
    
    print("Carregando dados processados...")
    df_combats = pd.read_parquet(combats_path)
    
    # Criar e treinar modelo
    model = PokemonCatBoost()
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
        ('Arceus', 'Togepi'),
        ('Kyogre', 'Groudon')
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
