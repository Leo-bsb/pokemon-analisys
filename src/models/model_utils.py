#
"""
Utilitários compartilhados para modelos de ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def create_pokemon_stats_dict(df_combats: pd.DataFrame) -> pd.DataFrame:
    """
    Cria um dicionário com estatísticas únicas de cada Pokémon.
    
    Args:
        df_combats: DataFrame com dados de combates
    
    Returns:
        DataFrame indexado por nome do Pokémon com suas estatísticas
    """
    
    pokemon_cols = ['pokemon', 'generation', 'legendary', 'type1', 'type2',
                    'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'total_stats']
    
    # Extrair informações do Pokémon 1
    df1 = df_combats[['pokemon_1', 'generation_1', 'legendary_1', 'type1_1', 'type2_1',
                       'hp_1', 'attack_1', 'defense_1', 'sp_attack_1', 'sp_defense_1', 
                       'speed_1', 'total_stats_1']].copy()
    df1.columns = pokemon_cols
    
    # Extrair informações do Pokémon 2
    df2 = df_combats[['pokemon_2', 'generation_2', 'legendary_2', 'type1_2', 'type2_2',
                       'hp_2', 'attack_2', 'defense_2', 'sp_attack_2', 'sp_defense_2', 
                       'speed_2', 'total_stats_2']].copy()
    df2.columns = pokemon_cols
    
    # Concatenar e remover duplicatas
    pokemon_stats = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset='pokemon')
    pokemon_stats.set_index('pokemon', inplace=True)
    
    # Tratar valores nulos em tipos
    pokemon_stats['type1'] = pokemon_stats['type1'].fillna('None').astype(str)
    pokemon_stats['type2'] = pokemon_stats['type2'].fillna('None').astype(str)
    
    return pokemon_stats


def calculate_type_advantage(tipos1: List[str], tipos2: List[str]) -> int:
    """
    Calcula vantagem de tipo do primeiro Pokémon sobre o segundo.
    
    Args:
        tipos1: Lista de tipos do primeiro Pokémon
        tipos2: Lista de tipos do segundo Pokémon
    
    Returns:
        1 se tem vantagem, 0 caso contrário
    """
    
    tipo_vantagens = {
        'Normal': [],
        'Fire': ['Grass', 'Ice', 'Bug', 'Steel'],
        'Water': ['Fire', 'Ground', 'Rock'],
        'Electric': ['Water', 'Flying'],
        'Grass': ['Water', 'Ground', 'Rock'],
        'Ice': ['Grass', 'Ground', 'Flying', 'Dragon'],
        'Fighting': ['Normal', 'Ice', 'Rock', 'Dark', 'Steel'],
        'Poison': ['Grass', 'Fairy'],
        'Ground': ['Fire', 'Electric', 'Poison', 'Rock', 'Steel'],
        'Flying': ['Grass', 'Fighting', 'Bug'],
        'Psychic': ['Fighting', 'Poison'],
        'Bug': ['Grass', 'Psychic', 'Dark'],
        'Rock': ['Fire', 'Ice', 'Flying', 'Bug'],
        'Ghost': ['Psychic', 'Ghost'],
        'Dragon': ['Dragon'],
        'Dark': ['Psychic', 'Ghost'],
        'Steel': ['Ice', 'Rock', 'Fairy'],
        'Fairy': ['Fighting', 'Dragon', 'Dark']
    }
    
    for t1 in tipos1:
        if t1 in tipo_vantagens:
            for t2 in tipos2:
                if t2 in tipo_vantagens[t1]:
                    return 1
    return 0


def prepare_features_for_prediction(
    pokemon_stats: pd.DataFrame,
    nome1: str,
    nome2: str,
    include_categorical: bool = False
) -> Dict:
    """
    Prepara features para predição dado dois nomes de Pokémon.
    
    Args:
        pokemon_stats: DataFrame com estatísticas dos Pokémon
        nome1: Nome do primeiro Pokémon
        nome2: Nome do segundo Pokémon
        include_categorical: Se True, inclui features categóricas
    
    Returns:
        Dicionário com features calculadas
    """
    
    if nome1 not in pokemon_stats.index or nome2 not in pokemon_stats.index:
        raise ValueError(f"Pokémon não encontrado no banco de dados: {nome1} ou {nome2}")
    
    p1 = pokemon_stats.loc[nome1]
    p2 = pokemon_stats.loc[nome2]
    
    # Calcular diferenças
    stats_diff = p1['total_stats'] - p2['total_stats']
    speed_diff = p1['speed'] - p2['speed']
    attack_diff = p1['attack'] - p2['attack']
    defense_diff = p1['defense'] - p2['defense']
    
    # Calcular vantagem de tipo
    tipos1 = [p1['type1']] if p1['type1'] != 'None' else []
    if p1['type2'] != 'None':
        tipos1.append(p1['type2'])
    
    tipos2 = [p2['type1']] if p2['type1'] != 'None' else []
    if p2['type2'] != 'None':
        tipos2.append(p2['type2'])
    
    vantagem = calculate_type_advantage(tipos1, tipos2)
    
    # Features básicas (numéricas)
    features = {
        'stats_diff': stats_diff,
        'speed_diff': speed_diff,
        'attack_diff': attack_diff,
        'defense_diff': defense_diff,
        'vantagem_p1': vantagem
    }
    
    # Adicionar features categóricas se solicitado
    if include_categorical:
        features.update({
            'generation_1': str(p1['generation']),
            'generation_2': str(p2['generation']),
            'legendary_1': str(p1['legendary']),
            'legendary_2': str(p2['legendary']),
            'type1_1': p1['type1'],
            'type1_2': p2['type1'],
            'type2_1': p1['type2'],
            'type2_2': p2['type2']
        })
    
    return features
