# -*- coding: utf-8 -*-
"""
Módulo de modelos de Machine Learning para predição de batalhas Pokémon
"""

from src.models.logistic_regression import PokemonLogisticRegression
from src.models.catboost_model import PokemonCatBoost

__all__ = ['PokemonLogisticRegression', 'PokemonCatBoost']
