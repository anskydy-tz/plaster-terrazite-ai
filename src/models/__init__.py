"""
Пакет моделей машинного обучения для предсказания рецептов терразитовой штукатурки
"""

from .terrazite_model import TerraziteRecipeModel
from .simple_classifier import SimpleAggregateClassifier

__all__ = [
    'TerraziteRecipeModel',
    'SimpleAggregateClassifier'
]
