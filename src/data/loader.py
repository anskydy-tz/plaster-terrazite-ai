"""
Модуль для загрузки и подготовки данных для ML моделей
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Класс для загрузки данных рецептов терразитовой штукатурки"""
    
    def __init__(self, data_path: str = "data/processed/recipes.json"):
        self.data_path = Path(data_path)
        self.recipes = None
        self.df = None
        
    def load_recipes(self) -> List[Dict]:
        """Загрузка рецептов из JSON файла"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.recipes = json.load(f)
        
        logger.info(f"Загружено {len(self.recipes)} рецептов")
        return self.recipes
    
    def prepare_dataframe(self) -> pd.DataFrame:
        """Преобразование рецептов в DataFrame для ML"""
        if self.recipes is None:
            self.load_recipes()
        
        data = []
        
        for recipe in self.recipes:
            # Базовые признаки
            row = {
                'id': recipe['id'],
                'name': recipe['name'],
                'type': recipe['type'],
                'total_weight': recipe.get('total_weight', 1000)
            }
            
            # Добавляем компоненты как отдельные колонки
            for component, weight in recipe['components'].items():
                # Нормализуем название колонки (убираем спецсимволы, сокращаем)
                col_name = self._normalize_component_name(component)
                row[col_name] = weight / 1000  # Переводим в тонны (нормализация)
            
            data.append(row)
        
        self.df = pd.DataFrame(data).fillna(0)
        
        # Логируем информацию о данных
        logger.info(f"Создан DataFrame: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
        logger.info(f"Типы рецептов: {self.df['type'].value_counts().to_dict()}")
        
        return self.df
    
    def _normalize_component_name(self, component: str) -> str:
        """Нормализация названий компонентов для использования в качестве имен колонок"""
        # Убираем лишние пробелы, приводим к нижнему регистру
        normalized = component.strip().lower()
        
        # Заменяем неалфавитно-цифровые символы на подчеркивания
        import re
        normalized = re.sub(r'[^\w\s]', '_', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Сокращаем слишком длинные названия
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        return normalized
    
    def get_features_and_targets(self, target_components: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка признаков и целевых переменных для ML
        
        Args:
            target_components: Список компонентов для предсказания.
                              Если None, использует все компоненты.
        
        Returns:
            X: Признаки (тип рецепта, общие характеристики)
            y: Целевые переменные (проценты компонентов)
        """
        if self.df is None:
            self.prepare_dataframe()
        
        # Признаки: тип рецепта (кодируем) и другие мета-признаки
        X = pd.get_dummies(self.df['type'], prefix='type')
        
        # Если нужны дополнительные признаки, можно добавить здесь
        
        # Целевые переменные: проценты компонентов
        if target_components is None:
            # Используем все колонки компонентов (исключая мета-колонки)
            meta_cols = ['id', 'name', 'type', 'total_weight']
            component_cols = [col for col in self.df.columns if col not in meta_cols]
            y = self.df[component_cols].values
        else:
            # Используем только указанные компоненты
            y = self.df[target_components].values
        
        logger.info(f"Размерность признаков (X): {X.shape}")
        logger.info(f"Размерность целевых переменных (y): {y.shape}")
        
        return X.values, y
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Разделение данных на обучающую и тестовую выборки"""
        from sklearn.model_selection import train_test_split
        
        X, y = self.get_features_and_targets()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"Обучающая выборка: {X_train.shape[0]} образцов")
        logger.info(f"Тестовая выборка: {X_test.shape[0]} образцов")
        
        return X_train, X_test, y_train, y_test
    
    def get_component_statistics(self) -> pd.DataFrame:
        """Статистика по компонентам"""
        if self.df is None:
            self.prepare_dataframe()
        
        # Мета-колонки
        meta_cols = ['id', 'name', 'type', 'total_weight']
        component_cols = [col for col in self.df.columns if col not in meta_cols]
        
        stats = []
        for col in component_cols:
            non_zero = self.df[col][self.df[col] > 0]
            if len(non_zero) > 0:
                stats.append({
                    'component': col,
                    'count': len(non_zero),
                    'mean': non_zero.mean(),
                    'std': non_zero.std(),
                    'min': non_zero.min(),
                    'max': non_zero.max(),
                    'total_used': self.df[col].sum()
                })
        
        return pd.DataFrame(stats).sort_values('count', ascending=False)


if __name__ == "__main__":
    # Пример использования
    loader = DataLoader()
    
    # Загрузка данных
    recipes = loader.load_recipes()
    print(f"Загружено рецептов: {len(recipes)}")
    
    # Подготовка DataFrame
    df = loader.prepare_dataframe()
    print(f"DataFrame: {df.shape}")
    
    # Статистика
    stats = loader.get_component_statistics()
    print(f"\nТоп-10 компонентов по использованию:")
    print(stats.head(10))
    
    # Разделение данных
    X_train, X_test, y_train, y_test = loader.split_data()
    print(f"\nРазделение данных:")
    print(f"Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"Тестовая выборка: {X_test.shape[0]} образцов")
