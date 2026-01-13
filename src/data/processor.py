"""
Модуль для обработки данных рецептов
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

class DataProcessor:
    """Класс для обработки данных рецептов"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.component_columns = None

    def prepare_recipe_features(self, recipes: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Подготовка признаков из рецептов"""
        if not recipes:
            return np.array([]), []

        # Собираем все уникальные компоненты из всех рецептов
        all_components = set()
        for recipe in recipes:
            # Игнорируем некомпонентные поля
            for key in recipe.keys():
                if key not in ['recipe_id', 'name', 'aggregate_type'] and not key.startswith('_'):
                    all_components.add(key)

        self.component_columns = sorted(list(all_components))

        # Создаем матрицу признаков
        features = []
        valid_recipes = []

        for recipe in recipes:
            row = []
            for comp in self.component_columns:
                value = recipe.get(comp, 0)
                # Нормализуем: делим на 1000, как ожидается в тестах
                row.append(float(value) / 1000.0)
            
            features.append(row)
            valid_recipes.append(recipe)

        features = np.array(features, dtype=np.float32)
        logger.info(f"Подготовлены признаки: {features.shape[0]} рецептов, {features.shape[1]} компонентов")
        return features, valid_recipes

    def prepare_targets(self, recipes: List[Dict]) -> Dict[str, Any]:
        """Подготовка целевых переменных"""
        # Регрессия: сумма компонентов (нормализованная)
        regression_targets = []
        for recipe in recipes:
            total = 0.0
            for comp in self.component_columns:
                total += recipe.get(comp, 0)
            regression_targets.append(total / 1000.0)

        # Классификация: тип заполнителя
        classification_targets = []
        for recipe in recipes:
            agg_type = recipe.get('aggregate_type', 'unknown')
            classification_targets.append(agg_type)

        # Кодируем метки классов
        if classification_targets:
            try:
                encoded_classes = self.label_encoder.fit_transform(classification_targets)
                class_names = self.label_encoder.classes_.tolist()
            except:
                # Если есть только один класс или другие проблемы
                encoded_classes = [0] * len(classification_targets)
                class_names = list(set(classification_targets))
        else:
            encoded_classes = []
            class_names = []

        targets = {
            'regression': np.array(regression_targets, dtype=np.float32),
            'classification': np.array(encoded_classes, dtype=np.int64),
            'class_names': class_names
        }

        logger.info(f"Подготовлены целевые переменные: {len(regression_targets)} образцов, {len(class_names)} классов")
        return targets

    def split_dataset(self, features: np.ndarray, targets: Dict[str, Any], 
                      test_size: float = 0.2, val_size: float = 0.2) -> Dict[str, Any]:
        """Разделение датасета на train/val/test"""
        if len(features) == 0:
            return {
                'X_train': np.array([]),
                'X_val': np.array([]),
                'X_test': np.array([]),
                'y_train': np.array([]),
                'y_val': np.array([]),
                'y_test': np.array([])
            }

        # Получаем регрессионные таргеты
        y = targets['regression']
        
        # Сначала делим на train+val и test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y, test_size=test_size, random_state=42, shuffle=True
        )

        # Затем train+val делим на train и val
        val_relative_size = val_size / (1 - test_size) if (1 - test_size) > 0 else 0
        if len(X_temp) > 0 and val_relative_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, random_state=42, shuffle=True
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, np.array([]), y_temp, np.array([])

        # Масштабирование признаков
        if len(X_train) > 0:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else X_val
            X_test_scaled = self.scaler.transform(X_test) if len(X_test) > 0 else X_test
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test

        dataset = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

        logger.info(f"Датасет разделен: train={X_train_scaled.shape[0]}, "
                   f"val={X_val_scaled.shape[0]}, test={X_test_scaled.shape[0]}")
        return dataset

    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Нормализация признаков"""
        if len(X) == 0:
            return X
        return self.scaler.transform(X)
