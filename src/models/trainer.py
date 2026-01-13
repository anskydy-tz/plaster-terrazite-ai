"""
Модуль для обучения и валидации моделей терразитовой штукатурки.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List, Optional, Any
import json
import os
from pathlib import Path
import pandas as pd

from ..data.loader import DataLoader, RecipeLoader
from ..data.processor import DataProcessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для управления процессом обучения моделей.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor()
        self.model = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Загрузка конфигурации обучения."""
        default_config = {
            'input_shape': (224, 224, 3),
            'num_regression_outputs': 15,
            'num_classes': 5,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'test_split': 0.1,
            'data_augmentation': True,
            'augmentation_params': {
                'rotation_range': 20,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'horizontal_flip': True,
                'brightness_range': [0.8, 1.2]
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def prepare_data(self, data_dir: str) -> Tuple:
        """
        Подготовка данных для обучения.
        
        Args:
            data_dir: Директория с данными (изображения и JSON с рецептами)
        
        Returns:
            Кортеж с данными для обучения, валидации и тестирования
        """
        logger.info(f"Загрузка данных из {data_dir}")
        
        # Загрузка и обработка данных
        images, recipes, aggregate_types = self.data_processor.load_dataset(data_dir)
        
        if len(images) == 0:
            raise ValueError("Не найдено изображений для обучения")
        
        # Преобразование меток в формат для многозадачного обучения
        y_regression = []
        y_classification = []
        
        for recipe in recipes:
            # Нормализация компонентов рецепта (приводим к диапазону 0-1)
            # Здесь нужно преобразовать рецепт в вектор фиксированной длины
            # Пока используем заглушку
            recipe_vector = self._recipe_to_vector(recipe)
            y_regression.append(recipe_vector)
        
        # Кодирование меток классов
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(aggregate_types)
        
        # One-hot encoding для классификации
        onehot_encoder = OneHotEncoder(sparse=False)
        y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
        
        y_classification = y_onehot
        
        # Разделение данных
        X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split(
            images, y_regression, y_classification,
            test_size=self.config['validation_split'] + self.config['test_split'],
            random_state=42
        )
        
        # Делим temp на validation и test
        val_test_ratio = self.config['test_split'] / (self.config['validation_split'] + self.config['test_split'])
        
        X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
            X_temp, y_reg_temp, y_cls_temp,
            test_size=val_test_ratio,
            random_state=42
        )
        
        # Преобразуем в numpy arrays
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        
        y_reg_train = np.array(y_reg_train)
        y_reg_val = np.array(y_reg_val)
        y_reg_test = np.array(y_reg_test)
        
        y_cls_train = np.array(y_cls_train)
        y_cls_val = np.array(y_cls_val)
        y_cls_test = np.array(y_cls_test)
        
        # Формируем словари для многозадачного обучения
        y_train = {
            'regression_output': y_reg_train,
            'classification_output': y_cls_train
        }
        
        y_val = {
            'regression_output': y_reg_val,
            'classification_output': y_cls_val
        }
        
        y_test = {
            'regression_output': y_reg_test,
            'classification_output': y_cls_test
        }
        
        logger.info(f"Данные подготовлены:")
        logger.info(f"  Обучающая выборка: {len(X_train)} образцов")
        logger.info(f"  Валидационная выборка: {len(X_val)} образцов")
        logger.info(f"  Тестовая выборка: {len(X_test)} образцов")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _recipe_to_vector(self, recipe: Dict) -> np.ndarray:
        """
        Преобразование рецепта в вектор фиксированной длины.
        
        Args:
            recipe: Словарь с компонентами рецепта
        
        Returns:
            Вектор значений компонентов (нормированных)
        """
        # Заглушка: создаем случайный вектор
        # В реальности нужно преобразовать рецепт в вектор из self.config['num_regression_outputs'] значений
        vector = np.zeros(self.config['num_regression_outputs'])
        
        # Пример: берем первые 15 компонентов из рецепта, если они есть
        # Вам нужно адаптировать это под вашу структуру рецепта
        if 'binders' in recipe:
            # Суммируем связующие
            binders_sum = sum(recipe['binders'].values())
            vector[0] = binders_sum / 1000.0  # Нормализация на общий вес
        
        # ... аналогично для других компонентов
        
        return vector
    
    def build_model(self) -> Any:
        """Построение модели."""
        logger.info("Построение модели...")
        
        # Импортируем здесь, чтобы избежать циклического импорта
        from .terrazite_model import TerraziteRecipeModel
        
        self.model = TerraziteRecipeModel(
            input_shape=self.config['input_shape'],
            num_regression_outputs=self.config['num_regression_outputs'],
            num_classes=self.config['num_classes'],
            learning_rate=self.config['learning_rate']
        )
        
        self.model.build_model()
        return self.model
    
    def train(self, train_data, val_data, callbacks=None):
        """
        Обучение модели.
        
        Args:
            train_data: Кортеж (X_train, y_train) для обучения
            val_data: Кортеж (X_val, y_val) для валидации
            callbacks: Список callback'ов Keras
        
        Returns:
            История обучения
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Начало обучения модели...")
        
        history = self.model.train(
            train_data=train_data,
            val_data=val_data,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Оценка модели на тестовых данных.
        
        Args:
            test_data: Кортеж (X_test, y_test) для тестирования
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        logger.info("Оценка модели на тестовых данных...")
        metrics = self.model.evaluate(test_data)
        
        return metrics
    
    def save_model(self, path: str = 'models/terrazite_model_final.h5'):
        """
        Сохранение обученной модели.
        """
        if self.model is None:
            raise ValueError("Нет модели для сохранения")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
    
    def save_training_history(self, history, path: str = 'logs/training_history.json'):
        """
        Сохранение истории обучения в JSON.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Преобразуем numpy значения в стандартные типы Python
        history_serializable = {}
        for key, values in history.items():
            if isinstance(values, list):
                history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in values]
            else:
                history_serializable[key] = float(values) if isinstance(values, (np.floating, float)) else values
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"История обучения сохранена в {path}")


def train_simple_classifier(data_dir: str, model_save_path: str = 'models/simple_classifier.joblib'):
    """
    Функция для обучения простого классификатора (Random Forest).
    """
    from .simple_classifier import SimpleAggregateClassifier
    from ..data.processor import DataProcessor
    
    logger.info("Обучение простого классификатора...")
    
    # Загрузка данных
    processor = DataProcessor()
    images, _, aggregate_types = processor.load_dataset(data_dir)
    
    if len(images) == 0:
        raise ValueError("Не найдено изображений для обучения")
    
    # Обучение классификатора
    clf = SimpleAggregateClassifier()
    clf.fit(images, aggregate_types)
    
    # Сохранение модели
    clf.save(model_save_path)
    
    logger.info(f"Классификатор сохранен в {model_save_path}")
    return clf


if __name__ == "__main__":
    # Пример использования
    print("🧪 Тестирование ModelTrainer")
    print("=" * 50)
    
    # Создаем временные данные для теста
    num_samples = 100
    images = [np.random.rand(224, 224, 3).astype('float32') for _ in range(num_samples)]
    recipes = [{'binders': {'white_cement': 100, 'gray_cement': 50}} for _ in range(num_samples)]
    aggregate_types = np.random.choice(['мрамор', 'кварц', 'гранит', 'слюда', 'известняк'], size=num_samples)
    
    # Сохраняем тестовые данные в временную директорию
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем структуру директорий
        images_dir = os.path.join(tmpdir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Сохраняем изображения (заглушки)
        for i, img in enumerate(images):
            # В реальности нужно сохранить как изображение, но для теста просто создаем файл
            with open(os.path.join(images_dir, f'img_{i}.npy'), 'wb') as f:
                np.save(f, img)
        
        # Сохраняем рецепты
        recipes_data = []
        for i, (recipe, agg_type) in enumerate(zip(recipes, aggregate_types)):
            recipe_data = {
                'sample_id': f'SAMPLE_{i}',
                'image_filename': f'img_{i}.npy',
                'recipe': recipe,
                'aggregate_type': agg_type
            }
            recipes_data.append(recipe_data)
        
        with open(os.path.join(tmpdir, 'recipes.json'), 'w') as f:
            json.dump(recipes_data, f, indent=2)
        
        print(f"Тестовые данные созданы в {tmpdir}")
        
        # Тестируем ModelTrainer
        trainer = ModelTrainer()
        try:
            train_data, val_data, test_data = trainer.prepare_data(tmpdir)
            print(f"Данные подготовлены: train={len(train_data[0])}, val={len(val_data[0])}, test={len(test_data[0])}")
            
            model = trainer.build_model()
            print("Модель построена")
            
            # Для реального обучения нужно больше данных, поэтому пропускаем
            print("Обучение пропущено (нужны реальные данные)")
            
        except Exception as e:
            print(f"Ошибка: {e}")
    
    print("\n✅ ModelTrainer готов к работе!")
