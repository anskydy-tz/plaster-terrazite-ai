"""
Скрипт для обучения модели предсказания рецептов
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import argparse
import logging

from src.models.terrazite_model import TerraziteRecipeModel
from src.data.loader import DataLoader, RecipeLoader
from src.data.processor import DataProcessor
from src.utils.logger import setup_logger

logger = setup_logger()


def prepare_training_data(data_dir: str):
    """Подготовка данных для обучения"""
    logger.info(f"Загрузка данных из {data_dir}")
    
    # Загрузка изображений и рецептов
    images, recipes, aggregate_types = DataLoader.load_dataset(data_dir)
    
    if not images:
        logger.warning("Нет данных для обучения. Создаю тестовые данные...")
        return create_synthetic_data()
    
    # Преобразование данных
    X = np.array(images)
    
    # Подготовка целевых переменных
    processor = DataProcessor()
    features, valid_recipes = processor.prepare_recipe_features(recipes)
    targets = processor.prepare_targets(valid_recipes)
    
    # Разделение данных
    dataset = processor.split_dataset(features, targets, test_size=0.2, val_size=0.1)
    
    # Создание фиктивных изображений на основе признаков (для прототипа)
    # В реальном проекте здесь должна быть связь изображений с рецептами
    X_train = X[:len(dataset['X_train'])] if len(X) >= len(dataset['X_train']) else np.random.rand(len(dataset['X_train']), 224, 224, 3)
    X_val = X[len(dataset['X_train']):len(dataset['X_train'])+len(dataset['X_val'])] if len(X) >= len(dataset['X_train'])+len(dataset['X_val']) else np.random.rand(len(dataset['X_val']), 224, 224, 3)
    X_test = X[-len(dataset['X_test']):] if len(X) >= len(dataset['X_test']) else np.random.rand(len(dataset['X_test']), 224, 224, 3)
    
    # Подготовка данных в формате для модели
    y_train = {
        'regression_output': dataset['y_reg_train'],
        'classification_output': tf.keras.utils.to_categorical(dataset['y_cls_train'], len(targets['class_names']))
    }
    
    y_val = {
        'regression_output': dataset['y_reg_val'],
        'classification_output': tf.keras.utils.to_categorical(dataset['y_cls_val'], len(targets['class_names']))
    }
    
    return (X_train, y_train), (X_val, y_val), (X_test, dataset['y_reg_test'], dataset['y_cls_test']), targets['class_names']


def create_synthetic_data():
    """Создание синтетических данных для тестирования"""
    logger.info("Создание синтетических данных...")
    
    # Создаем случайные изображения
    n_samples = 100
    X_train = np.random.rand(n_samples, 224, 224, 3).astype('float32')
    X_val = np.random.rand(20, 224, 224, 3).astype('float32')
    X_test = np.random.rand(30, 224, 224, 3).astype('float32')
    
    # Создаем случайные метки
    y_reg_train = np.random.rand(n_samples, 15).astype('float32')
    y_reg_val = np.random.rand(20, 15).astype('float32')
    y_reg_test = np.random.rand(30, 15).astype('float32')
    
    # Нормализуем чтобы сумма была 1
    y_reg_train = y_reg_train / y_reg_train.sum(axis=1, keepdims=True)
    y_reg_val = y_reg_val / y_reg_val.sum(axis=1, keepdims=True)
    y_reg_test = y_reg_test / y_reg_test.sum(axis=1, keepdims=True)
    
    # Создаем метки классификации
    n_classes = 5
    y_cls_train = np.random.randint(0, n_classes, n_samples)
    y_cls_val = np.random.randint(0, n_classes, 20)
    y_cls_test = np.random.randint(0, n_classes, 30)
    
    # Конвертируем в one-hot
    y_cls_train_onehot = tf.keras.utils.to_categorical(y_cls_train, n_classes)
    y_cls_val_onehot = tf.keras.utils.to_categorical(y_cls_val, n_classes)
    
    y_train = {
        'regression_output': y_reg_train,
        'classification_output': y_cls_train_onehot
    }
    
    y_val = {
        'regression_output': y_reg_val,
        'classification_output': y_cls_val_onehot
    }
    
    class_names = ['мрамор', 'кварц', 'гранит', 'слюда', 'известняк']
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_reg_test, y_cls_test), class_names


def train_model(args):
    """Основная функция обучения"""
    logger.info("🚀 Начало обучения модели")
    
    # Подготовка данных
    (X_train, y_train), (X_val, y_val), (X_test, y_reg_test, y_cls_test), class_names = prepare_training_data(args.data_dir)
    
    logger.info(f"Данные подготовлены:")
    logger.info(f"  Обучающая выборка: {len(X_train)} образцов")
    logger.info(f"  Валидационная выборка: {len(X_val)} образцов")
    logger.info(f"  Тестовая выборка: {len(X_test)} образцов")
    logger.info(f"  Классы: {class_names}")
    
    # Создание модели
    model = TerraziteRecipeModel(
        num_regression_outputs=15,
        num_classes=len(class_names),
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate
    )
    
    model.build_model()
    
    # Обучение
    logger.info(f"Начало обучения на {args.epochs} эпох...")
    history = model.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Оценка модели
    logger.info("Оценка модели на тестовых данных...")
    test_data = (X_test, {
        'regression_output': y_reg_test,
        'classification_output': tf.keras.utils.to_categorical(y_cls_test, len(class_names))
    })
    
    metrics = model.evaluate(test_data)
    
    # Сохранение модели
    model.save_model(args.model_path)
    
    # Сохранение истории обучения
    history_path = Path(args.model_path).parent / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Сохранение метрик
    metrics_path = Path(args.model_path).parent / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Модель сохранена в {args.model_path}")
    logger.info(f"📊 Метрики: {metrics}")
    
    return model, history, metrics


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description="Обучение модели Terrazite AI")
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Директория с данными')
    parser.add_argument('--model-path', type=str, default='models/terrazite_model.h5',
                       help='Путь для сохранения модели')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер батча')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Скорость обучения')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    # Создание необходимых директорий
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Обучение модели
    train_model(args)


if __name__ == "__main__":
    main()
