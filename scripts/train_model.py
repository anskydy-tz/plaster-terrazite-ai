"""
Скрипт для обучения модели предсказания рецептов терразитовой штукатурки
Обновленная версия для работы с манифестами данных
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
import pandas as pd

from src.models.terrazite_model import TerraziteRecipeModel
from src.data.loader import ManifestDataLoader, RecipeLoader
from src.data.processor import DataProcessor
from src.utils.logger import setup_logger

logger = setup_logger()


def prepare_training_data_from_manifest(data_dir: str, target_size=(224, 224)):
    """
    Подготовка данных для обучения с использованием манифестов
    
    Args:
        data_dir: Директория с данными
        target_size: Размер изображений
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), component_names
    """
    logger.info(f"Загрузка данных из манифестов в {data_dir}")
    
    try:
        # Создаем загрузчик манифестов
        loader = ManifestDataLoader(data_dir)
        
        # Загружаем и подготавливаем данные
        datasets = loader.prepare_training_data(
            train_manifest="train",
            val_manifest="val",
            test_manifest="test",
            recipes_json=os.path.join(data_dir, "recipes.json"),
            target_size=target_size
        )
        
        # Извлекаем данные для обучения
        X_train = datasets['train']['images']
        y_train_reg = datasets['train']['labels']
        
        X_val = datasets['val']['images']
        y_val_reg = datasets['val']['labels']
        
        X_test = datasets['test']['images']
        y_test_reg = datasets['test']['labels']
        
        # Для классификации типов рецептов
        # Получаем типы рецептов из манифестов
        train_manifest = datasets['train']['manifest']
        val_manifest = datasets['val']['manifest']
        test_manifest = datasets['test']['manifest']
        
        # Создаем словарь для преобразования типов в числовые метки
        all_types = pd.concat([
            train_manifest['recipe_type'],
            val_manifest['recipe_type'],
            test_manifest['recipe_type']
        ]).unique()
        
        type_to_idx = {t: i for i, t in enumerate(sorted(all_types))}
        idx_to_type = {i: t for t, i in type_to_idx.items()}
        
        # Преобразуем типы рецептов в one-hot векторы
        def create_type_labels(manifest_df):
            labels = []
            for _, row in manifest_df.iterrows():
                recipe_id = str(row['recipe_id'])
                # Находим рецепт в recipes.json для получения типа
                recipe_type = row['recipe_type']
                labels.append(type_to_idx.get(recipe_type, 0))
            return np.array(labels)
        
        y_train_cls = create_type_labels(train_manifest)
        y_val_cls = create_type_labels(val_manifest)
        y_test_cls = create_type_labels(test_manifest)
        
        # Подготавливаем данные в формате для модели
        y_train = {
            'regression_output': y_train_reg,
            'classification_output': tf.keras.utils.to_categorical(
                y_train_cls, len(type_to_idx)
            )
        }
        
        y_val = {
            'regression_output': y_val_reg,
            'classification_output': tf.keras.utils.to_categorical(
                y_val_cls, len(type_to_idx)
            )
        }
        
        # Получаем имена компонентов
        recipes_json_path = os.path.join(data_dir, "recipes.json")
        component_names = loader.get_component_names_from_json(recipes_json_path)
        
        logger.info(f"Данные успешно подготовлены:")
        logger.info(f"  Обучающая выборка: {len(X_train)} образцов")
        logger.info(f"  Валидационная выборка: {len(X_val)} образцов")
        logger.info(f"  Тестовая выборка: {len(X_test)} образцов")
        logger.info(f"  Количество компонентов: {len(component_names)}")
        logger.info(f"  Количество типов рецептов: {len(type_to_idx)}")
        logger.info(f"  Типы рецептов: {list(type_to_idx.keys())}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test_reg, y_test_cls), component_names, idx_to_type
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {e}")
        logger.warning("Создаю синтетические данные для тестирования...")
        return create_synthetic_data()


def get_component_names_from_json(json_path):
    """Вспомогательная функция для получения имен компонентов из JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        all_components = set()
        for recipe in recipes:
            all_components.update(recipe.get('components', {}).keys())
        
        return sorted(list(all_components))
    except Exception as e:
        logger.error(f"Ошибка загрузки компонентов: {e}")
        return []


# Добавляем метод в ManifestDataLoader через monkey patch для совместимости
ManifestDataLoader.get_component_names_from_json = staticmethod(get_component_names_from_json)


def create_synthetic_data():
    """
    Создание синтетических данных для тестирования (резервный вариант)
    """
    logger.info("Создание синтетических данных...")
    
    # Создаем случайные изображения
    n_samples = 100
    X_train = np.random.rand(n_samples, 224, 224, 3).astype('float32')
    X_val = np.random.rand(20, 224, 224, 3).astype('float32')
    X_test = np.random.rand(30, 224, 224, 3).astype('float32')
    
    # Создаем случайные метки для регрессии (компоненты)
    n_components = 15
    y_reg_train = np.random.rand(n_samples, n_components).astype('float32')
    y_reg_val = np.random.rand(20, n_components).astype('float32')
    y_reg_test = np.random.rand(30, n_components).astype('float32')
    
    # Нормализуем чтобы сумма была 1
    y_reg_train = y_reg_train / y_reg_train.sum(axis=1, keepdims=True)
    y_reg_val = y_reg_val / y_reg_val.sum(axis=1, keepdims=True)
    y_reg_test = y_reg_test / y_reg_test.sum(axis=1, keepdims=True)
    
    # Создаем метки классификации
    n_classes = 3  # внутренняя, фасадная, декоративная
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
    
    component_names = [
        'мрамор', 'кварц', 'гранит', 'слюда', 'известняк', 
        'цемент', 'песок', 'вода', 'пигмент_красный', 
        'пигмент_синий', 'пигмент_желтый', 'пластификатор', 
        'волокно', 'добавка_1', 'добавка_2'
    ]
    
    idx_to_type = {0: 'внутренняя', 1: 'фасадная', 2: 'декоративная'}
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_reg_test, y_cls_test), component_names, idx_to_type


def train_model(args):
    """Основная функция обучения"""
    logger.info("="*60)
    logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ TERRAZITE AI")
    logger.info("="*60)
    
    # Подготовка данных
    logger.info("📊 Подготовка данных...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test_reg, y_test_cls), component_names, idx_to_type = prepare_training_data_from_manifest(
        args.data_dir, target_size=(args.image_size, args.image_size)
    )
    
    logger.info(f"📈 Данные подготовлены:")
    logger.info(f"  Обучающая выборка: {len(X_train)} образцов")
    logger.info(f"  Валидационная выборка: {len(X_val)} образцов")
    logger.info(f"  Тестовая выборка: {len(X_test)} образцов")
    logger.info(f"  Компонентов для предсказания: {len(component_names)}")
    logger.info(f"  Типов рецептов: {len(idx_to_type)}")
    
    # Создание модели
    logger.info("🏗️ Создание модели...")
    model = TerraziteRecipeModel(
        num_regression_outputs=len(component_names),
        num_classes=len(idx_to_type),
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        image_size=args.image_size
    )
    
    model.build_model()
    
    # Обучение
    logger.info(f"🎯 Начало обучения на {args.epochs} эпох...")
    logger.info(f"  Размер батча: {args.batch_size}")
    logger.info(f"  Скорость обучения: {args.learning_rate}")
    
    history = model.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_early_stopping=args.early_stopping,
        patience=args.patience
    )
    
    # Оценка модели
    logger.info("📊 Оценка модели на тестовых данных...")
    test_data = (X_test, {
        'regression_output': y_test_reg,
        'classification_output': tf.keras.utils.to_categorical(y_test_cls, len(idx_to_type))
    })
    
    metrics = model.evaluate(test_data)
    
    # Сохранение модели
    model_save_path = args.model_path
    model.save_model(model_save_path)
    
    # Сохранение истории обучения
    history_path = Path(model_save_path).parent / "training_history.json"
    with open(history_path, 'w') as f:
        # Преобразуем numpy типы в стандартные Python типы
        history_serializable = {}
        for key, values in history.history.items():
            history_serializable[key] = [float(v) for v in values]
        json.dump(history_serializable, f, indent=2)
    
    # Сохранение метрик
    metrics_path = Path(model_save_path).parent / "training_metrics.json"
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Сохранение информации о компонентах и типах
    info_path = Path(model_save_path).parent / "model_info.json"
    model_info = {
        'component_names': component_names,
        'recipe_types': idx_to_type,
        'num_components': len(component_names),
        'num_types': len(idx_to_type),
        'image_size': args.image_size,
        'training_date': pd.Timestamp.now().isoformat(),
        'dataset_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Модель сохранена в {model_save_path}")
    logger.info(f"📊 Метрики тестирования: {metrics}")
    logger.info(f"📄 История обучения: {history_path}")
    logger.info(f"📋 Информация о модели: {info_path}")
    
    # Визуализация результатов
    if args.create_plots:
        try:
            plot_dir = Path(args.model_path).parent / "plots"
            plot_dir.mkdir(exist_ok=True)
            
            # Импортируем здесь, чтобы не требовать matplotlib для базовой работы
            import matplotlib.pyplot as plt
            
            # График потерь
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Обучающая выборка')
            plt.plot(history.history['val_loss'], label='Валидационная выборка')
            plt.title('Функция потерь')
            plt.xlabel('Эпоха')
            plt.ylabel('Потери')
            plt.legend()
            plt.grid(True)
            
            # График точности
            plt.subplot(1, 2, 2)
            if 'classification_output_accuracy' in history.history:
                plt.plot(history.history['classification_output_accuracy'], label='Обучающая выборка')
                plt.plot(history.history['val_classification_output_accuracy'], label='Валидационная выборка')
                plt.title('Точность классификации')
                plt.xlabel('Эпоха')
                plt.ylabel('Точность')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plot_path = plot_dir / "training_history.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Графики сохранены: {plot_path}")
        except Exception as e:
            logger.warning(f"Не удалось создать графики: {e}")
    
    logger.info("="*60)
    logger.info("✅ ОБУЧЕНИЕ МОДЕЛИ ЗАВЕРШЕНО")
    logger.info("="*60)
    
    return model, history, metrics


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description="Обучение модели Terrazite AI")
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Директория с данными (по умолчанию: data/processed)')
    parser.add_argument('--model-path', type=str, default='models/terrazite_model.h5',
                       help='Путь для сохранения модели (по умолчанию: models/terrazite_model.h5)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох обучения (по умолчанию: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер батча (по умолчанию: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Скорость обучения (по умолчанию: 0.001)')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout rate (по умолчанию: 0.3)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Размер изображений (по умолчанию: 224)')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Использовать раннюю остановку')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience для ранней остановки (по умолчанию: 10)')
    parser.add_argument('--create-plots', action='store_true',
                       help='Создать графики обучения')
    
    args = parser.parse_args()
    
    # Создание необходимых директорий
    model_dir = Path(args.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверка наличия данных
    data_dir = Path(args.data_dir)
    required_files = [
        data_dir / "data_manifest_train.csv",
        data_dir / "data_manifest_val.csv", 
        data_dir / "data_manifest_test.csv",
        data_dir / "recipes.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.warning(f"Отсутствуют необходимые файлы: {missing_files}")
        logger.info("Сначала запустите создание манифестов: python scripts/create_data_manifest.py")
        if len(missing_files) == len(required_files):
            logger.info("Или создайте тестовые данные: python create_test_excel.py")
    
    # Обучение модели
    try:
        model, history, metrics = train_model(args)
        
        print("\n" + "="*60)
        print("🎉 ОБУЧЕНИЕ МОДЕЛИ УСПЕШНО ЗАВЕРШЕНО!")
        print("="*60)
        print(f"\n📁 Модель сохранена: {args.model_path}")
        print(f"📊 Метрики тестирования:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n📈 Следующие шаги:")
        print(f"  1. Протестируйте модель: python scripts/test_model.py --model-path {args.model_path}")
        print(f"  2. Запустите API сервер: uvicorn src.api.main:app --reload")
        print(f"  3. Запустите веб-интерфейс: streamlit run streamlit_app.py")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
