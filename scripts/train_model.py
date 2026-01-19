"""
Скрипт для обучения модели подбора терразитовой штукатурки
Использует ManifestDataLoader для работы с манифестами изображений
"""
import sys
from pathlib import Path

# Добавляем путь для импорта модулей проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json
import logging
import argparse
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import ManifestDataLoader
from src.utils.logger import setup_logger

# Настройка логгера
logger = setup_logger("train_model")

# Настройка TensorFlow для лучшей производительности
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info(f"Обнаружен GPU: {physical_devices[0]}")
        USE_GPU = True
    else:
        logger.info("GPU не обнаружен, используется CPU")
        USE_GPU = False
except:
    logger.info("Не удалось настроить GPU, используется CPU")
    USE_GPU = False


class TerraziteModel:
    """Класс для создания и обучения модели подбора терразитовой штукатурки"""
    
    def __init__(self, input_shape=(224, 224, 3), num_components=15):
        """
        Инициализация модели
        
        Args:
            input_shape: Размер входных изображений
            num_components: Количество компонентов для предсказания
        """
        self.input_shape = input_shape
        self.num_components = num_components
        self.model = None
        self.history = None
        self.component_names = []
        
        logger.info(f"Инициализация модели с входным размером: {input_shape}")
        logger.info(f"Количество предсказываемых компонентов: {num_components}")
    
    def build_cnn_model(self):
        """Создание CNN модели для анализа изображений"""
        logger.info("Создание CNN модели...")
        
        # Входной слой для изображений
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # Блок 1: Извлечение базовых признаков
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Блок 2: Извлечение текстурных признаков
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Блок 3: Извлечение детальных признаков (цвет, текстура)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Блок 4: Дополнительные слои для сложных текстур
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Полносвязные слои для регрессии компонентов
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Выходной слой: предсказание процентов компонентов
        # Используем softmax, чтобы сумма была примерно 1 (100%)
        outputs = layers.Dense(self.num_components, 
                              activation='softmax',
                              name='component_output')(x)
        
        # Создание модели
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Компиляция модели
        optimizer = optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Среднеквадратичная ошибка для регрессии
            metrics=['mae', 'cosine_similarity']  # Средняя абсолютная ошибка и косинусное сходство
        )
        
        logger.info("Модель успешно создана и скомпилирована")
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def build_advanced_model(self):
        """Создание расширенной модели с несколькими выходами"""
        logger.info("Создание расширенной модели...")
        
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # Базовый CNN блок (можно заменить на предобученную модель)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Разделение на ветви для разных типов признаков
        # Ветвь для основных компонентов (мрамор, кварц и т.д.)
        main_branch = layers.Dense(128, activation='relu')(x)
        main_branch = layers.Dropout(0.3)(main_branch)
        main_output = layers.Dense(self.num_components, 
                                  activation='softmax',
                                  name='main_components')(main_branch)
        
        # Ветвь для пигментов
        pigment_branch = layers.Dense(64, activation='relu')(x)
        pigment_branch = layers.Dropout(0.3)(pigment_branch)
        pigment_output = layers.Dense(5, activation='softmax',
                                     name='pigments')(pigment_branch)
        
        # Ветвь для типа штукатурки
        type_branch = layers.Dense(32, activation='relu')(x)
        type_branch = layers.Dropout(0.2)(type_branch)
        type_output = layers.Dense(3, activation='softmax',
                                  name='plaster_type')(type_branch)
        
        # Создание модели с несколькими выходами
        self.model = models.Model(
            inputs=inputs,
            outputs=[main_output, pigment_output, type_output]
        )
        
        # Компиляция с разными весами для выходов
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'main_components': 'mse',
                'pigments': 'mse',
                'plaster_type': 'categorical_crossentropy'
            },
            loss_weights={
                'main_components': 0.7,
                'pigments': 0.2,
                'plaster_type': 0.1
            },
            metrics=['mae', 'accuracy']
        )
        
        logger.info("Расширенная модель успешно создана")
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def train(self, train_data, val_data, epochs=100, batch_size=16, 
              model_type='cnn', callbacks_list=None):
        """
        Обучение модели
        
        Args:
            train_data: Кортеж (X_train, y_train) для обучения
            val_data: Кортеж (X_val, y_val) для валидации
            epochs: Количество эпох
            batch_size: Размер батча
            model_type: Тип модели ('cnn' или 'advanced')
            callbacks_list: Список callback'ов
            
        Returns:
            history: История обучения
        """
        logger.info(f"Начало обучения модели. Эпох: {epochs}, Батч: {batch_size}")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер валидационной выборки: {X_val.shape}")
        
        # Построение модели
        if model_type == 'advanced':
            self.build_advanced_model()
        else:
            self.build_cnn_model()
        
        # Callbacks
        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()
        
        # Обучение модели
        logger.info("Запуск обучения...")
        
        try:
            if model_type == 'advanced':
                # Для расширенной модели нужны разные y для каждого выхода
                y_train_dict = {
                    'main_components': y_train,
                    'pigments': y_train[:, :5],  # Первые 5 компонентов - пигменты
                    'plaster_type': self._get_plaster_type_labels(X_train)
                }
                y_val_dict = {
                    'main_components': y_val,
                    'pigments': y_val[:, :5],
                    'plaster_type': self._get_plaster_type_labels(X_val)
                }
                
                self.history = self.model.fit(
                    X_train, y_train_dict,
                    validation_data=(X_val, y_val_dict),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    verbose=1
                )
            else:
                # Для обычной модели
                self.history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    verbose=1
                )
            
            logger.info("Обучение успешно завершено!")
            return self.history
            
        except Exception as e:
            logger.error(f"Ошибка во время обучения: {e}")
            raise
    
    def _get_default_callbacks(self):
        """Получение стандартных callback'ов для обучения"""
        # Создание директории для сохранения моделей
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # Имя модели с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"terrazite_model_{timestamp}"
        
        callbacks = [
            # Сохранение лучшей модели
            callbacks.ModelCheckpoint(
                filepath=str(model_dir / f"{model_name}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Ранняя остановка
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Уменьшение learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            ),
            
            # Логгирование в TensorBoard
            callbacks.TensorBoard(
                log_dir=str(Path("logs") / model_name),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def _get_plaster_type_labels(self, X_data):
        """Генерация меток типа штукатурки (для расширенной модели)"""
        # В реальной реализации нужно получать типы из данных
        # Сейчас генерируем случайные метки для тестирования
        num_samples = X_data.shape[0]
        types = np.random.choice([0, 1, 2], size=(num_samples, 3))
        return types / types.sum(axis=1, keepdims=True)  # One-hot encoding
    
    def evaluate(self, test_data):
        """Оценка модели на тестовых данных"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        X_test, y_test = test_data
        
        logger.info(f"Оценка модели на тестовых данных ({X_test.shape[0]} образцов)")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info("Результаты оценки:")
        for i, metric in enumerate(self.model.metrics_names):
            logger.info(f"  {metric}: {results[i]:.4f}")
        
        # Предсказания
        predictions = self.model.predict(X_test, verbose=0)
        
        # Расчет дополнительных метрик
        mae = np.mean(np.abs(predictions - y_test))
        mse = np.mean((predictions - y_test) ** 2)
        
        logger.info(f"  Дополнительные метрики:")
        logger.info(f"    MAE: {mae:.4f}")
        logger.info(f"    MSE: {mse:.4f}")
        
        return results, predictions
    
    def save_model(self, path="models/terrazite_model.h5"):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Нет модели для сохранения!")
        
        model_path = Path(path)
        model_path.parent.mkdir(exist_ok=True)
        
        self.model.save(model_path)
        logger.info(f"Модель сохранена: {model_path}")
        
        # Сохранение архитектуры в JSON
        model_json = model_path.with_suffix('.json')
        with open(model_json, 'w') as f:
            f.write(self.model.to_json())
        
        logger.info(f"Архитектура модели сохранена: {model_json}")
    
    def load_model(self, path):
        """Загрузка модели"""
        model_path = Path(path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        logger.info(f"Модель загружена: {model_path}")
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def plot_training_history(self, save_path="models/training_history.png"):
        """Визуализация истории обучения"""
        if self.history is None:
            logger.warning("Нет истории обучения для визуализации")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Обучающая')
        axes[0, 0].plot(history['val_loss'], label='Валидационная')
        axes[0, 0].set_title('Функция потерь (Loss)')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in history:
            axes[0, 1].plot(history['mae'], label='Обучающая')
            axes[0, 1].plot(history['val_mae'], label='Валидационная')
            axes[0, 1].set_title('Средняя абсолютная ошибка (MAE)')
            axes[0, 1].set_xlabel('Эпоха')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate (если есть)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Эпоха')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cosine Similarity (если есть)
        if 'cosine_similarity' in history:
            axes[1, 1].plot(history['cosine_similarity'], label='Обучающая')
            axes[1, 1].plot(history['val_cosine_similarity'], label='Валидационная')
            axes[1, 1].set_title('Косинусное сходство')
            axes[1, 1].set_xlabel('Эпоха')
            axes[1, 1].set_ylabel('Cosine Similarity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Графики обучения сохранены: {save_path}")
    
    def plot_predictions_vs_actual(self, test_data, save_path="models/predictions_vs_actual.png"):
        """Визуализация сравнения предсказаний и реальных значений"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        X_test, y_test = test_data
        predictions = self.model.predict(X_test, verbose=0)
        
        # Выбираем несколько компонентов для визуализации
        num_components_to_plot = min(8, self.num_components)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_components_to_plot):
            ax = axes[i]
            
            # Scatter plot предсказаний vs реальных значений
            ax.scatter(y_test[:, i], predictions[:, i], alpha=0.6, s=30)
            
            # Линия идеальных предсказаний
            min_val = min(y_test[:, i].min(), predictions[:, i].min())
            max_val = max(y_test[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Реальные значения')
            ax.set_ylabel('Предсказания')
            ax.set_title(f'Компонент {i+1}')
            ax.grid(True, alpha=0.3)
            
            # Добавляем R^2 в заголовок
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test[:, i], predictions[:, i])
            ax.set_title(f'Компонент {i+1} (R²={r2:.3f})')
        
        # Скрываем неиспользованные subplots
        for i in range(num_components_to_plot, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Сравнение предсказаний и реальных значений по компонентам', fontsize=14)
        plt.tight_layout()
        
        # Сохранение
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График сравнения предсказаний сохранен: {save_path}")


def prepare_data_for_training():
    """Подготовка данных для обучения"""
    logger.info("="*60)
    logger.info("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    logger.info("="*60)
    
    try:
        # Создаем загрузчик данных
        loader = ManifestDataLoader()
        
        # Загружаем и готовим все данные
        datasets = loader.prepare_training_data(
            train_manifest="train",
            val_manifest="val",
            test_manifest="test",
            recipes_json="data/processed/recipes.json",
            target_size=(224, 224)
        )
        
        # Проверяем, что данные загружены
        if not datasets:
            raise ValueError("Не удалось загрузить данные")
        
        # Получаем данные для обучения, валидации и тестирования
        train_data = (datasets['train']['images'], datasets['train']['labels'])
        val_data = (datasets['val']['images'], datasets['val']['labels'])
        test_data = (datasets['test']['images'], datasets['test']['labels'])
        
        logger.info(f"\nДанные подготовлены:")
        logger.info(f"  Обучающая выборка: {train_data[0].shape[0]} изображений")
        logger.info(f"  Валидационная выборка: {val_data[0].shape[0]} изображений")
        logger.info(f"  Тестовая выборка: {test_data[0].shape[0]} изображений")
        
        # Получаем имена компонентов
        component_names = loader.get_component_names_from_json("data/processed/recipes.json")
        logger.info(f"\nКомпоненты для предсказания ({len(component_names)}):")
        for i, name in enumerate(component_names[:10]):  # Показываем первые 10
            logger.info(f"  {i+1}. {name}")
        if len(component_names) > 10:
            logger.info(f"  ... и еще {len(component_names) - 10} компонентов")
        
        return train_data, val_data, test_data, component_names
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Основная функция обучения"""
    parser = argparse.ArgumentParser(description='Обучение модели подбора терразитовой штукатурки')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=16, help='Размер батча')
    parser.add_argument('--model-type', choices=['cnn', 'advanced'], default='cnn',
                       help='Тип модели (cnn или advanced)')
    parser.add_argument('--test-only', action='store_true', 
                       help='Только тестирование без обучения')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Путь к предобученной модели для тестирования')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ TERRAZITE AI")
    logger.info("="*60)
    logger.info(f"Параметры:")
    logger.info(f"  Эпох: {args.epochs}")
    logger.info(f"  Размер батча: {args.batch_size}")
    logger.info(f"  Тип модели: {args.model_type}")
    logger.info(f"  Тестирование только: {args.test_only}")
    logger.info(f"  GPU доступен: {USE_GPU}")
    
    try:
        # Подготовка данных
        train_data, val_data, test_data, component_names = prepare_data_for_training()
        
        # Создание модели
        model = TerraziteModel(
            input_shape=(224, 224, 3),
            num_components=len(component_names)
        )
        model.component_names = component_names
        
        if args.test_only:
            # Только тестирование существующей модели
            if args.model_path:
                model.load_model(args.model_path)
            else:
                # Ищем последнюю сохраненную модель
                model_dir = Path("models")
                model_files = list(model_dir.glob("terrazite_model_*.h5"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    model.load_model(latest_model)
                else:
                    raise FileNotFoundError("Не найдены предобученные модели")
            
            # Оценка модели
            results, predictions = model.evaluate(test_data)
            
            # Визуализация
            model.plot_predictions_vs_actual(test_data)
            
        else:
            # Обучение новой модели
            history = model.train(
                train_data=train_data,
                val_data=val_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_type=args.model_type
            )
            
            # Сохранение модели
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"models/terrazite_model_{timestamp}.h5"
            model.save_model(model_save_path)
            
            # Визуализация истории обучения
            model.plot_training_history()
            model.plot_predictions_vs_actual(test_data)
            
            # Оценка на тестовых данных
            logger.info("\n" + "="*60)
            logger.info("ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
            logger.info("="*60)
            
            results, predictions = model.evaluate(test_data)
            
            # Сохранение результатов
            save_results(history, results, component_names, args)
        
        logger.info("\n" + "="*60)
        logger.info("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info("="*60)
        
        return model
        
    except Exception as e:
        logger.error(f"Ошибка во время выполнения: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def save_results(history, results, component_names, args):
    """Сохранение результатов обучения"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохранение истории обучения
    history_df = pd.DataFrame(history.history)
    history_path = results_dir / f"training_history_{timestamp}.csv"
    history_df.to_csv(history_path, index=False)
    
    # Сохранение метрик
    metrics = {
        'timestamp': timestamp,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'model_type': args.model_type,
        'final_loss': history.history['val_loss'][-1],
        'final_mae': history.history.get('val_mae', [0])[-1],
        'num_components': len(component_names)
    }
    
    metrics_path = results_dir / f"training_metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # Сохранение информации о компонентах
    components_info = {
        'component_names': component_names,
        'num_components': len(component_names),
        'timestamp': timestamp
    }
    
    components_path = results_dir / f"components_info_{timestamp}.json"
    with open(components_path, 'w', encoding='utf-8') as f:
        json.dump(components_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nРезультаты сохранены:")
    logger.info(f"  История обучения: {history_path}")
    logger.info(f"  Метрики: {metrics_path}")
    logger.info(f"  Информация о компонентах: {components_path}")
    
    # Вывод сводки
    print("\n" + "="*60)
    print("СВОДКА РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("="*60)
    print(f"Время завершения: {timestamp}")
    print(f"Количество эпох: {args.epochs}")
    print(f"Размер батча: {args.batch_size}")
    print(f"Тип модели: {args.model_type}")
    print(f"Финальная loss: {metrics['final_loss']:.4f}")
    print(f"Финальная MAE: {metrics['final_mae']:.4f}")
    print(f"Количество компонентов: {metrics['num_components']}")
    print("="*60)


if __name__ == "__main__":
    main()
