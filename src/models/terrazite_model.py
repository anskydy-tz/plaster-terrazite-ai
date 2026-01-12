"""
Основная модель для предсказания рецепта терразитовой штукатурки
Многозадачная модель: регрессия для компонентов + классификация для типа заполнителя
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TerraziteRecipeModel:
    """
    Нейронная сеть для предсказания рецепта терразитовой штукатурки по изображению
    
    Архитектура:
    - Вход: изображение 224x224x3
    - Backbone: EfficientNet-B0 (предобученный)
    - Выход 1: регрессия для 10+ компонентов рецепта
    - Выход 2: классификация типа минерального заполнителя
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_regression_outputs: int = 15,
        num_classes: int = 5,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Инициализация модели
        
        Args:
            input_shape: Размер входного изображения (высота, ширина, каналы)
            num_regression_outputs: Количество выходов регрессии (компоненты рецепта)
            num_classes: Количество классов для классификации заполнителя
            dropout_rate: Rate для слоев Dropout
            learning_rate: Скорость обучения
        """
        self.input_shape = input_shape
        self.num_regression_outputs = num_regression_outputs
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        logger.info(f"Инициализирована модель с параметрами:")
        logger.info(f"  input_shape: {input_shape}")
        logger.info(f"  regression_outputs: {num_regression_outputs}")
        logger.info(f"  classes: {num_classes}")
    
    def build_model(self) -> Model:
        """
        Построение архитектуры нейронной сети
        
        Returns:
            Скомпилированная модель Keras
        """
        # Входной слой
        inputs = layers.Input(shape=self.input_shape)
        
        # Базовый энкодер (предобученный)
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        
        # Замораживаем базовые слои (можно разморозить при дообучении)
        base_model.trainable = False
        
        # Общие признаки
        x = base_model.output
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Ветвь 1: Регрессия для компонентов рецепта
        reg_branch = layers.Dense(256, activation='relu')(x)
        reg_branch = layers.BatchNormalization()(reg_branch)
        reg_branch = layers.Dropout(self.dropout_rate * 0.7)(reg_branch)
        reg_branch = layers.Dense(128, activation='relu')(reg_branch)
        reg_branch = layers.Dense(64, activation='relu')(reg_branch)
        
        # Выход регрессии - проценты компонентов (используем sigmoid для значений 0-1)
        regression_output = layers.Dense(
            self.num_regression_outputs,
            activation='sigmoid',
            name='regression_output'
        )(reg_branch)
        
        # Ветвь 2: Классификация типа заполнителя
        cls_branch = layers.Dense(128, activation='relu')(x)
        cls_branch = layers.BatchNormalization()(cls_branch)
        cls_branch = layers.Dropout(self.dropout_rate * 0.7)(cls_branch)
        cls_branch = layers.Dense(64, activation='relu')(cls_branch)
        
        # Выход классификации
        classification_output = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='classification_output'
        )(cls_branch)
        
        # Собираем модель
        self.model = Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='terrazite_recipe_model'
        )
        
        # Компиляция модели
        self.compile_model()
        
        logger.info("Модель успешно построена")
        return self.model
    
    def compile_model(self) -> None:
        """Компиляция модели с оптимизатором и функциями потерь"""
        
        # Оптимизатор
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Функции потерь для каждой задачи
        losses = {
            'regression_output': 'mse',  # Mean Squared Error для регрессии
            'classification_output': 'categorical_crossentropy'  # Для классификации
        }
        
        # Веса потерь (можно настраивать)
        loss_weights = {
            'regression_output': 0.7,  # Более важна точность рецепта
            'classification_output': 0.3  # Менее важен тип заполнителя
        }
        
        # Метрики для каждой задачи
        metrics = {
            'regression_output': [
                'mae',  # Mean Absolute Error
                keras.metrics.RootMeanSquaredError(name='rmse')
            ],
            'classification_output': [
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        logger.info("Модель успешно скомпилирована")
    
    def train(
        self,
        train_data: Tuple[np.ndarray, Dict],
        val_data: Optional[Tuple[np.ndarray, Dict]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[List] = None
    ) -> Dict:
        """
        Обучение модели
        
        Args:
            train_data: Кортеж (X_train, y_dict_train) для обучения
            val_data: Кортеж (X_val, y_dict_val) для валидации
            epochs: Количество эпох обучения
            batch_size: Размер батча
            callbacks: Список callback'ов Keras
        
        Returns:
            История обучения
        """
        X_train, y_train_dict = train_data
        
        # Подготовка валидационных данных
        if val_data is not None:
            X_val, y_val_dict = val_data
            validation_data = (X_val, y_val_dict)
        else:
            validation_data = None
        
        # Callbacks по умолчанию
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        logger.info(f"Начало обучения на {len(X_train)} образцах")
        logger.info(f"Параметры: epochs={epochs}, batch_size={batch_size}")
        
        # Обучение
        self.history = self.model.fit(
            X_train,
            y_train_dict,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Обучение завершено")
        return self.history.history
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Предсказание рецепта для одного изображения
        
        Args:
            image: Изображение в формате numpy array (1, H, W, 3)
        
        Returns:
            Словарь с предсказаниями:
            - 'recipe_components': проценты компонентов
            - 'aggregate_type': тип заполнителя
            - 'confidence': уверенность предсказания
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Предсказание
        recipe_pred, aggregate_pred = self.model.predict(image, verbose=0)
        
        # Обработка результатов
        recipe_percentages = recipe_pred[0] * 100  # Преобразуем в проценты
        aggregate_idx = np.argmax(aggregate_pred[0])
        aggregate_confidence = aggregate_pred[0][aggregate_idx] * 100
        
        # Типы заполнителей (можно расширить)
        aggregate_types = ['мрамор', 'кварц', 'гранит', 'слюда', 'известняк']
        aggregate_type = aggregate_types[aggregate_idx] if aggregate_idx < len(aggregate_types) else 'неизвестно'
        
        return {
            'recipe_components': recipe_percentages.tolist(),
            'aggregate_type': aggregate_type,
            'confidence': float(aggregate_confidence),
            'aggregate_probabilities': aggregate_pred[0].tolist()
        }
    
    def evaluate(self, test_data: Tuple[np.ndarray, Dict]) -> Dict:
        """
        Оценка модели на тестовых данных
        
        Args:
            test_data: Кортеж (X_test, y_dict_test) для тестирования
        
        Returns:
            Словарь с метриками оценки
        """
        X_test, y_test_dict = test_data
        
        logger.info(f"Оценка на {len(X_test)} тестовых образцах")
        
        results = self.model.evaluate(X_test, y_test_dict, verbose=0)
        
        # Формируем словарь с метриками
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        logger.info(f"Результаты оценки:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path: str = 'models/terrazite_model.h5') -> None:
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения файла модели
        """
        self.model.save(path)
        logger.info(f"Модель сохранена в {path}")
    
    def load_model(self, path: str = 'models/terrazite_model.h5') -> None:
        """
        Загрузка модели
        
        Args:
            path: Путь к файлу модели
        """
        self.model = keras.models.load_model(path)
        logger.info(f"Модель загружена из {path}")
    
    def summary(self) -> None:
        """Вывод информации о модели"""
        self.model.summary()
    
    def _get_default_callbacks(self) -> List:
        """Создание стандартных callback'ов для обучения"""
        callbacks = [
            # Early stopping для предотвращения переобучения
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            # Уменьшение learning rate при плато
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Сохранение лучшей модели
            keras.callbacks.ModelCheckpoint(
                'models/best_terrazite_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Логирование в TensorBoard
            keras.callbacks.TensorBoard(
                log_dir='logs/tensorboard',
                histogram_freq=1
            ),
            # Сохранение истории обучения
            keras.callbacks.CSVLogger(
                'logs/training_history.csv'
            )
        ]
        
        return callbacks
    
    def fine_tune(self, unfreeze_layers: int = 50) -> None:
        """
        Дообучение модели (разморозка части слоев)
        
        Args:
            unfreeze_layers: Количество размораживаемых слоев
        """
        logger.info(f"Разморозка {unfreeze_layers} слоев для дообучения")
        
        # Размораживаем часть слоев
        for layer in self.model.layers[-unfreeze_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        
        # Перекомпилируем с меньшим learning rate
        self.learning_rate = 1e-5
        self.compile_model()
        
        logger.info("Модель готова к дообучению")


def create_simple_model(input_shape=(224, 224, 3)) -> Model:
    """
    Создание упрощенной модели для быстрого прототипирования
    
    Args:
        input_shape: Размер входного изображения
    
    Returns:
        Упрощенная модель Keras
    """
    inputs = layers.Input(shape=input_shape)
    
    # Простая CNN архитектура
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Выходы
    regression_output = layers.Dense(15, activation='sigmoid', name='regression_output')(x)
    classification_output = layers.Dense(5, activation='softmax', name='classification_output')(x)
    
    model = Model(inputs=inputs, outputs=[regression_output, classification_output])
    
    model.compile(
        optimizer='adam',
        loss={
            'regression_output': 'mse',
            'classification_output': 'categorical_crossentropy'
        },
        metrics={
            'regression_output': ['mae'],
            'classification_output': ['accuracy']
        }
    )
    
    return model


if __name__ == "__main__":
    # Пример использования модели
    print("🧪 Тестирование модели TerraziteRecipeModel")
    print("=" * 50)
    
    # Создаем экземпляр модели
    model = TerraziteRecipeModel()
    
    # Строим модель
    model.build_model()
    
    # Показываем архитектуру
    print("\n📊 Архитектура модели:")
    model.summary()
    
    # Создаем тестовые данные
    print("\n🧪 Создание тестовых данных...")
    test_image = np.random.rand(1, 224, 224, 3).astype('float32')
    
    # Тестируем предсказание
    print("🧠 Тестирование предсказания на случайном изображении...")
    prediction = model.predict(test_image)
    
    print("\n📈 Результаты предсказания:")
    print(f"Тип заполнителя: {prediction['aggregate_type']}")
    print(f"Уверенность: {prediction['confidence']:.1f}%")
    print(f"Компонентов предсказано: {len(prediction['recipe_components'])}")
    
    print("\n✅ Модель готова к работе!")
