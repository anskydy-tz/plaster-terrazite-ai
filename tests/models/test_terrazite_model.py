"""
Тесты для основной модели TerraziteRecipeModel
"""
import sys
import os
import pytest
import numpy as np
import tensorflow as tf

# Добавляем путь к корню проекта для импорта модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.terrazite_model import TerraziteRecipeModel, create_simple_model
from src.models.trainer import ModelTrainer


class TestTerraziteModel:
    """Тесты для класса TerraziteRecipeModel"""
    
    @pytest.fixture
    def sample_image(self):
        """Фикстура: тестовое изображение"""
        return np.random.rand(224, 224, 3).astype('float32')
    
    @pytest.fixture
    def sample_batch_images(self):
        """Фикстура: батч тестовых изображений"""
        return np.random.rand(10, 224, 224, 3).astype('float32')
    
    @pytest.fixture
    def sample_labels(self):
        """Фикстура: тестовые метки"""
        return {
            'regression_output': np.random.rand(10, 15).astype('float32'),
            'classification_output': tf.keras.utils.to_categorical(
                np.random.randint(0, 5, 10), 5
            )
        }
    
    def test_model_initialization(self):
        """Тест инициализации модели"""
        # Тестирование с параметрами по умолчанию
        model = TerraziteRecipeModel()
        assert model.input_shape == (224, 224, 3)
        assert model.num_regression_outputs == 15
        assert model.num_classes == 5
        assert model.dropout_rate == 0.3
        assert model.learning_rate == 0.001
        assert model.model is None
        assert model.history is None
        
        # Тестирование с пользовательскими параметрами
        custom_model = TerraziteRecipeModel(
            input_shape=(128, 128, 3),
            num_regression_outputs=20,
            num_classes=3,
            dropout_rate=0.5,
            learning_rate=0.0001
        )
        assert custom_model.input_shape == (128, 128, 3)
        assert custom_model.num_regression_outputs == 20
        assert custom_model.num_classes == 3
        assert custom_model.dropout_rate == 0.5
        assert custom_model.learning_rate == 0.0001
    
    def test_model_building(self):
        """Тест построения модели"""
        model = TerraziteRecipeModel()
        built_model = model.build_model()
        
        # Проверяем что модель создана
        assert built_model is not None
        assert model.model is not None
        
        # Проверяем входной слой
        assert built_model.input_shape == (None, 224, 224, 3)
        
        # Проверяем выходные слои
        assert len(built_model.outputs) == 2
        assert 'regression_output' in [output.name for output in built_model.outputs]
        assert 'classification_output' in [output.name for output in built_model.outputs]
        
        # Проверяем количество параметров
        total_params = built_model.count_params()
        assert total_params > 1000  # Модель должна иметь разумное количество параметров
        
        print(f"✅ Модель построена. Параметров: {total_params:,}")
    
    def test_model_prediction(self, sample_image):
        """Тест предсказания модели"""
        model = TerraziteRecipeModel()
        model.build_model()
        
        # Тестирование на одном изображении
        prediction = model.predict(sample_image)
        
        # Проверяем структуру ответа
        assert isinstance(prediction, dict)
        assert 'recipe_components' in prediction
        assert 'aggregate_type' in prediction
        assert 'confidence' in prediction
        assert 'aggregate_probabilities' in prediction
        
        # Проверяем типы данных
        assert isinstance(prediction['recipe_components'], list)
        assert isinstance(prediction['aggregate_type'], str)
        assert isinstance(prediction['confidence'], float)
        assert isinstance(prediction['aggregate_probabilities'], list)
        
        # Проверяем размерности
        assert len(prediction['recipe_components']) == 15
        assert len(prediction['aggregate_probabilities']) == 5
        
        # Проверяем что уверенность в пределах [0, 100]
        assert 0 <= prediction['confidence'] <= 100
        
        # Проверяем что сумма вероятностей ≈ 1
        probs_sum = sum(prediction['aggregate_probabilities'])
        assert abs(probs_sum - 1.0) < 0.01
    
    def test_model_training_structure(self, sample_batch_images, sample_labels):
        """Тест структуры обучения модели"""
        model = TerraziteRecipeModel()
        model.build_model()
        
        # Проверяем что модель скомпилирована
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        assert model.model.metrics is not None
        
        # Тестируем один шаг обучения
        with tf.device('/CPU:0'):  # Используем CPU для тестов
            history = model.model.fit(
                sample_batch_images,
                sample_labels,
                epochs=1,
                batch_size=2,
                verbose=0
            )
        
        # Проверяем что история обучения содержит метрики
        assert 'loss' in history.history
        assert 'regression_output_loss' in history.history
        assert 'classification_output_loss' in history.history
    
    def test_model_save_load(self, tmp_path, sample_image):
        """Тест сохранения и загрузки модели"""
        model = TerraziteRecipeModel()
        model.build_model()
        
        # Сохраняем модель
        save_path = tmp_path / "test_model.h5"
        model.save_model(str(save_path))
        
        # Проверяем что файл создан
        assert save_path.exists()
        assert save_path.stat().st_size > 1000  # Файл не пустой
        
        # Создаем новую модель и загружаем
        loaded_model = TerraziteRecipeModel()
        loaded_model.load_model(str(save_path))
        
        # Проверяем что модель загружена
        assert loaded_model.model is not None
        
        # Сравниваем предсказания
        original_pred = model.predict(sample_image)
        loaded_pred = loaded_model.predict(sample_image)
        
        # Предсказания должны быть одинаковыми (с учетом погрешности float)
        assert len(original_pred['recipe_components']) == len(loaded_pred['recipe_components'])
        
        print("✅ Модель успешно сохранена и загружена")
    
    def test_simple_model_creation(self):
        """Тест создания упрощенной модели"""
        simple_model = create_simple_model()
        
        # Проверяем архитектуру
        assert simple_model.input_shape == (None, 224, 224, 3)
        assert len(simple_model.outputs) == 2
        
        # Проверяем слой за слоем
        layer_types = [layer.__class__.__name__ for layer in simple_model.layers]
        assert 'InputLayer' in layer_types
        assert 'Conv2D' in layer_types
        assert 'Dense' in layer_types
        
        # Проверяем что модель может делать предсказания
        test_input = np.random.rand(1, 224, 224, 3).astype('float32')
        predictions = simple_model.predict(test_input, verbose=0)
        
        assert len(predictions) == 2
        assert predictions[0].shape == (1, 15)  # Регрессия
        assert predictions[1].shape == (1, 5)   # Классификация


class TestModelTrainer:
    """Тесты для класса ModelTrainer"""
    
    @pytest.fixture
    def sample_dataset(self):
        """Фикстура: тестовый датасет"""
        X_train = np.random.rand(20, 224, 224, 3).astype('float32')
        X_val = np.random.rand(5, 224, 224, 3).astype('float32')
        
        y_train_reg = np.random.rand(20, 15).astype('float32')
        y_val_reg = np.random.rand(5, 15).astype('float32')
        
        y_train_cls = tf.keras.utils.to_categorical(
            np.random.randint(0, 5, 20), 5
        )
        y_val_cls = tf.keras.utils.to_categorical(
            np.random.randint(0, 5, 5), 5
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_reg_train': y_train_reg,
            'y_reg_val': y_val_reg,
            'y_cls_train': y_train_cls,
            'y_cls_val': y_val_cls
        }
    
    def test_trainer_initialization(self):
        """Тест инициализации тренера"""
        trainer = ModelTrainer()
        
        assert trainer.model is None
        assert trainer.history is None
        assert trainer.callbacks is not None
    
    def test_model_creation(self):
        """Тест создания модели через тренер"""
        trainer = ModelTrainer()
        model = trainer.create_model()
        
        assert model is not None
        assert trainer.model is not None
        
        # Проверяем архитектуру
        assert model.input_shape == (None, 224, 224, 3)
        assert len(model.outputs) == 2
    
    def test_training_process(self, sample_dataset):
        """Тест процесса обучения"""
        trainer = ModelTrainer()
        
        # Создаем модель
        trainer.create_model()
        
        # Обучаем на 1 эпоху
        history = trainer.train(
            train_data=(
                sample_dataset['X_train'],
                {
                    'regression_output': sample_dataset['y_reg_train'],
                    'classification_output': sample_dataset['y_cls_train']
                }
            ),
            val_data=(
                sample_dataset['X_val'],
                {
                    'regression_output': sample_dataset['y_reg_val'],
                    'classification_output': sample_dataset['y_cls_val']
                }
            ),
            epochs=1,
            batch_size=4
        )
        
        # Проверяем что обучение прошло
        assert history is not None
        assert trainer.history is not None
        
        # Проверяем метрики
        assert 'loss' in history
        assert 'regression_output_loss' in history
        assert 'classification_output_loss' in history


def test_model_integration():
    """Интеграционный тест полного цикла модели"""
    # Создаем тестовые данные
    images = np.random.rand(5, 224, 224, 3).astype('float32')
    
    # Создаем и обучаем модель
    model = TerraziteRecipeModel()
    model.build_model()
    
    # Тестируем батч предсказаний
    for i in range(images.shape[0]):
        prediction = model.predict(images[i])
        
        # Проверяем валидность предсказания
        assert prediction['aggregate_type'] in ['мрамор', 'кварц', 'гранит', 'слюда', 'известняк', 'неизвестно']
        assert all(0 <= p <= 100 for p in prediction['recipe_components'])
    
    print("✅ Интеграционный тест пройден")


if __name__ == "__main__":
    # Запуск тестов напрямую
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
