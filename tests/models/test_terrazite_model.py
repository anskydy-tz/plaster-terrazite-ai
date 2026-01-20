"""
Исправленные тесты для PyTorch модели TerraziteModel
"""
import sys
import os
import pytest
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.terrazite_model import TerraziteModel, create_model


class TestTerraziteModelPyTorch:
    """Тесты для PyTorch модели TerraziteModel"""
    
    @pytest.fixture
    def sample_image_batch(self):
        """Фикстура: батч тестовых изображений для PyTorch"""
        return torch.randn(4, 3, 224, 224)
    
    @pytest.fixture
    def sample_components(self):
        """Фикстура: тестовые компоненты"""
        return torch.randn(4, 100)
    
    def test_model_initialization(self):
        """Тест инициализации модели"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        )
        
        assert model.num_categories == 5
        assert model.num_components == 100
        assert model.hidden_size == 512
        assert model.dropout_rate == 0.3
        assert model.use_pretrained is True
        
        print(f"✅ Модель инициализирована")
        print(f"   Категорий: {model.num_categories}")
        print(f"   Компонентов: {model.num_components}")
    
    def test_model_forward_pass(self, sample_image_batch, sample_components):
        """Тест прямого прохода модели"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        )
        
        # Проверяем что модель создана
        assert model.image_encoder is not None
        assert model.component_encoder is not None
        assert model.multimodal_encoder is not None
        
        # Прямой проход с компонентами
        outputs_with_components = model(sample_image_batch, sample_components)
        
        assert 'category_logits' in outputs_with_components
        assert 'component_logits' in outputs_with_components
        assert 'recipe_features' in outputs_with_components
        assert 'multimodal_features' in outputs_with_components
        assert 'component_regression' in outputs_with_components
        
        # Проверяем размерности
        assert outputs_with_components['category_logits'].shape == (4, 5)
        assert outputs_with_components['component_logits'].shape == (4, 100)
        assert outputs_with_components['recipe_features'].shape == (4, 128)
        assert outputs_with_components['multimodal_features'].shape == (4, 512)
        assert outputs_with_components['component_regression'].shape == (4, 100)
        
        # Прямой проход без компонентов
        outputs_without_components = model(sample_image_batch, None)
        
        assert outputs_without_components['category_logits'].shape == (4, 5)
        
        print(f"✅ Прямой проход работает")
        print(f"   С компонентами: все выходы созданы")
        print(f"   Без компонентов: категории созданы")
    
    def test_model_predict_category(self, sample_image_batch, sample_components):
        """Тест предсказания категории"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        )
        
        # Предсказание с компонентами
        predicted_with, probs_with = model.predict_category(sample_image_batch, sample_components)
        
        assert predicted_with.shape == (4,)
        assert probs_with.shape == (4, 5)
        
        # Предсказание без компонентов
        predicted_without, probs_without = model.predict_category(sample_image_batch, None)
        
        assert predicted_without.shape == (4,)
        assert probs_without.shape == (4, 5)
        
        # Проверяем что вероятности суммируются к 1
        for i in range(4):
            assert torch.allclose(probs_with[i].sum(), torch.tensor(1.0), atol=1e-5)
            assert torch.allclose(probs_without[i].sum(), torch.tensor(1.0), atol=1e-5)
        
        print(f"✅ Предсказание категории работает")
        print(f"   Форма предсказаний: {predicted_with.shape}")
        print(f"   Форма вероятностей: {probs_with.shape}")
    
    def test_model_predict_components(self, sample_image_batch):
        """Тест предсказания компонентов"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        )
        
        predictions = model.predict_components(sample_image_batch, threshold=0.1)
        
        assert 'binary_predictions' in predictions
        assert 'probabilities' in predictions
        assert 'values' in predictions
        
        assert predictions['binary_predictions'].shape == (4, 100)
        assert predictions['probabilities'].shape == (4, 100)
        assert predictions['values'].shape == (4, 100)
        
        # Проверяем что бинарные предсказания действительно бинарные
        binary_vals = predictions['binary_predictions'].unique()
        assert torch.all((binary_vals == 0) | (binary_vals == 1))
        
        # Проверяем что вероятности в диапазоне [0, 1]
        assert torch.all(predictions['probabilities'] >= 0)
        assert torch.all(predictions['probabilities'] <= 1)
        
        print(f"✅ Предсказание компонентов работает")
        print(f"   Бинарные предсказания: {predictions['binary_predictions'].shape}")
        print(f"   Вероятности: {predictions['probabilities'].shape}")
    
    def test_model_get_info(self):
        """Тест получения информации о модели"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        )
        
        info = model.get_model_info()
        
        assert 'name' in info
        assert 'num_categories' in info
        assert 'num_components' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'component_groups' in info
        assert 'recipe_categories' in info
        
        assert info['name'] == 'TerraziteModel'
        assert info['num_categories'] == 5
        assert info['num_components'] == 100
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        
        print(f"✅ Информация о модели:")
        print(f"   Всего параметров: {info['total_parameters']:,}")
        print(f"   Обучаемых: {info['trainable_parameters']:,}")
    
    def test_create_model_factory(self):
        """Тест фабричной функции создания модели"""
        # Создание обычной модели
        model = create_model('terrazite', num_categories=5, num_components=50)
        
        assert isinstance(model, TerraziteModel)
        assert model.num_categories == 5
        assert model.num_components == 50
        
        print(f"✅ Фабричная функция работает для 'terrazite'")
    
    def test_model_on_cpu(self, sample_image_batch):
        """Тест работы модели на CPU"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        ).cpu()
        
        # Убеждаемся что модель на CPU
        assert next(model.parameters()).device.type == 'cpu'
        
        # Делаем предсказание
        outputs = model(sample_image_batch.cpu(), None)
        
        assert outputs['category_logits'].shape == (4, 5)
        assert outputs['category_logits'].device.type == 'cpu'
        
        print(f"✅ Модель работает на CPU")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self, sample_image_batch):
        """Тест работы модели на GPU (если доступен)"""
        model = TerraziteModel(
            num_categories=5,
            num_components=100,
            hidden_size=512
        ).cuda()
        
        # Убеждаемся что модель на GPU
        assert next(model.parameters()).device.type == 'cuda'
        
        # Делаем предсказание
        outputs = model(sample_image_batch.cuda(), None)
        
        assert outputs['category_logits'].shape == (4, 5)
        assert outputs['category_logits'].device.type == 'cuda'
        
        print(f"✅ Модель работает на GPU")


class TestMultiTaskLoss:
    """Тесты для мультизадачной функции потерь"""
    
    @pytest.fixture
    def sample_outputs(self):
        """Фикстура: тестовые выходы модели"""
        return {
            'category_logits': torch.randn(4, 5),
            'component_logits': torch.randn(4, 100),
            'component_regression': torch.randn(4, 100)
        }
    
    @pytest.fixture
    def sample_targets(self):
        """Фикстура: тестовые цели"""
        return {
            'category': torch.randint(0, 5, (4,)),
            'components_binary': torch.randint(0, 2, (4, 100)).float(),
            'components_values': torch.randn(4, 100)
        }
    
    def test_loss_initialization(self):
        """Тест инициализации функции потерь"""
        from src.models.terrazite_model import MultiTaskLoss
        
        loss_fn = MultiTaskLoss(
            category_weight=1.0,
            component_weight=0.5,
            regression_weight=0.3
        )
        
        assert loss_fn.category_weight == 1.0
        assert loss_fn.component_weight == 0.5
        assert loss_fn.regression_weight == 0.3
        
        assert isinstance(loss_fn.category_loss, torch.nn.CrossEntropyLoss)
        assert isinstance(loss_fn.component_loss, torch.nn.BCEWithLogitsLoss)
        assert isinstance(loss_fn.regression_loss, torch.nn.MSELoss)
        
        print(f"✅ Функция потерь инициализирована")
        print(f"   Веса: category={loss_fn.category_weight}, "
              f"component={loss_fn.component_weight}, regression={loss_fn.regression_weight}")
    
    def test_loss_computation(self, sample_outputs, sample_targets):
        """Тест вычисления потерь"""
        from src.models.terrazite_model import MultiTaskLoss
        
        loss_fn = MultiTaskLoss(
            category_weight=1.0,
            component_weight=0.5,
            regression_weight=0.3
        )
        
        losses = loss_fn(sample_outputs, sample_targets)
        
        assert 'category' in losses
        assert 'component' in losses
        assert 'regression' in losses
        assert 'total' in losses
        
        assert isinstance(losses['category'], torch.Tensor)
        assert isinstance(losses['component'], torch.Tensor)
        assert isinstance(losses['regression'], torch.Tensor)
        assert isinstance(losses['total'], torch.Tensor)
        
        # Проверяем что общая потеря - сумма взвешенных потерь
        expected_total = (losses['category'] + losses['component'] + losses['regression'])
        assert torch.allclose(losses['total'], expected_total)
        
        print(f"✅ Потери вычислены:")
        print(f"   Категория: {losses['category'].item():.4f}")
        print(f"   Компоненты: {losses['component'].item():.4f}")
        print(f"   Регрессия: {losses['regression'].item():.4f}")
        print(f"   Всего: {losses['total'].item():.4f}")


def test_integration():
    """Интеграционный тест полного цикла модели"""
    print("\n" + "="*60)
    print("ИНТЕГРАЦИОННЫЙ ТЕСТ ПОЛНОГО ЦИКЛА")
    print("="*60)
    
    # Создаем модель
    model = TerraziteModel(
        num_categories=5,
        num_components=50,
        hidden_size=256
    )
    
    # Создаем тестовые данные
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    components = torch.randn(batch_size, 50)
    
    # Прямой проход
    outputs = model(images, components)
    
    # Проверяем выходы
    assert outputs['category_logits'].shape == (batch_size, 5)
    assert outputs['component_logits'].shape == (batch_size, 50)
    
    # Предсказание категории
    predicted, probs = model.predict_category(images, components)
    assert predicted.shape == (batch_size,)
    assert probs.shape == (batch_size, 5)
    
    # Предсказание компонентов
    component_preds = model.predict_components(images, threshold=0.1)
    assert component_preds['binary_predictions'].shape == (batch_size, 50)
    
    # Информация о модели
    info = model.get_model_info()
    assert info['total_parameters'] > 1000
    
    print(f"\n✅ Интеграционный тест пройден:")
    print(f"   Создана модель: {info['name']}")
    print(f"   Параметров: {info['total_parameters']:,}")
    print(f"   Тестовый батч: {batch_size} изображений")
    print(f"   Категории: {info['num_categories']}")
    print(f"   Компоненты: {info['num_components']}")


if __name__ == "__main__":
    # Запуск тестов напрямую
    pytest.main([__file__, "-v"])
