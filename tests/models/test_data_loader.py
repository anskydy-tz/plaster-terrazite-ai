"""
Тесты для загрузчиков данных
"""
import sys
import os
import pytest
import numpy as np
import json
import tempfile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loader import DataLoader, RecipeLoader
from src.data.processor import DataProcessor


class TestDataLoader:
    """Тесты для DataLoader"""
    
    def test_load_image(self):
        """Тест загрузки изображения"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Создаем тестовое изображение
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(tmp.name)
            
            # Загружаем через DataLoader
            loaded_img = DataLoader.load_image(tmp.name, target_size=(224, 224))
            
            assert loaded_img is not None
            assert loaded_img.shape == (224, 224, 3)
            assert loaded_img.dtype == np.float32
            assert 0 <= loaded_img.min() <= 1
            assert 0 <= loaded_img.max() <= 1
            
            # Удаляем временный файл
            os.unlink(tmp.name)
    
    def test_load_recipe_json(self):
        """Тест загрузки JSON с рецептами"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            # Создаем тестовый JSON
            test_recipes = [
                {
                    "recipe_id": "TEST_001",
                    "name": "Тестовый рецепт",
                    "white_cement": 100.0
                },
                {
                    "recipe_id": "TEST_002",
                    "name": "Еще один рецепт",
                    "gray_cement": 50.0
                }
            ]
            json.dump(test_recipes, tmp)
            tmp.flush()
            
            # Загружаем через DataLoader
            recipes = DataLoader.load_recipe_json(tmp.name)
            
            assert len(recipes) == 2
            assert recipes[0]['recipe_id'] == 'TEST_001'
            assert recipes[1]['recipe_id'] == 'TEST_002'
            
            # Удаляем временный файл
            os.unlink(tmp.name)


class TestRecipeLoader:
    """Тесты для RecipeLoader"""
    
    def test_parse_float(self):
        """Тест парсинга значений в float"""
        # Тест с числом
        assert RecipeLoader._parse_float(100.5) == 100.5
        assert RecipeLoader._parse_float(100) == 100.0
        
        # Тест со строкой
        assert RecipeLoader._parse_float("150.5") == 150.5
        
        # Тест с NaN
        assert RecipeLoader._parse_float(float('nan')) == 0.0
        
        # Тест с None
        assert RecipeLoader._parse_float(None) == 0.0


class TestDataProcessor:
    """Тесты для DataProcessor"""
    
    @pytest.fixture
    def sample_recipes(self):
        """Фикстура: тестовые рецепты"""
        return [
            {
                'recipe_id': 'REC_001',
                'white_cement': 100.0,
                'gray_cement': 50.0,
                'lime': 0.0,
                'sand_0_063mm': 300.0,
                'marble_white_30_50mm': 200.0,
                'aggregate_type': 'marble'
            },
            {
                'recipe_id': 'REC_002',
                'white_cement': 150.0,
                'gray_cement': 0.0,
                'lime': 25.0,
                'sand_0_063mm': 400.0,
                'quartz_10_30mm': 180.0,
                'aggregate_type': 'quartz'
            }
        ]
    
    def test_prepare_recipe_features(self, sample_recipes):
        """Тест подготовки признаков из рецептов"""
        processor = DataProcessor()
        features, valid_recipes = processor.prepare_recipe_features(sample_recipes)
        
        assert features.shape[0] == 2  # Два рецепта
        assert features.shape[1] > 10  # Много признаков
        assert len(valid_recipes) == 2
        
        # Проверяем что значения нормализованы (делены на 1000)
        assert features[0, 0] == 0.1  # 100 / 1000
    
    def test_prepare_targets(self, sample_recipes):
        """Тест подготовки целевых переменных"""
        processor = DataProcessor()
        features, valid_recipes = processor.prepare_recipe_features(sample_recipes)
        targets = processor.prepare_targets(valid_recipes)
        
        assert 'regression' in targets
        assert 'classification' in targets
        assert 'class_names' in targets
        
        assert targets['regression'].shape[0] == 2
        assert len(targets['classification']) == 2
        
        # Проверяем что метки классов закодированы
        assert set(targets['classification']) <= {0, 1}
    
    def test_split_dataset(self, sample_recipes):
        """Тест разделения датасета"""
        processor = DataProcessor()
        features, valid_recipes = processor.prepare_recipe_features(sample_recipes)
        targets = processor.prepare_targets(valid_recipes)
        
        dataset = processor.split_dataset(features, targets, test_size=0.5, val_size=0.5)
        
        # Проверяем что данные разделены
        assert 'X_train' in dataset
        assert 'X_val' in dataset
        assert 'X_test' in dataset
        
        # Проверяем размеры (с учетом маленькой выборки)
        assert dataset['X_train'].shape[0] == 1
        assert dataset['X_val'].shape[0] == 0  # При таком разделении val может быть 0
        assert dataset['X_test'].shape[0] == 1
        
        # Проверяем что признаки нормализованы
        assert np.allclose(dataset['X_train'].mean(), 0, atol=0.1)
        assert np.allclose(dataset['X_train'].std(), 1, atol=0.5)
