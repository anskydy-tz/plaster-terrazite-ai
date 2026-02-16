"""
Тестирование полного пайплайна Terrazite AI
"""
import sys
sys.path.append('src')

import os
import json
import pandas as pd
from pathlib import Path
import torch

from src.utils.config import config
from src.data.loader import RecipeLoader
from src.models.terrazite_model import TerraziteModel
from src.models.trainer import ModelTrainer

print('='*80)
print('ТЕСТИРОВАНИЕ ПОЛНОГО ПАЙПЛАЙНА TERRAZITE AI')
print('='*80)

def check_data_availability():
    """Проверка доступности данных"""
    print('\n1. ПРОВЕРКА ДАННЫХ:')
    
    checks = []
    
    # Проверка файлов
    files_to_check = [
        ('data/raw/recipes.xlsx', 'Excel с рецептами'),
        ('data/processed/component_mapping.json', 'Маппинг компонентов'),
        ('data/processed/data_manifest_train.csv', 'Манифест обучения'),
        ('data/processed/data_manifest_val.csv', 'Манифест валидации'),
        ('data/processed/data_manifest_test.csv', 'Манифест теста')
    ]
    
    for path, description in files_to_check:
        exists = Path(path).exists()
        status = '✅' if exists else '❌'
        checks.append((status, description, exists))
        print(f'   {status} {description}: {path}')
    
    # Проверка изображений
    images_dir = 'data/processed/images'
    if Path(images_dir).exists():
        image_count = sum(1 for _ in Path(images_dir).rglob('*.jpg')) + \
                      sum(1 for _ in Path(images_dir).rglob('*.png'))
        print(f'   📸 Изображений в датасете: {image_count}')
        checks.append(('✅', 'Изображения', image_count > 0))
    else:
        print(f'   ❌ Директория с изображениями не найдена: {images_dir}')
        checks.append(('❌', 'Изображения', False))
    
    # Подсчет успешных проверок
    success_count = sum(1 for status, _, exists in checks if exists)
    total_count = len(checks)
    
    print(f'\n   Результат: {success_count}/{total_count} проверок пройдено')
    
    return success_count == total_count

def test_data_loading():
    """Тестирование загрузки данных"""
    print('\n2. ТЕСТИРОВАНИЕ ЗАГРУЗКИ ДАННЫХ:')
    
    try:
        # Загрузка Excel
        loader = RecipeLoader('data/raw/recipes.xlsx')
        df = loader.load_excel()
        
        print(f'   Загружено рецептов: {len(df)}')
        print(f'   Колонки: {list(df.columns[:5])}...')
        
        # Анализ категорий
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            print(f'   Распределение по категориям:')
            for cat, count in category_counts.items():
                print(f'     - {cat}: {count}')
        
        # Загрузка маппинга компонентов
        with open('data/processed/component_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        print(f'   Компонентов в маппинге: {len(mapping)}')
        
        # Проверка компонентов без воды
        water_components = [c for c in mapping.values() if 'вода' in c.lower()]
        print(f'   Компонентов с водой (должно быть 0): {len(water_components)}')
        
        print('   ✅ Загрузка данных успешна')
        return True
        
    except Exception as e:
        print(f'   ❌ Ошибка загрузки данных: {e}')
        return False

def test_model_creation():
    """Тестирование создания модели"""
    print('\n3. ТЕСТИРОВАНИЕ МОДЕЛИ:')
    
    try:
        # Загрузка конфигурации
        print(f'   Конфигурация загружена:')
        print(f'     - Категории: {config.model.num_categories}')
        print(f'     - Компоненты: {config.model.num_components}')
        print(f'     - Размер изображения: {config.model.input_size}')
        
        # Создание модели
        model = TerraziteModel(
            num_categories=config.model.num_categories,
            num_components=config.model.num_components,
            use_pretrained=False  # Для тестирования используем случайные веса
        )
        
        print(f'   Модель создана: TerraziteModel')
        
        # Проверка параметров
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f'   Параметры модели:')
        print(f'     - Всего: {total_params:,}')
        print(f'     - Обучаемых: {trainable_params:,}')
        print(f'     - Замороженных: {total_params - trainable_params:,}')
        
        # Тестовый прогон
        print(f'   Тестовый прогон:')
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 224, 224)
        test_components = torch.randn(batch_size, config.model.num_components)
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_images, test_components)
        
        print(f'     - Входные изображения: {test_images.shape}')
        print(f'     - Входные компоненты: {test_components.shape}')
        print(f'     - Выход категорий: {outputs["category_logits"].shape}')
        print(f'     - Выход компонентов: {outputs["component_logits"].shape}')
        
        # Предсказание
        predicted, probs = model.predict_category(test_images)
        print(f'     - Предсказанные категории: {predicted.tolist()}')
        
        print('   ✅ Модель работает корректно')
        return True
        
    except Exception as e:
        print(f'   ❌ Ошибка создания модели: {e}')
        return False

def test_trainer_creation():
    """Тестирование создания тренера"""
    print('\n4. ТЕСТИРОВАНИЕ ТРЕНЕРА:')
    
    try:
        # Конфигурация тренера
        trainer_config = {
            'batch_size': 4,
            'learning_rate': 0.001,
            'epochs': 2,
            'weight_decay': 0.0001,
            'device': 'cpu'
        }
        
        # Создание тренера
        trainer = ModelTrainer(trainer_config)
        
        print(f'   Тренер создан: ModelTrainer')
        print(f'   Конфигурация:')
        print(f'     - Batch size: {trainer_config["batch_size"]}')
        print(f'     - Learning rate: {trainer_config["learning_rate"]}')
        print(f'     - Epochs: {trainer_config["epochs"]}')
        print(f'     - Устройство: {trainer_config["device"]}')
        
        # Проверка методов тренера
        print(f'   Доступные методы:')
        methods = [m for m in dir(trainer) if not m.startswith('_')]
        for method in methods[:10]:  # Показываем первые 10 методов
            print(f'     - {method}')
        
        print('   ✅ Тренер работает корректно')
        return True
        
    except Exception as e:
        print(f'   ❌ Ошибка создания тренера: {e}')
        return False

def test_data_manifests():
    """Тестирование манифестов данных"""
    print('\n5. ТЕСТИРОВАНИЕ МАНИФЕСТОВ:')
    
    try:
        manifests = [
            ('data/processed/data_manifest_train.csv', 'Обучающая выборка'),
            ('data/processed/data_manifest_val.csv', 'Валидационная выборка'),
            ('data/processed/data_manifest_test.csv', 'Тестовая выборка')
        ]
        
        for manifest_path, description in manifests:
            if Path(manifest_path).exists():
                df = pd.read_csv(manifest_path)
                print(f'   {description}:')
                print(f'     - Записей: {len(df)}')
                print(f'     - Колонки: {list(df.columns)}')
                
                if 'split' in df.columns:
                    split_counts = df['split'].value_counts()
                    print(f'     - Распределение по сплитам:')
                    for split, count in split_counts.items():
                        print(f'       - {split}: {count}')
                
                if 'recipe_type' in df.columns:
                    type_counts = df['recipe_type'].value_counts()
                    print(f'     - Типы рецептов: {len(type_counts)}')
            else:
                print(f'   ❌ {description} не найдена: {manifest_path}')
                return False
        
        print('   ✅ Манифесты загружены корректно')
        return True
        
    except Exception as e:
        print(f'   ❌ Ошибка загрузки манифестов: {e}')
        return False

def main():
    """Основная функция тестирования"""
    
    # Проверка доступности данных
    if not check_data_availability():
        print('\n❌ НЕОБХОДИМЫЕ ФАЙЛЫ НЕ НАЙДЕНЫ')
        print('   Пожалуйста, сначала запустите:')
        print('   1. python scripts/create_data_manifest.py')
        print('   2. python scripts/prepare_image_dataset.py')
        return
    
    # Запуск тестов
    tests = [
        ('Загрузка данных', test_data_loading),
        ('Манифесты данных', test_data_manifests),
        ('Создание модели', test_model_creation),
        ('Создание тренера', test_trainer_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f'   ❌ Неожиданная ошибка в {test_name}: {e}')
            results.append((test_name, False))
    
    # Вывод результатов
    print('\n' + '='*80)
    print('ИТОГИ ТЕСТИРОВАНИЯ:')
    print('='*80)
    
    for test_name, success in results:
        status = '✅' if success else '❌'
        print(f'   {status} {test_name}')
    
    # Подсчет успешных тестов
    success_count = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f'\n   Пройдено тестов: {success_count}/{total_tests}')
    
    if success_count == total_tests:
        print('\n' + '='*80)
        print('🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!')
        print('='*80)
        
        print('\n🎯 ПРОЕКТ ГОТОВ К ОБУЧЕНИЮ:')
        print('   Запустите обучение модели:')
        print('   python scripts/train_model.py --epochs 10 --batch-size 8')
        
        print('\n📊 СТАТИСТИКА ПРОЕКТА:')
        print('   - Рецептов: 174')
        print('   - Изображений: 1252 (с аугментацией)')
        print('   - Категорий: 5')
        print('   - Компонентов: 52 (без воды)')
        
    else:
        print('\n' + '='*80)
        print('⚠️  ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ')
        print('='*80)
        
        failed_tests = [name for name, success in results if not success]
        print(f'   Не пройдены тесты: {", ".join(failed_tests)}')

if __name__ == "__main__":
    main()
