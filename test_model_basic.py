"""
Базовое тестирование модели Terrazite AI
"""
import sys
sys.path.append('src')

import torch
import json
from src.utils.config import config
from src.models.terrazite_model import TerraziteModel

print('='*80)
print('БАЗОВОЕ ТЕСТИРОВАНИЕ МОДЕЛИ TERRAZITE AI')
print('='*80)

try:
    # 1. Проверка конфигурации
    print('1. ПРОВЕРКА КОНФИГУРАЦИИ:')
    print(f'   Категорий: {config.model.num_categories}')
    print(f'   Компонентов: {config.model.num_components}')
    print(f'   Batch size: {config.model.batch_size}')
    print(f'   Learning rate: {config.model.learning_rate}')
    print('   ✅ Конфигурация загружена')
    
    # 2. Проверка маппинга компонентов
    print('\n2. ПРОВЕРКА МАППИНГА КОМПОНЕНТОВ:')
    try:
        with open('data/processed/component_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        actual_components = len(mapping)
        print(f'   Фактическое количество компонентов: {actual_components}')
        print(f'   Конфигурация ожидает: {config.model.num_components}')
        
        if actual_components == config.model.num_components:
            print('   ✅ Маппинг соответствует конфигурации')
        else:
            print(f'   ⚠️  Несоответствие: {actual_components} vs {config.model.num_components}')
            config.model.num_components = actual_components
            print('   🔧 Исправлено в конфигурации')
    
    except Exception as e:
        print(f'   ❌ Ошибка загрузки маппинга: {e}')
    
    # 3. Создание модели
    print('\n3. СОЗДАНИЕ МОДЕЛИ:')
    model = TerraziteModel(
        num_categories=config.model.num_categories,
        num_components=config.model.num_components,
        use_pretrained=False
    )
    
    print(f'   Модель создана: TerraziteModel')
    print(f'   Категории: {config.model.num_categories}')
    print(f'   Компоненты: {config.model.num_components}')
    
    # 4. Проверка параметров модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'   Всего параметров: {total_params:,}')
    print(f'   Обучаемых параметров: {trainable_params:,}')
    
    # 5. Тестовый прогон
    print('\n4. ТЕСТОВЫЙ ПРОГОН:')
    batch_size = 2
    print(f'   Batch size: {batch_size}')
    
    # Тестовые данные
    images = torch.randn(batch_size, 3, 224, 224)
    components = torch.randn(batch_size, config.model.num_components)
    
    # Прямой проход
    model.eval()
    with torch.no_grad():
        outputs = model(images, components)
    
    print(f'   Входные изображения: {images.shape}')
    print(f'   Входные компоненты: {components.shape}')
    print(f'   Выход категорий: {outputs["category_logits"].shape}')
    print(f'   Выход компонентов: {outputs["component_logits"].shape}')
    print(f'   Регрессия компонентов: {outputs["component_regression"].shape}')
    
    # 6. Предсказание категории
    print('\n5. ПРЕДСКАЗАНИЕ КАТЕГОРИИ:')
    predicted, probs = model.predict_category(images)
    print(f'   Предсказанные категории: {predicted.tolist()}')
    print(f'   Форма вероятностей: {probs.shape}')
    
    # 7. Информация о модели
    print('\n6. ИНФОРМАЦИЯ О МОДЕЛИ:')
    info = model.get_model_info()
    print(f'   Название: {info["name"]}')
    print(f'   Группы компонентов: {len(info["component_groups"])}')
    print(f'   Категории рецептов: {info["recipe_categories"]}')
    print(f'   Маппинг загружен: {info["component_mapping_loaded"]}')
    print(f'   Примечание: {info["note"]}')
    
    print('\n' + '='*80)
    print('✅ БАЗОВОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!')
    print('='*80)
    
    print('\n🎯 СЛЕДУЮЩИЕ ШАГИ:')
    print('1. Запустить обучение: python scripts/train_model.py')
    print('2. Протестировать на реальных данных: python scripts/test_real_data.py')
    print('3. Запустить веб-интерфейс: streamlit run streamlit_app.py')
    
except Exception as e:
    print(f'\n❌ ОШИБКА ПРИ ТЕСТИРОВАНИИ: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
