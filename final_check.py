import sys
sys.path.append('src')
import json
import torch
from utils.config import config
from data.loader import RecipeLoader
from models.terrazite_model import TerraziteModel

print('='*60)
print('ФИНАЛЬНАЯ ПРОВЕРКА ВСЕГО ПРОЕКТА')
print('='*60)

checks = []

# 1. Проверка конфигурации
try:
    config_check = f'✓ Категории: {config.model.num_categories}, Компоненты: {config.model.num_components}'
    checks.append(('Конфигурация', config_check))
    print(f'Конфигурация: Категории={config.model.num_categories}, Компоненты={config.model.num_components}')
except Exception as e:
    checks.append(('Конфигурация', f'✗ Ошибка: {e}'))
    print(f'Ошибка конфигурации: {e}')

# 2. Проверка маппинга
try:
    with open('data/processed/component_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    actual_components = len(mapping)
    water_found = any('вода' in comp.lower() for comp in mapping.values())
    
    if water_found:
        checks.append(('Маппинг', f'✗ Обнаружены компоненты с водой (всего: {actual_components})'))
    else:
        checks.append(('Маппинг', f'✓ {actual_components} компонентов (вода исключена)'))
    
    print(f'Маппинг: {actual_components} компонентов')
    
    # Проверяем соответствие с конфигурацией
    if config.model.num_components != actual_components:
        print(f'ВНИМАНИЕ: В конфигурации {config.model.num_components} компонентов, в маппинге {actual_components}')
        
except Exception as e:
    checks.append(('Маппинг', f'✗ Ошибка: {e}'))
    print(f'Ошибка маппинга: {e}')

# 3. Проверка загрузки данных
try:
    loader = RecipeLoader('data/raw/recipes.xlsx')
    df = loader.load_excel()
    checks.append(('Загрузка данных', f'✓ {len(df)} рецептов'))
    print(f'Загружено рецептов: {len(df)}')
except Exception as e:
    checks.append(('Загрузка данных', f'✗ Ошибка: {e}'))
    print(f'Ошибка загрузки данных: {e}')

# 4. Проверка модели
try:
    model = TerraziteModel(
        num_categories=config.model.num_categories,
        num_components=config.model.num_components,
        use_pretrained=False
    )
    checks.append(('Модель', f'✓ Создана'))
    
    # Проверка прямого прохода
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    components = torch.randn(batch_size, config.model.num_components)
    
    outputs = model(images, components)
    print(f'Модель: создана успешно, прямой проход работает')
    print(f'  - Выход категорий: {outputs["category_logits"].shape}')
    print(f'  - Выход компонентов: {outputs["component_logits"].shape}')
    
except Exception as e:
    checks.append(('Модель', f'✗ Ошибка: {e}'))
    print(f'Ошибка модели: {e}')
    import traceback
    traceback.print_exc()

# Вывод результатов
print('\n' + '='*60)
print('РЕЗУЛЬТАТЫ ПРОВЕРКИ:')
print('='*60)
for check_name, result in checks:
    print(f'  {check_name:20} {result}')

# Подсчет успешных проверок
success_count = sum(1 for _, result in checks if result.startswith('✓'))
total_count = len(checks)

print('\n' + '='*60)
if success_count == total_count:
    print(f'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ({success_count}/{total_count})')
    print('Проект готов к работе!')
else:
    print(f'⚠  ПРОЙДЕНО ПРОВЕРОК: {success_count}/{total_count}')
    print('Требуется исправление ошибок.')
print('='*60)
