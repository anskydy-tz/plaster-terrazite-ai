# Создаем файл final_check.py
echo "import sys
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
except Exception as e:
    checks.append(('Конфигурация', f'✗ Ошибка: {e}'))

# 2. Проверка маппинга
try:
    with open('data/processed/component_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    water_found = any('вода' in comp.lower() for comp in mapping.values())
    if water_found:
        checks.append(('Маппинг', '✗ Обнаружены компоненты с водой'))
    else:
        checks.append(('Маппинг', f'✓ {len(mapping)} компонентов'))
except Exception as e:
    checks.append(('Маппинг', f'✗ Ошибка: {e}'))

# 3. Проверка загрузки данных
try:
    loader = RecipeLoader('data/raw/recipes.xlsx')
    df = loader.load_excel()
    checks.append(('Загрузка данных', f'✓ {len(df)} рецептов'))
except Exception as e:
    checks.append(('Загрузка данных', f'✗ Ошибка: {e}'))

# 4. Проверка модели
try:
    model = TerraziteModel(
        num_categories=config.model.num_categories,
        num_components=config.model.num_components,
        use_pretrained=False
    )
    checks.append(('Модель', f'✓ Создана'))
except Exception as e:
    checks.append(('Модель', f'✗ Ошибка: {e}'))

# Вывод результатов
print('РЕЗУЛЬТАТЫ ПРОВЕРКИ:')
for check_name, result in checks:
    print(f'  {check_name:20} {result}')

# Подсчет успешных проверок
success_count = sum(1 for _, result in checks if result.startswith('✓'))
total_count = len(checks)

print()
print('='*60)
if success_count == total_count:
    print(f'✓ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ({success_count}/{total_count})')
    print('Проект готов к работе!')
else:
    print(f'⚠ ПРОЙДЕНО ПРОВЕРОК: {success_count}/{total_count}')
    print('Требуется исправление ошибок.')
print('='*60)" > final_check.py
