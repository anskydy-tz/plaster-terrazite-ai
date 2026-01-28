# Создаем файл create_component_mapping_corrected.py
echo "import json
import pandas as pd
from pathlib import Path

def create_component_mapping():
    # Загружаем recipes.json
    json_path = Path('data/processed/recipes.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    
    print(f'Загружено рецептов: {len(recipes)}')
    
    # Собираем все уникальные компоненты, исключая служебные поля и воду
    all_components = set()
    service_fields = ['название', 'тип', 'id', 'name', 'type', 'recipe_id', 'recipe_name']
    
    for recipe in recipes:
        if 'components' in recipe:
            for component_name, component_amount in recipe['components'].items():
                # Проверяем, не является ли поле служебным
                component_name_clean = component_name.strip()
                component_name_lower = component_name_clean.lower()
                
                # Пропускаем служебные поля
                skip = False
                for service_field in service_fields:
                    if service_field in component_name_lower:
                        skip = True
                        break
                
                # ЯВНО исключаем компоненты с водой
                if 'вода' in component_name_lower or 'воды' in component_name_lower:
                    print(f'Исключен компонент с водой: {component_name_clean}')
                    skip = True
                
                if not skip and component_name_clean and component_name_clean != '0':
                    # Также проверяем, не является ли значение нулевым (пустым компонентом)
                    if component_amount != 0 and component_amount != '0' and component_amount != '':
                        all_components.add(component_name_clean)
    
    # Преобразуем в список и сортируем
    component_list = sorted(list(all_components))
    
    print(f'\nНайдено уникальных компонентов (без служебных полей и воды): {len(component_list)}')
    print(f'Первые 20 компонентов:')
    for i, comp in enumerate(component_list[:20]):
        print(f'  {i+1:3d}. {comp}')
    
    # Создаем маппинг (индекс -> название компонента)
    component_mapping = {i: comp for i, comp in enumerate(component_list)}
    
    # Сохраняем в JSON
    output_path = Path('data/processed/component_mapping.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(component_mapping, f, ensure_ascii=False, indent=2)
    
    print(f'\nСоздан маппинг компонентов: {output_path}')
    
    # Также создаем обратный маппинг (название -> индекс)
    reverse_mapping = {comp: i for i, comp in enumerate(component_list)}
    reverse_output_path = Path('data/processed/component_reverse_mapping.json')
    with open(reverse_output_path, 'w', encoding='utf-8') as f:
        json.dump(reverse_mapping, f, ensure_ascii=False, indent=2)
    
    print(f'Создан обратный маппинг: {reverse_output_path}')
    
    # Создаем CSV для удобства просмотра
    csv_path = Path('data/processed/component_mapping.csv')
    df = pd.DataFrame(list(component_mapping.items()), columns=['Index', 'Component'])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'Создан CSV файл: {csv_path}')
    
    # Проверяем, что вода действительно исключена
    print(f'\nПроверка исключения воды:')
    water_found = False
    for idx, comp in component_mapping.items():
        if 'вода' in comp.lower():
            water_found = True
            print(f'  ВНИМАНИЕ: Найден компонент с водой: {comp}')
    
    if not water_found:
        print('  ✓ Вода успешно исключена из всех маппингов')
    
    return component_mapping

if __name__ == '__main__':
    create_component_mapping()" > create_component_mapping_corrected.py
