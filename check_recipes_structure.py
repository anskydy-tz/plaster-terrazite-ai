import json

def check_recipes_structure():
    with open('data/processed/recipes.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print('Тип данных:', type(data))
    print('Длина данных:', len(data) if isinstance(data, list) else 'не список')
    
    if isinstance(data, list) and len(data) > 0:
        print('\nПервый элемент (первые 500 символов):')
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:500])
        
        print('\nКлючи первого элемента:')
        print(list(data[0].keys()))
        
        if 'components' in data[0]:
            print('\nКомпоненты первого рецепта:')
            components = data[0]['components']
            if isinstance(components, dict):
                for comp, amount in components.items():
                    print(f'  {comp}: {amount}')
            else:
                print('Компоненты не в формате dict:', type(components))
                
        # Проверим тип данных компонентов в нескольких рецептах
        print('\nПроверка структуры компонентов в первых 5 рецептах:')
        for i in range(min(5, len(data))):
            if 'components' in data[i]:
                comps = data[i]['components']
                print(f'Рецепт {i}: тип компонентов - {type(comps)}')
                if isinstance(comps, dict):
                    print(f'  Количество компонентов: {len(comps)}')
                    if comps:
                        first_key = list(comps.keys())[0]
                        print(f'  Пример компонента: {first_key} -> {comps[first_key]}')

if __name__ == "__main__":
    check_recipes_structure()
