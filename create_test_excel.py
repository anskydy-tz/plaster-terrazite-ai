"""
Создание тестовых данных для проекта Terrazite AI
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_test_excel():
    """Создание тестового Excel файла с рецептами"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем тестовые данные
    recipes = []
    components = ['мрамор', 'кварц', 'гранит', 'слюда', 'известняк', 'цемент', 'песок', 
                  'вода', 'пигмент_красный', 'пигмент_синий', 'пигмент_желтый', 
                  'пластификатор', 'волокно', 'добавка_1', 'добавка_2']
    
    for i in range(1, 51):  # 50 рецептов
        recipe = {
            'id': i,
            'название': 'Терразитовая смесь ' + str(i),
            'тип': np.random.choice(['внутренняя', 'фасадная', 'декоративная'], p=[0.4, 0.4, 0.2]),
            'описание': 'Рецепт терразитовой штукатурки №' + str(i) + ' для тестирования'
        }
        
        # Добавляем компоненты (сумма = 100%)
        comp_values = np.random.dirichlet(np.ones(len(components)), size=1)[0] * 100
        for comp, val in zip(components, comp_values):
            recipe[comp] = round(val, 2)
        
        recipes.append(recipe)
    
    # Создаем DataFrame
    df = pd.DataFrame(recipes)
    
    # Сохраняем в Excel
    excel_path = data_dir / "recipes.xlsx"
    df.to_excel(excel_path, index=False)
    
    print('Создан тестовый Excel файл:', excel_path)
    print('Количество рецептов:', len(df))
    print('Колонки:', ', '.join(df.columns.tolist()))
    
    # Создаем структуру папок для изображений
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for i in range(1, 11):  # Создаем 10 папок с рецептами
        recipe_dir = images_dir / str(i)
        recipe_dir.mkdir(exist_ok=True)
        print('Создана папка для изображений:', recipe_dir)
    
    return excel_path

def create_sample_images():
    """Создание заглушек для изображений (пустые файлы)"""
    images_dir = Path("data/raw/images")
    
    for recipe_dir in images_dir.iterdir():
        if recipe_dir.is_dir():
            # Создаем 5 тестовых изображений для каждого рецепта
            for j in range(1, 6):
                img_path = recipe_dir / f"image_{j}.txt"
                with open(img_path, 'w') as f:
                    f.write(f"Тестовое изображение для рецепта {recipe_dir.name}\n")
                    f.write(f"Здесь должна быть фотография образца штукатурки\n")
            print(f"Созданы тестовые файлы для рецепта {recipe_dir.name}")
    
    print("Тестовые файлы изображений созданы")
    print("В реальном проекте замените .txt файлы на реальные фотографии .jpg/.png")

if __name__ == "__main__":
    print("="*60)
    print("СОЗДАНИЕ ТЕСТОВЫХ ДАННЫХ ДЛЯ TERRAZITE AI")
    print("="*60)
    
    excel_path = create_test_excel()
    create_sample_images()
    
    print("\n" + "="*60)
    print("ТЕСТОВЫЕ ДАННЫЯ СОЗДАНЫ")
    print("="*60)
    print("\nСледующие шаги:")
    print("1. Запустите обработку Excel: python scripts/process_excel.py")
    print("2. Запустите создание манифеста: python scripts/create_data_manifest.py")
    print("3. Замените тестовые .txt файлы на реальные фотографии .jpg/.png")
