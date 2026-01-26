#!/usr/bin/env python3
"""
Тест исправленного датасета с фиксированными векторами компонентов
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.utils.config import setup_config
from src.data.loader import RecipeLoader, TerraziteDataset
import torch
from torch.utils.data import DataLoader

def test_fixed_dataset():
    """Тестирование датасета с фиксированными векторами"""
    print("=" * 80)
    print("ТЕСТ ИСПРАВЛЕННОГО ДАТАСЕТА")
    print("=" * 80)
    
    # 1. Настройка конфигурации
    config = setup_config()
    
    # 2. Загрузка данных
    excel_path = Path("data/raw/recipes.xlsx")
    if not excel_path.exists():
        print(f"❌ Excel файл не найден: {excel_path}")
        return
    
    print(f"\n1. Загрузка данных из: {excel_path}")
    recipe_loader = RecipeLoader()
    recipe_loader.load_excel(str(excel_path))
    recipes = recipe_loader.get_all_recipes()
    
    print(f"   ✅ Загружено рецептов: {len(recipes)}")
    
    # 3. Получение маппинга компонентов
    component_features = recipe_loader.component_features
    if not component_features or 'component_to_idx' not in component_features:
        print("❌ Маппинг компонентов не загружен")
        return
    
    print(f"\n2. Маппинг компонентов загружен")
    print(f"   ✅ Компонентов: {component_features.get('total_components', 0)}")
    
    # 4. Создание датасета
    print("\n3. Создание датасета...")
    dataset = TerraziteDataset(
        recipes_data=recipes[:10],  # Берем только 10 для теста
        image_dir=config.data.images_dir,
        transform=None,
        include_components=True,
        component_mapping=component_features
    )
    
    print(f"   ✅ Датасет создан: {len(dataset)} элементов")
    
    # 5. Проверка первого элемента
    print("\n4. Проверка первого элемента...")
    try:
        sample = dataset[0]
        print(f"   ✅ Первый элемент загружен:")
        print(f"      - Изображение: {sample['image'].shape}")
        print(f"      - Категория: {sample['category'].shape} (значение: {sample['category'].item()})")
        print(f"      - Компоненты: {sample['components'].shape}")
        
        # Проверка, что все векторы компонентов одинаковой длины
        component_shapes = []
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            component_shapes.append(item['components'].shape[0])
        
        if len(set(component_shapes)) == 1:
            print(f"   ✅ Все векторы компонентов имеют одинаковую длину: {component_shapes[0]}")
        else:
            print(f"❌ Векторы компонентов имеют разную длину: {component_shapes}")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке элемента: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Проверка DataLoader
    print("\n5. Проверка DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"   ✅ Батч {batch_idx}:")
            print(f"      - Изображения: {batch['image'].shape}")
            print(f"      - Категории: {batch['category'].shape}")
            print(f"      - Компоненты: {batch['components'].shape}")
            
            # Проверяем, что все векторы компонентов в батче одинаковой длины
            components_batch = batch['components']
            if components_batch.dim() == 2:
                print(f"      ✅ Все векторы в батче имеют одинаковую длину: {components_batch.shape[1]}")
            else:
                print(f"❌ Проблема с размерностью векторов: {components_batch.shape}")
            
            break  # Проверяем только первый батч
        
    except Exception as e:
        print(f"❌ Ошибка в DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТ ПРОЙДЕН УСПЕШНО!")
    print("=" * 80)
    
    print("\nРекомендации:")
    print("1. Теперь можно запустить обучение: python scripts/train_model.py --epochs 1 --batch-size 4")
    print("2. Проверьте, что component_mapping правильно загружается в RecipeLoader")
    
    return dataset

if __name__ == "__main__":
    dataset = test_fixed_dataset()
