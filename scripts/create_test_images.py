"""
Создание тестовых изображений для проекта Terrazite AI
"""
import sys
sys.path.append('src')

import numpy as np
from PIL import Image
from pathlib import Path
import json
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_solid_color_image(color, size=(224, 224)):
    """Создание изображения однородного цвета"""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = color
    return Image.fromarray(image)

def create_gradient_image(start_color, end_color, size=(224, 224)):
    """Создание градиентного изображения"""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    for i in range(size[0]):
        # Линейная интерполяция между цветами
        ratio = i / size[0]
        color = [
            int(start_color[j] * (1 - ratio) + end_color[j] * ratio)
            for j in range(3)
        ]
        image[i, :] = color
    
    return Image.fromarray(image)

def create_texture_image(base_color, texture_strength=0.3, size=(224, 224)):
    """Создание текстурированного изображения"""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = base_color
    
    # Добавляем шум для текстуры
    noise = np.random.randint(-texture_strength*50, texture_strength*50, 
                             (size[0], size[1], 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image)

def create_test_images_for_recipe(recipe_id, recipe_name, recipe_type, output_dir, num_images=3):
    """Создание тестовых изображений для рецепта"""
    recipe_dir = output_dir / f"recipe_{recipe_id}"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    
    # Цвета в зависимости от типа рецепта
    if 'Терразит' in recipe_type:
        # Серые и бежевые оттенки
        colors = [
            (200, 200, 200),  # Светло-серый
            (180, 160, 140),  # Бежевый
            (150, 150, 150)   # Серый
        ]
    elif 'Шовный' in recipe_type:
        # Серые оттенки
        colors = [
            (160, 160, 160),
            (140, 140, 140),
            (120, 120, 120)
        ]
    elif 'Мастика' in recipe_type:
        # Светлые оттенки
        colors = [
            (240, 240, 240),  # Почти белый
            (230, 220, 210),  # Кремовый
            (220, 210, 200)   # Светло-бежевый
        ]
    elif 'Терраццо' in recipe_type:
        # Разноцветные (мраморные)
        colors = [
            (250, 240, 230),  # Светло-мраморный
            (230, 220, 210),  # Мраморный
            (210, 200, 190)   # Темно-мраморный
        ]
    elif 'Ретушь' in recipe_type:
        # Коричневые оттенки
        colors = [
            (180, 150, 120),  # Светло-коричневый
            (160, 130, 100),  # Коричневый
            (140, 110, 80)    # Темно-коричневый
        ]
    else:
        # Стандартные оттенки
        colors = [
            (200, 200, 200),
            (180, 180, 180),
            (160, 160, 160)
        ]
    
    created_images = []
    
    for i in range(num_images):
        # Создаем разные типы изображений
        if i == 0:
            # Однородный цвет
            img = create_solid_color_image(colors[i % len(colors)])
        elif i == 1:
            # Градиент
            start_color = colors[i % len(colors)]
            end_color = colors[(i + 1) % len(colors)]
            img = create_gradient_image(start_color, end_color)
        else:
            # Текстура
            img = create_texture_image(colors[i % len(colors)], texture_strength=0.2)
        
        # Сохраняем изображение
        img_path = recipe_dir / f"image_{i+1}.jpg"
        img.save(img_path, 'JPEG', quality=95)
        created_images.append(str(img_path))
        
        logger.debug(f"Создано изображение: {img_path}")
    
    return created_images

def main():
    """Основная функция"""
    logger.info("Создание тестовых изображений для Terrazite AI")
    logger.info("=" * 60)
    
    # Директория для сохранения тестовых изображений
    output_dir = Path("data/raw/test_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем рецепты из Excel или создаем тестовые
    try:
        from src.data.loader import RecipeLoader
        
        excel_path = Path("data/raw/recipes.xlsx")
        if excel_path.exists():
            logger.info(f"Загрузка рецептов из: {excel_path}")
            loader = RecipeLoader(str(excel_path))
            df = loader.load_excel()
            
            recipes = []
            for idx, row in df.iterrows():
                recipe_name = row['recipe_name'] if 'recipe_name' in row.index else f"Recipe_{idx}"
                recipe_type = row['category'] if 'category' in row.index else 'Терразит'
                recipes.append({
                    'id': idx + 1,
                    'name': recipe_name,
                    'type': recipe_type
                })
            
            logger.info(f"Загружено {len(recipes)} рецептов")
        else:
            logger.warning(f"Excel файл не найден: {excel_path}")
            logger.info("Создание тестовых рецептов")
            
            # Создаем тестовые рецепты
            recipes = []
            recipe_types = ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь']
            
            for i in range(50):  # 50 тестовых рецептов
                recipe_type = recipe_types[i % len(recipe_types)]
                recipes.append({
                    'id': i + 1,
                    'name': f"{recipe_type}_Тестовый_{i+1}",
                    'type': recipe_type
                })
            
            logger.info(f"Создано {len(recipes)} тестовых рецептов")
            
    except Exception as e:
        logger.error(f"Ошибка загрузки рецептов: {e}")
        logger.info("Создание минимального набора тестовых рецептов")
        
        # Минимальный набор тестовых рецептов
        recipes = []
        recipe_types = ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь']
        
        for i, recipe_type in enumerate(recipe_types):
            recipes.append({
                'id': i + 1,
                'name': f"{recipe_type}_Тестовый",
                'type': recipe_type
            })
    
    # Создаем изображения для каждого рецепта
    all_images = []
    manifest_data = []
    
    logger.info(f"Создание изображений в: {output_dir}")
    
    for recipe in recipes:
        try:
            images = create_test_images_for_recipe(
                recipe['id'], 
                recipe['name'],
                recipe['type'],
                output_dir,
                num_images=3
            )
            
            recipe_data = {
                'recipe_id': recipe['id'],
                'recipe_name': recipe['name'],
                'recipe_type': recipe['type'],
                'images': images
            }
            all_images.append(recipe_data)
            
            # Заполняем данные для манифеста
            for img_path in images:
                manifest_data.append({
                    'image_path': img_path,
                    'recipe_id': recipe['id'],
                    'recipe_name': recipe['name'],
                    'recipe_type': recipe['type'],
                    'split': 'train'  # По умолчанию все в train для тестов
                })
            
            logger.info(f"  Рецепт {recipe['id']}: {recipe['name']} - {len(images)} изображений")
            
        except Exception as e:
            logger.error(f"Ошибка создания изображений для рецепта {recipe['id']}: {e}")
    
    # Сохраняем детальный манифест изображений
    manifest_path = output_dir / "images_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_recipes': len(all_images),
            'total_images': sum(len(item['images']) for item in all_images),
            'recipes': all_images
        }, f, ensure_ascii=False, indent=2)
    
    # СОЗДАЕМ CSV МАНИФЕСТ, СОВМЕСТИМЫЙ С create_data_manifest.py
    csv_manifest_path = Path("data/raw/test_images_manifest.csv")
    if manifest_data:
        df = pd.DataFrame(manifest_data)
        df.to_csv(csv_manifest_path, index=False, encoding='utf-8')
        logger.info(f"CSV манифест сохранен: {csv_manifest_path}")
    else:
        logger.warning("Нет данных для создания CSV манифеста")
    
    logger.info("=" * 60)
    logger.info(f"Создано изображений: {sum(len(item['images']) for item in all_images)}")
    logger.info(f"Создано рецептов: {len(all_images)}")
    logger.info(f"Детальный манифест: {manifest_path}")
    logger.info(f"CSV манифест: {csv_manifest_path}")
    logger.info("=" * 60)
    
    print(f"\n✅ ТЕСТОВЫЕ ИЗОБРАЖЕНИЯ УСПЕШНО СОЗДАНЫ!")
    print(f"   Директория: {output_dir}")
    print(f"   Рецептов: {len(all_images)}")
    print(f"   Изображений: {sum(len(item['images']) for item in all_images)}")
    print(f"   CSV манифест: {csv_manifest_path}")
    print(f"\n📋 Чтобы использовать в пайплайне:")
    print(f"   python scripts/prepare_image_dataset.py --manifest data/raw/test_images_manifest.csv")
    
    return all_images

if __name__ == "__main__":
    main()
