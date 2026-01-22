#!/usr/bin/env python3
"""Создание тестовых изображений для всех рецептов"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_images():
    """Создание тестовых изображений для всех рецептов"""
    
    # Загружаем созданный манифест
    manifest_path = Path("data/data_manifest_detailed.json")
    if not manifest_path.exists():
        print("Сначала запустите create_data_manifest.py!")
        return
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    base_dir = Path("data/raw/images")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Цвета для разных типов рецептов
    colors = {
        "Терразит": (180, 160, 140),    # Коричневый
        "Шовный": (150, 150, 150),      # Серый
        "Мастика": (200, 190, 180),     # Бежевый
        "Терраццо": (160, 140, 120),    # Темно-бежевый
        "Ретушь": (170, 170, 150)       # Нейтральный
    }
    
    created_count = 0
    recipes = manifest.get('recipes', [])
    
    print(f"Всего рецептов в манифесте: {len(recipes)}")
    print("Создание тестовых изображений...")
    
    for recipe in recipes:
        recipe_id = recipe.get('id', '')
        recipe_name = recipe.get('name', '')
        recipe_type = recipe.get('type', 'Терразит')
        
        if not recipe_id:
            continue
            
        # Создаем директорию для рецепта
        recipe_dir = base_dir / str(recipe_id)
        recipe_dir.mkdir(exist_ok=True)
        
        # Базовый цвет для типа рецепта
        base_color = colors.get(recipe_type, (180, 160, 140))
        
        # Создаем 3 тестовых изображения для каждого рецепта
        for i in range(1, 4):
            # Генерируем уникальный цвет с небольшими вариациями
            color_variation = tuple(np.clip(np.array(base_color) + np.random.randint(-20, 20, 3), 0, 255))
            
            # Создаем изображение с текстурой
            img_array = np.random.randint(color_variation[0]-30, color_variation[0]+30, (224, 224, 3), dtype=np.uint8)
            
            # Добавляем текстурный шум
            texture = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
            img_array = np.clip(img_array + texture, 0, 255).astype(np.uint8)
            
            # Создаем PIL изображение
            img = Image.fromarray(img_array)
            
            # Добавляем текст с информацией о рецепте
            draw = ImageDraw.Draw(img)
            
            # Простой текст (без шрифта)
            text = f"ID: {recipe_id}\n{recipe_name}\nТип: {recipe_type}"
            
            # Разбиваем текст на строки
            lines = text.split('\n')
            y_position = 10
            
            for line in lines:
                # Простой способ добавления текста
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        if x_offset == 0 and y_offset == 0:
                            continue
                        draw.text((10 + x_offset, y_position + y_offset), line, fill=(0, 0, 0))
                
                draw.text((10, y_position), line, fill=(255, 255, 255))
                y_position += 20
            
            # Сохраняем изображение
            img_filename = f"{recipe_name.replace(' ', '_').replace('/', '_')}_{i}.jpg"
            img_path = recipe_dir / img_filename
            img.save(img_path, "JPEG", quality=95)
            
            created_count += 1
            
            # Для больших наборов данных показываем прогресс
            if len(recipes) > 50 and created_count % 50 == 0:
                print(f"  Создано {created_count} изображений...")
    
    print(f"\n✅ Создано {created_count} тестовых изображений")
    print(f"📁 Директория: {base_dir}")
    print(f"📊 Рецептов с изображениями: {len([d for d in base_dir.iterdir() if d.is_dir()])}")
    
    # Статистика по типам рецептов
    type_stats = {}
    for recipe in recipes:
        recipe_type = recipe.get('type', 'Терразит')
        type_stats[recipe_type] = type_stats.get(recipe_type, 0) + 1
    
    print("\n📈 Статистика по типам рецептов:")
    for recipe_type, count in type_stats.items():
        print(f"  {recipe_type}: {count} рецептов")

def check_images_structure():
    """Проверка структуры изображений"""
    images_dir = Path("data/raw/images")
    
    if not images_dir.exists():
        print("❌ Директория с изображениями не существует")
        return
    
    # Подсчет директорий (рецептов)
    recipe_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    
    print(f"📁 Всего директорий с рецептами: {len(recipe_dirs)}")
    print(f"📷 Всего изображений: {sum(len(list(d.glob('*.jpg'))) for d in recipe_dirs)}")
    
    # Показываем первые 5 директорий
    print("\nПример структуры (первые 5 рецептов):")
    for i, recipe_dir in enumerate(recipe_dirs[:5]):
        images = list(recipe_dir.glob('*.jpg'))
        print(f"  {recipe_dir.name}: {len(images)} изображений")
        if images:
            print(f"    Пример: {images[0].name}")

if __name__ == "__main__":
    print("="*60)
    print("СОЗДАНИЕ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ ДЛЯ TERRAZITE AI")
    print("="*60)
    
    # 1. Создаем изображения
    create_test_images()
    
    print("\n" + "="*60)
    print("ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")
    print("="*60)
    
    # 2. Проверяем структуру
    check_images_structure()
    
    print("\n" + "="*60)
    print("✅ ЗАДАЧА ВЫПОЛНЕНА")
    print("="*60)
    print("\nТеперь можно:")
    print("1. Запустить create_data_manifest.py снова для создания ML манифеста")
    print("2. Подготовить датасет: python scripts/prepare_image_dataset.py")
    print("3. Протестировать модель")
