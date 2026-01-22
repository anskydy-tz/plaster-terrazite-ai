#!/usr/bin/env python3
"""Рабочий скрипт создания тестовых изображений"""

import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def create_simple_images():
    """Создание простых тестовых изображений без текста"""
    
    # Очищаем старую директорию
    images_dir = Path("data/raw/images")
    if images_dir.exists():
        shutil.rmtree(images_dir)
    
    # Создаем новую директорию
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные из Excel (прямо через pandas, без манифеста)
    import pandas as pd
    
    excel_path = Path("data/raw/recipes.xlsx")
    if not excel_path.exists():
        print("❌ Excel файл не найден")
        return 0
    
    df = pd.read_excel(excel_path, header=0)
    
    # Удаляем строку с итогами
    first_col = df.columns[0]
    df = df[~df[first_col].astype(str).str.contains('Общая сумма')]
    
    print(f"Обработка {len(df)} рецептов...")
    
    created_count = 0
    
    for idx, row in df.iterrows():
        recipe_id = str(idx + 1)
        recipe_name = str(row[first_col]).strip()[:50]  # Берем первые 50 символов
        
        # Создаем безопасное имя директории
        safe_dir_name = recipe_id  # Используем просто ID
        
        recipe_dir = images_dir / safe_dir_name
        recipe_dir.mkdir(exist_ok=True)
        
        # Определяем цвет по названию
        if 'Терразит' in recipe_name:
            base_color = (180, 160, 140)  # Коричневый
        elif 'Шовный' in recipe_name:
            base_color = (150, 150, 150)  # Серый
        elif 'Мастика' in recipe_name:
            base_color = (200, 190, 180)  # Бежевый
        elif 'Терраццо' in recipe_name:
            base_color = (160, 140, 120)  # Темно-бежевый
        elif 'Ретушь' in recipe_name:
            base_color = (170, 170, 150)  # Нейтральный
        else:
            base_color = (180, 160, 140)  # По умолчанию
        
        # Создаем 3 простых изображения без текста
        for i in range(1, 4):
            # Создаем цветное изображение с небольшой текстурой
            color_variation = np.array(base_color) + np.random.randint(-30, 30, 3)
            color_variation = np.clip(color_variation, 0, 255).astype(np.uint8)
            
            # Создаем базовое изображение
            img_array = np.full((224, 224, 3), color_variation, dtype=np.uint8)
            
            # Добавляем немного текстуры
            texture = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img_array = np.clip(img_array + texture, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            img_path = recipe_dir / f"sample_{i}.jpg"
            img.save(img_path, "JPEG", quality=90)
            created_count += 1
        
        # Показываем прогресс для больших наборов
        if len(df) > 50 and (idx + 1) % 50 == 0:
            print(f"  Обработано {idx + 1}/{len(df)} рецептов...")
    
    return created_count

def check_structure():
    """Проверка структуры созданных изображений"""
    images_dir = Path("data/raw/images")
    
    if not images_dir.exists():
        print("❌ Директория не создана")
        return
    
    dirs = list(images_dir.iterdir())
    dirs = [d for d in dirs if d.is_dir()]
    
    print(f"\n📁 Создано директорий: {len(dirs)}")
    
    if dirs:
        # Подсчет изображений в первых 5 директориях
        print("📊 Пример структуры (первые 5 рецептов):")
        for i, d in enumerate(dirs[:5]):
            images = list(d.glob("*.jpg"))
            print(f"  {d.name}: {len(images)} изображений")
    
    # Общий подсчет
    total_images = 0
    for d in dirs:
        images = list(d.glob("*.jpg"))
        total_images += len(images)
    
    print(f"\n📷 Всего создано изображений: {total_images}")

def create_manifest_file():
    """Создание простого файла манифеста для скрипта prepare_image_dataset.py"""
    import pandas as pd
    
    images_dir = Path("data/raw/images")
    if not images_dir.exists():
        print("❌ Директория с изображениями не найдена")
        return
    
    # Создаем список всех изображений
    image_records = []
    
    for recipe_dir in images_dir.iterdir():
        if not recipe_dir.is_dir():
            continue
            
        recipe_id = recipe_dir.name
        
        for img_file in recipe_dir.glob("*.jpg"):
            image_records.append({
                'image_path': str(img_file.relative_to(Path("data/raw"))),
                'recipe_id': recipe_id,
                'recipe_name': f"Рецепт_{recipe_id}",
                'recipe_type': 'Терразит',  # Можно уточнить позже
                'split': 'train'  # По умолчанию все в train
            })
    
    if not image_records:
        print("❌ Не найдено изображений")
        return
    
    # Создаем DataFrame и сохраняем
    df = pd.DataFrame(image_records)
    
    # Разделяем на train/val/test (70/15/15)
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)
    
    train_end = int(0.7 * n)
    val_end = train_end + int(0.15 * n)
    
    df.loc[indices[:train_end], 'split'] = 'train'
    df.loc[indices[train_end:val_end], 'split'] = 'val'
    df.loc[indices[val_end:], 'split'] = 'test'
    
    # Сохраняем
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "data_manifest_full.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Манифест создан: {output_path}")
    print(f"📊 Распределение:")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Val: {len(df[df['split'] == 'val'])}")
    print(f"  Test: {len(df[df['split'] == 'test'])}")
    
    return output_path

if __name__ == "__main__":
    print("="*60)
    print("СОЗДАНИЕ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ (РАБОЧАЯ ВЕРСИЯ)")
    print("="*60)
    
    # 1. Создаем изображения
    print("\n1. Создание изображений...")
    count = create_simple_images()
    print(f"✅ Создано {count} изображений")
    
    # 2. Проверяем структуру
    print("\n2. Проверка структуры...")
    check_structure()
    
    # 3. Создаем манифест
    print("\n3. Создание CSV манифеста...")
    manifest_path = create_manifest_file()
    
    print("\n" + "="*60)
    print("✅ ЗАДАЧА ВЫПОЛНЕНА")
    print("="*60)
    print("\nТеперь можно:")
    if manifest_path:
        print(f"1. Манифест готов: {manifest_path}")
        print("2. Запустить подготовку датасета:")
        print("   python scripts/prepare_image_dataset.py")
    print("3. Протестировать модель: python test_model_basic.py")
