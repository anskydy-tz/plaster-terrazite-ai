#!/usr/bin/env python3
"""
Скрипт для подготовки датасета изображений для обучения модели.
Создает связи между изображениями и рецептами из базы данных.
"""
import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

import json
import shutil
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime
import pandas as pd

from src.utils.config import config, setup_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageDatasetPreparer:
    """
    Класс для подготовки датасета изображений.
    """
    
    def __init__(self, images_dir: str = None, recipes_json: str = None):
        """
        Инициализация подготовщика данных.
        
        Args:
            images_dir: Директория с изображениями
            recipes_json: Путь к JSON с рецептами
        """
        if images_dir is None:
            images_dir = config.data.images_dir
        if recipes_json is None:
            recipes_json = Path(config.project_root) / config.data.processed_data_dir / config.data.processed_json
        
        self.images_dir = Path(images_dir)
        self.recipes_json = Path(recipes_json)
        self.dataset_info = {}
        
        # Создаем структуру директорий
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Создание структуры директорий для датасета."""
        directories = [
            "data/raw/images",
            "data/processed/images/train",
            "data/processed/images/val",
            "data/processed/images/test",
            "data/processed/images/augmented",
            "data/processed/metadata"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Структура директорий создана")
    
    def scan_existing_images(self) -> Dict[str, List[str]]:
        """
        Сканирование существующих изображений.
        
        Returns:
            Словарь: имя_рецепта -> список путей к изображениям
        """
        images_by_recipe = {}
        
        if not self.images_dir.exists():
            logger.warning(f"Директория с изображениями не найдена: {self.images_dir}")
            return images_by_recipe
        
        # Поддерживаемые форматы
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Рекурсивно ищем все изображения
        for ext in image_extensions:
            for image_path in self.images_dir.rglob(f"*{ext}"):
                # Пытаемся определить рецепт из имени файла или директории
                recipe_name = self._extract_recipe_name(image_path)
                if recipe_name:
                    if recipe_name not in images_by_recipe:
                        images_by_recipe[recipe_name] = []
                    images_by_recipe[recipe_name].append(str(image_path))
        
        logger.info(f"Найдено изображений: {sum(len(imgs) for imgs in images_by_recipe.values())}")
        logger.info(f"Уникальных рецептов с изображениями: {len(images_by_recipe)}")
        
        return images_by_recipe
    
    def _extract_recipe_name(self, image_path: Path) -> Optional[str]:
        """
        Извлечение имени рецепта из пути к изображению.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Имя рецепта или None
        """
        # Пробуем несколько стратегий:
        
        # 1. Из имени файла (удаляем расширение и номера)
        filename = image_path.stem
        
        # Убираем цифры и специальные символы в конце
        import re
        cleaned_name = re.sub(r'[\d_\-]*$', '', filename)
        cleaned_name = cleaned_name.strip('_').strip('-')
        
        # 2. Из имени родительской директории
        parent_name = image_path.parent.name
        
        # 3. Ищем в базе рецептов наиболее похожее имя
        if not hasattr(self, '_recipe_names'):
            self._load_recipe_names()
        
        # Ищем совпадения
        possible_names = [cleaned_name, parent_name]
        
        for name in possible_names:
            if name in self._recipe_names:
                return name
        
        # Если точного совпадения нет, ищем частичное
        for recipe_name in self._recipe_names:
            if recipe_name.lower() in cleaned_name.lower() or cleaned_name.lower() in recipe_name.lower():
                return recipe_name
        
        return None
    
    def _load_recipe_names(self):
        """Загрузка имен рецептов из JSON."""
        if not self.recipes_json.exists():
            self._recipe_names = []
            return
        
        with open(self.recipes_json, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
        
        self._recipe_names = [recipe['name'] for recipe in recipes_data.get('recipes', [])]
        logger.info(f"Загружено имен рецептов: {len(self._recipe_names)}")
    
    def create_dataset_manifest(self, images_by_recipe: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Создание манифеста датасета.
        
        Args:
            images_by_recipe: Словарь рецепт -> список изображений
            
        Returns:
            DataFrame с информацией о датасете
        """
        rows = []
        
        # Загружаем информацию о рецептах
        if not self.recipes_json.exists():
            logger.error(f"Файл с рецептами не найден: {self.recipes_json}")
            return pd.DataFrame()
        
        with open(self.recipes_json, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
        
        # Создаем словарь рецептов для быстрого поиска
        recipes_dict = {recipe['name']: recipe for recipe in recipes_data['recipes']}
        
        # Создаем записи для каждого изображения
        for recipe_name, image_paths in images_by_recipe.items():
            if recipe_name not in recipes_dict:
                logger.warning(f"Рецепт '{recipe_name}' не найден в базе данных")
                continue
            
            recipe_info = recipes_dict[recipe_name]
            
            for i, image_path in enumerate(image_paths):
                row = {
                    'image_id': f"{recipe_name}_{i:03d}",
                    'image_path': str(image_path),
                    'recipe_name': recipe_name,
                    'recipe_category': recipe_info['category'],
                    'component_count': recipe_info['component_count'],
                    'total_weight': recipe_info['total_weight'],
                    'split': self._assign_split(recipe_name, i, len(image_paths))
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Сохраняем манифест
        manifest_path = Path("data/processed/metadata/dataset_manifest.csv")
        df.to_csv(manifest_path, index=False, encoding='utf-8')
        
        logger.info(f"Манифест создан: {manifest_path}")
        logger.info(f"Записей в манифесте: {len(df)}")
        logger.info(f"Распределение по сплитам: {dict(df['split'].value_counts())}")
        
        return df
    
    def _assign_split(self, recipe_name: str, image_index: int, total_images: int) -> str:
        """
        Назначение сплита для изображения.
        
        Args:
            recipe_name: Имя рецепта
            image_index: Индекс изображения
            total_images: Общее количество изображений для рецепта
            
        Returns:
            'train', 'val' или 'test'
        """
        # Простая стратегия: 70% train, 15% val, 15% test
        # Используем хеш имени рецепта для детерминированности
        import hashlib
        recipe_hash = int(hashlib.md5(recipe_name.encode()).hexdigest()[:8], 16)
        
        # Первое изображение всегда в train для гарантии
        if image_index == 0:
            return 'train'
        
        # Распределяем остальные
        split_value = (recipe_hash + image_index) % 100
        
        if split_value < 70:
            return 'train'
        elif split_value < 85:
            return 'val'
        else:
            return 'test'
    
    def copy_images_to_dataset_structure(self, manifest_df: pd.DataFrame):
        """
        Копирование изображений в структурированную директорию датасета.
        
        Args:
            manifest_df: DataFrame с манифестом
        """
        if manifest_df.empty:
            logger.warning("Манифест пуст. Изображения не скопированы.")
            return
        
        for _, row in manifest_df.iterrows():
            src_path = Path(row['image_path'])
            split = row['split']
            recipe_name = row['recipe_name']
            image_id = row['image_id']
            
            # Создаем целевой путь
            dst_dir = Path(f"data/processed/images/{split}/{recipe_name}")
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst_path = dst_dir / f"{image_id}{src_path.suffix}"
            
            try:
                # Копируем изображение
                shutil.copy2(src_path, dst_path)
                
                # Создаем уменьшенную версию для предпросмотра
                self._create_preview_image(src_path, dst_dir / f"{image_id}_preview.jpg")
                
            except Exception as e:
                logger.error(f"Ошибка копирования {src_path}: {e}")
        
        logger.info(f"Изображения скопированы в структурированную директорию")
    
    def _create_preview_image(self, src_path: Path, dst_path: Path, size=(256, 256)):
        """
        Создание уменьшенной версии изображения для предпросмотра.
        
        Args:
            src_path: Путь к исходному изображению
            dst_path: Путь для сохранения превью
            size: Размер превью
        """
        try:
            img = Image.open(src_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(dst_path, "JPEG", quality=85)
        except Exception as e:
            logger.warning(f"Не удалось создать превью для {src_path}: {e}")
    
    def augment_images(self, manifest_df: pd.DataFrame):
        """
        Аугментация изображений для увеличения датасета.
        
        Args:
            manifest_df: DataFrame с манифестом
        """
        if not config.data.augmentation_enabled:
            logger.info("Аугментация отключена в конфигурации")
            return
        
        logger.info("Начало аугментации изображений...")
        
        # Фильтруем только train изображения
        train_df = manifest_df[manifest_df['split'] == 'train']
        
        augmented_rows = []
        augmented_count = 0
        
        for _, row in train_df.iterrows():
            src_path = Path(row['image_path'])
            
            if not src_path.exists():
                continue
            
            # Создаем 3 аугментированные версии
            for aug_idx in range(3):
                aug_image = self._apply_augmentation(src_path, aug_idx)
                if aug_image is not None:
                    # Сохраняем аугментированное изображение
                    aug_dir = Path("data/processed/images/augmented") / row['recipe_name']
                    aug_dir.mkdir(parents=True, exist_ok=True)
                    
                    aug_filename = f"{row['image_id']}_aug{aug_idx}{src_path.suffix}"
                    aug_path = aug_dir / aug_filename
                    
                    aug_image.save(aug_path)
                    
                    # Добавляем запись в манифест
                    aug_row = row.copy()
                    aug_row['image_path'] = str(aug_path)
                    aug_row['image_id'] = f"{row['image_id']}_aug{aug_idx}"
                    aug_row['split'] = 'train'  # Аугментированные тоже в train
                    
                    augmented_rows.append(aug_row)
                    augmented_count += 1
        
        # Добавляем аугментированные строки в манифест
        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            updated_manifest = pd.concat([manifest_df, aug_df], ignore_index=True)
            
            # Сохраняем обновленный манифест
            manifest_path = Path("data/processed/metadata/dataset_manifest_augmented.csv")
            updated_manifest.to_csv(manifest_path, index=False, encoding='utf-8')
            
            logger.info(f"Аугментация завершена. Добавлено {augmented_count} изображений")
            logger.info(f"Обновленный манифест: {manifest_path}")
            
            return updated_manifest
        
        return manifest_df
    
    def _apply_augmentation(self, image_path: Path, aug_idx: int) -> Optional[Image.Image]:
        """
        Применение аугментации к изображению.
        
        Args:
            image_path: Путь к изображению
            aug_idx: Индекс аугментации
            
        Returns:
            Аугментированное изображение или None
        """
        try:
            img = Image.open(image_path)
            
            # Преобразуем в numpy для OpenCV
            img_np = np.array(img)
            
            # Применяем разные аугментации в зависимости от индекса
            if aug_idx == 0:
                # Поворот
                angle = np.random.uniform(-config.data.rotation_range, config.data.rotation_range)
                h, w = img_np.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(img_np, matrix, (w, h))
                
            elif aug_idx == 1:
                # Сдвиг
                tx = np.random.uniform(-config.data.width_shift_range, config.data.width_shift_range) * img_np.shape[1]
                ty = np.random.uniform(-config.data.height_shift_range, config.data.height_shift_range) * img_np.shape[0]
                matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                augmented = cv2.warpAffine(img_np, matrix, (img_np.shape[1], img_np.shape[0]))
                
            elif aug_idx == 2:
                # Изменение яркости/контраста
                alpha = np.random.uniform(0.8, 1.2)  # Контраст
                beta = np.random.uniform(-30, 30)    # Яркость
                augmented = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
            
            else:
                # Горизонтальное отражение
                if config.data.horizontal_flip and np.random.random() > 0.5:
                    augmented = cv2.flip(img_np, 1)
                else:
                    augmented = img_np
            
            # Преобразуем обратно в PIL Image
            return Image.fromarray(augmented)
            
        except Exception as e:
            logger.warning(f"Ошибка аугментации {image_path}: {e}")
            return None
    
    def generate_dataset_report(self, manifest_df: pd.DataFrame):
        """
        Генерация отчета о датасете.
        
        Args:
            manifest_df: DataFrame с манифестом
        """
        if manifest_df.empty:
            logger.warning("Манифест пуст. Отчет не сгенерирован.")
            return
        
        report_path = Path("reports/dataset_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О ДАТАСЕТЕ ИЗОБРАЖЕНИЙ TERRAZITE AI\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего изображений: {len(manifest_df)}\n\n")
            
            # Статистика по сплитам
            f.write("РАСПРЕДЕЛЕНИЕ ПО СПЛИТАМ:\n")
            f.write("-" * 40 + "\n")
            split_counts = manifest_df['split'].value_counts()
            for split, count in split_counts.items():
                percentage = (count / len(manifest_df)) * 100
                f.write(f"{split.upper()}: {count} изображений ({percentage:.1f}%)\n")
            
            f.write("\nРАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:\n")
            f.write("-" * 40 + "\n")
            category_counts = manifest_df['recipe_category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(manifest_df)) * 100
                f.write(f"{category}: {count} изображений ({percentage:.1f}%)\n")
            
            # Статистика по рецептам
            f.write("\nРЕЦЕПТЫ С ИЗОБРАЖЕНИЯМИ:\n")
            f.write("-" * 40 + "\n")
            recipe_counts = manifest_df['recipe_name'].value_counts()
            f.write(f"Всего уникальных рецептов: {len(recipe_counts)}\n")
            
            f.write("\nТоп-10 рецептов по количеству изображений:\n")
            for i, (recipe, count) in enumerate(recipe_counts.head(10).items(), 1):
                f.write(f"  {i:2d}. {recipe}: {count} изображений\n")
            
            # Информация о изображениях
            f.write("\nИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИЯХ:\n")
            f.write("-" * 40 + "\n")
            
            # Пример анализа размеров (первые 10 изображений)
            sizes = []
            for _, row in manifest_df.head(10).iterrows():
                try:
                    img = Image.open(row['image_path'])
                    sizes.append(img.size)
                except:
                    pass
            
            if sizes:
                avg_width = sum(s[0] for s in sizes) / len(sizes)
                avg_height = sum(s[1] for s in sizes) / len(sizes)
                f.write(f"Средний размер: {avg_width:.0f}x{avg_height:.0f}\n")
            
            # Пути к данным
            f.write("\nПУТИ К ДАННЫМ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Исходные изображения: {self.images_dir}\n")
            f.write(f"Обработанные изображения: data/processed/images/\n")
            f.write(f"Манифест: data/processed/metadata/dataset_manifest.csv\n")
        
        logger.info(f"Отчет о датасете сохранен: {report_path}")
        return report_path


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description='Подготовка датасета изображений для Terrazite AI')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Директория с исходными изображениями')
    parser.add_argument('--recipes-json', type=str, default=None,
                       help='Путь к JSON файлу с рецептами')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Отключить аугментацию изображений')
    parser.add_argument('--scan-only', action='store_true',
                       help='Только сканирование без копирования')
    
    args = parser.parse_args()
    
    # Настройка конфигурации
    setup_config()
    
    logger.info("Запуск подготовки датасета изображений...")
    
    # Создаем подготовщик
    preparer = ImageDatasetPreparer(
        images_dir=args.images_dir,
        recipes_json=args.recipes_json
    )
    
    # Сканируем существующие изображения
    images_by_recipe = preparer.scan_existing_images()
    
    if not images_by_recipe:
        logger.warning("Изображения не найдены. Создаю структуру для тестирования.")
        # Создаем тестовую структуру
        preparer._create_test_structure()
        images_by_recipe = preparer.scan_existing_images()
    
    # Создаем манифест
    manifest_df = preparer.create_dataset_manifest(images_by_recipe)
    
    if manifest_df.empty:
        logger.error("Манифест пуст. Проверьте наличие изображений и рецептов.")
        return
    
    if not args.scan_only:
        # Копируем изображения в структурированную директорию
        preparer.copy_images_to_dataset_structure(manifest_df)
        
        # Аугментация
        if not args.no_augmentation:
            manifest_df = preparer.augment_images(manifest_df)
    
    # Генерация отчета
    report_path = preparer.generate_dataset_report(manifest_df)
    
    logger.info("Подготовка датасета завершена!")
    
    # Вывод сводной информации
    print("\n" + "=" * 80)
    print("СВОДКА ПО ДАТАСЕТУ:")
    print("=" * 80)
    print(f"Всего изображений: {len(manifest_df)}")
    
    if not manifest_df.empty:
        print("\nРаспределение по категориям:")
        category_counts = manifest_df['recipe_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(manifest_df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\nРаспределение по сплитам:")
        split_counts = manifest_df['split'].value_counts()
        for split, count in split_counts.items():
            print(f"  {split}: {count}")
        
        print(f"\nМанифест сохранен: data/processed/metadata/dataset_manifest.csv")
        if report_path:
            print(f"Отчет сохранен: {report_path}")
        
        print("\nСледующие шаги:")
        print("1. Добавьте больше изображений в data/raw/images/")
        print("2. Запустите обучение: python scripts/train_model.py")
        print("3. Проверьте качество: python scripts/evaluate_model.py")


def _create_test_structure(self):
    """Создание тестовой структуры для демонстрации."""
    logger.info("Создание тестовой структуры изображений...")
    
    # Создаем тестовые изображения
    test_recipes = [
        "Терразит К62А",
        "Шовный МШ1", 
        "Мастика К1",
        "Терраццо Ц1М",
        "Ретушь 1"
    ]
    
    for recipe in test_recipes:
        recipe_dir = self.images_dir / recipe
        recipe_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем 3 тестовых изображения для каждого рецепта
        for i in range(3):
            # Создаем простое цветное изображение
            img = Image.new('RGB', (800, 600), color=(
                np.random.randint(100, 200),
                np.random.randint(100, 200),
                np.random.randint(100, 200)
            ))
            
            # Добавляем текст с именем рецепта
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Используем стандартный шрифт
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            text = f"{recipe}\nОбразец {i+1}"
            draw.text((50, 50), text, fill=(0, 0, 0), font=font)
            
            # Сохраняем
            img_path = recipe_dir / f"{recipe}_sample{i+1}.jpg"
            img.save(img_path, "JPEG", quality=90)
    
    logger.info(f"Создано тестовых изображений: {len(test_recipes) * 3}")


# Добавляем метод в класс
ImageDatasetPreparer._create_test_structure = _create_test_structure


if __name__ == "__main__":
    main()
