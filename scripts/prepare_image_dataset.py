#!/usr/bin/env python3
"""
Скрипт для подготовки датасета изображений.
Работает с манифестом от create_data_manifest.py.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import shutil
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
from typing import Optional, List, Dict, Any  # ДОБАВЛЕНО: Optional

from src.utils.config import config, setup_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageDatasetPreparer:
    """
    Класс для подготовки датасета изображений на основе существующего манифеста.
    """
    
    def __init__(self, manifest_path: str = None):
        """
        Инициализация подготовщика данных.
        
        Args:
            manifest_path: Путь к CSV манифесту от create_data_manifest.py
        """
        if manifest_path is None:
            # Ищем манифест в стандартных местах
            possible_paths = [
                "data/processed/data_manifest_full.csv",
                "data/processed/data_manifest_train.csv",
                "data/raw/data_manifest.csv"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    manifest_path = path
                    break
        
        if not manifest_path or not Path(manifest_path).exists():
            raise FileNotFoundError(f"Манифест не найден. Сначала запустите create_data_manifest.py")
        
        self.manifest_path = Path(manifest_path)
        self.manifest_df = None
        self.dataset_info = {}
        
        # Загружаем манифест
        self._load_manifest()
        
        # Создаем структуру директорий
        self._create_directory_structure()
    
    def _load_manifest(self):
        """Загрузка манифеста."""
        try:
            self.manifest_df = pd.read_csv(self.manifest_path)
            logger.info(f"Манифест загружен: {self.manifest_path}")
            logger.info(f"Записей: {len(self.manifest_df)}")
            logger.info(f"Колонки: {list(self.manifest_df.columns)}")
            
            # Проверяем наличие необходимых колонок
            required_columns = ['image_path', 'recipe_id', 'split']
            missing_columns = [col for col in required_columns if col not in self.manifest_df.columns]
            
            if missing_columns:
                logger.warning(f"Отсутствуют колонки: {missing_columns}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки манифеста: {e}")
            raise
    
    def _create_directory_structure(self):
        """Создание структуры директорий для датасета."""
        directories = [
            "data/processed/images/train",
            "data/processed/images/val", 
            "data/processed/images/test",
            "data/processed/images/augmented",
            "data/processed/metadata"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Структура директорий создана")
    
    def copy_images_to_dataset_structure(self):
        """
        Копирование изображений в структурированную директорию датасета
        на основе манифеста.
        """
        if self.manifest_df.empty:
            logger.warning("Манифест пуст. Изображения не скопированы.")
            return
        
        copied_count = 0
        missing_count = 0
        
        for _, row in self.manifest_df.iterrows():
            # Определяем исходный путь
            src_path = self._resolve_image_path(row)
            
            if not src_path or not src_path.exists():
                logger.debug(f"Изображение не найдено: {row.get('image_path', 'unknown')}")
                missing_count += 1
                continue
            
            # Определяем сплит (если есть в манифесте)
            split = row.get('split', 'train')
            
            # Создаем целевой путь
            recipe_id = str(row.get('recipe_id', 'unknown'))
            image_filename = src_path.name
            
            # Если файл .txt, заменяем на .jpg для совместимости
            if image_filename.endswith('.txt'):
                image_filename = image_filename.replace('.txt', '.jpg')
            
            dst_dir = Path(f"data/processed/images/{split}/{recipe_id}")
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            dst_path = dst_dir / image_filename
            
            try:
                # Копируем изображение
                if src_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    
                    # Создаем уменьшенную версию для предпросмотра
                    self._create_preview_image(src_path, dst_dir / f"preview_{image_filename}")
                else:
                    logger.warning(f"Неподдерживаемый формат: {src_path.suffix}")
                    
            except Exception as e:
                logger.error(f"Ошибка копирования {src_path}: {e}")
        
        logger.info(f"Изображения скопированы: {copied_count}")
        logger.info(f"Изображений не найдено: {missing_count}")
        
        return copied_count
    
    def _resolve_image_path(self, row: pd.Series) -> Optional[Path]:
        """
        Определение пути к изображению на основе данных из манифеста.
        
        Args:
            row: Строка манифеста
            
        Returns:
            Путь к изображению или None
        """
        # Пробуем несколько возможных мест
        image_path = row.get('image_path', '')
        
        if pd.isna(image_path) or not image_path:
            return None
        
        # Если путь абсолютный
        if Path(image_path).is_absolute():
            return Path(image_path)
        
        # Пробуем относительно разных корневых директорий
        possible_roots = [
            Path("data/raw"),
            Path("."),
            Path("..")
        ]
        
        for root in possible_roots:
            full_path = root / image_path
            if full_path.exists():
                return full_path
        
        # Если не нашли, пробуем поискать по recipe_id
        recipe_id = str(row.get('recipe_id', ''))
        if recipe_id:
            recipe_dirs = [
                Path("data/raw/images") / recipe_id,
                Path("data/raw") / recipe_id,
                Path("images") / recipe_id
            ]
            
            for recipe_dir in recipe_dirs:
                if recipe_dir.exists():
                    # Ищем любой файл изображения в директории
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.txt']:
                        for img_file in recipe_dir.glob(f"*{ext}"):
                            return img_file
        
        return None
    
    def _create_preview_image(self, src_path: Path, dst_path: Path, size=(256, 256)):
        """Создание уменьшенной версии изображения для предпросмотра."""
        try:
            img = Image.open(src_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(dst_path, "JPEG", quality=85)
        except Exception as e:
            logger.warning(f"Не удалось создать превью для {src_path}: {e}")
    
    def augment_images(self, augment_train_only: bool = True):
        """
        Аугментация изображений для увеличения датасета.
        
        Args:
            augment_train_only: Аугментировать только тренировочные данные
        """
        if not config.data.augmentation_enabled:
            logger.info("Аугментация отключена в конфигурации")
            return self.manifest_df
        
        logger.info("Начало аугментации изображений...")
        
        # Фильтруем изображения для аугментации
        if augment_train_only:
            images_to_augment = self.manifest_df[self.manifest_df['split'] == 'train']
        else:
            images_to_augment = self.manifest_df
        
        augmented_rows = []
        augmented_count = 0
        
        for _, row in images_to_augment.iterrows():
            src_path = self._resolve_image_path(row)
            
            if not src_path or not src_path.exists():
                continue
            
            # Создаем 2 аугментированные версии
            for aug_idx in range(2):
                aug_image = self._apply_augmentation(src_path, aug_idx)
                if aug_image is not None:
                    # Сохраняем аугментированное изображение
                    split = row.get('split', 'train')
                    recipe_id = str(row.get('recipe_id', 'unknown'))
                    
                    aug_dir = Path(f"data/processed/images/augmented/{split}/{recipe_id}")
                    aug_dir.mkdir(parents=True, exist_ok=True)
                    
                    original_name = src_path.stem
                    aug_filename = f"{original_name}_aug{aug_idx}{src_path.suffix}"
                    aug_path = aug_dir / aug_filename
                    
                    aug_image.save(aug_path)
                    
                    # Добавляем запись в манифест
                    aug_row = row.copy()
                    aug_row['image_path'] = str(aug_path.relative_to(Path("data/processed")))
                    aug_row['is_augmented'] = True
                    aug_row['augmentation_type'] = f'aug{aug_idx}'
                    
                    augmented_rows.append(aug_row)
                    augmented_count += 1
        
        # Добавляем аугментированные строки в манифест
        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            updated_manifest = pd.concat([self.manifest_df, aug_df], ignore_index=True)
            
            # Сохраняем обновленный манифест
            manifest_path = Path("data/processed/metadata/dataset_manifest_augmented.csv")
            updated_manifest.to_csv(manifest_path, index=False, encoding='utf-8')
            
            logger.info(f"Аугментация завершена. Добавлено {augmented_count} изображений")
            logger.info(f"Обновленный манифест: {manifest_path}")
            
            self.manifest_df = updated_manifest
            return updated_manifest
        
        return self.manifest_df
    
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
            
            # Применяем разные аугментации
            if aug_idx == 0:
                # Поворот и сдвиг
                angle = np.random.uniform(-config.data.rotation_range, config.data.rotation_range)
                h, w = img_np.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(img_np, matrix, (w, h))
                
            elif aug_idx == 1:
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
    
    def create_category_mapping(self):
        """
        Создание маппинга категорий рецептов на основе манифеста.
        """
        if 'recipe_type' not in self.manifest_df.columns:
            logger.warning("Колонка 'recipe_type' не найдена в манифесте")
            return None
        
        # Создаем маппинг типов рецептов на категории из конфигурации
        recipe_types = self.manifest_df['recipe_type'].unique()
        
        category_mapping = {}
        for recipe_type in recipe_types:
            # Сопоставляем типы рецептов с категориями из конфигурации
            recipe_type_lower = str(recipe_type).lower()
            
            if any(keyword in recipe_type_lower for keyword in ['терразит', 'terrazit']):
                category = 'Терразит'
            elif any(keyword in recipe_type_lower for keyword in ['шовн', 'shovn']):
                category = 'Шовный'
            elif any(keyword in recipe_type_lower for keyword in ['мастик', 'mastik']):
                category = 'Мастика'
            elif any(keyword in recipe_type_lower for keyword in ['терраццо', 'terratso']):
                category = 'Терраццо'
            elif any(keyword in recipe_type_lower for keyword in ['ретуш', 'retush']):
                category = 'Ретушь'
            else:
                category = 'Терразит'  # По умолчанию
            
            category_mapping[recipe_type] = category
        
        # Сохраняем маппинг
        mapping_path = Path("data/processed/metadata/category_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(category_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Маппинг категорий сохранен: {mapping_path}")
        logger.info(f"Уникальных типов рецептов: {len(category_mapping)}")
        
        return category_mapping
    
    def generate_dataset_report(self):
        """Генерация отчета о датасете."""
        if self.manifest_df.empty:
            logger.warning("Манифест пуст. Отчет не сгенерирован.")
            return
        
        report_path = Path("reports/dataset_preparation_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О ПОДГОТОВКЕ ДАТАСЕТА TERRAZITE AI\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Исходный манифест: {self.manifest_path}\n")
            f.write(f"Всего записей: {len(self.manifest_df)}\n\n")
            
            # Статистика по сплитам
            if 'split' in self.manifest_df.columns:
                f.write("РАСПРЕДЕЛЕНИЕ ПО СПЛИТАМ:\n")
                f.write("-" * 40 + "\n")
                split_counts = self.manifest_df['split'].value_counts()
                for split, count in split_counts.items():
                    percentage = (count / len(self.manifest_df)) * 100
                    f.write(f"{split.upper()}: {count} записей ({percentage:.1f}%)\n")
            
            # Статистика по типам рецептов
            if 'recipe_type' in self.manifest_df.columns:
                f.write("\nТИПЫ РЕЦЕПТОВ:\n")
                f.write("-" * 40 + "\n")
                type_counts = self.manifest_df['recipe_type'].value_counts()
                for recipe_type, count in type_counts.items():
                    percentage = (count / len(self.manifest_df)) * 100
                    f.write(f"{recipe_type}: {count} записей ({percentage:.1f}%)\n")
            
            # Статистика по аугментации
            if 'is_augmented' in self.manifest_df.columns:
                f.write("\nАУГМЕНТАЦИЯ:\n")
                f.write("-" * 40 + "\n")
                augmented_count = self.manifest_df['is_augmented'].sum()
                original_count = len(self.manifest_df) - augmented_count
                f.write(f"Оригинальных: {original_count}\n")
                f.write(f"Аугментированных: {augmented_count}\n")
                f.write(f"Итого: {len(self.manifest_df)}\n")
            
            f.write("\nСТРУКТУРА ДАТАСЕТА:\n")
            f.write("-" * 40 + "\n")
            f.write("data/processed/images/\n")
            f.write("  ├── train/          # Тренировочные данные\n")
            f.write("  ├── val/            # Валидационные данные\n")
            f.write("  ├── test/           # Тестовые данные\n")
            f.write("  └── augmented/      # Аугментированные данные\n")
            f.write("\ndata/processed/metadata/\n")
            f.write("  ├── category_mapping.json  # Маппинг категорий\n")
            f.write("  └── dataset_manifest_augmented.csv  # Полный манифест\n")
        
        logger.info(f"Отчет о датасете сохранен: {report_path}")
        return report_path


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description='Подготовка датасета изображений на основе манифеста')
    parser.add_argument('--manifest', type=str, default=None,
                       help='Путь к CSV манифесту (по умолчанию ищет в data/processed/)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Отключить аугментацию изображений')
    parser.add_argument('--copy-only', action='store_true',
                       help='Только копирование без аугментации')
    parser.add_argument('--create-mapping', action='store_true',
                       help='Создать маппинг категорий')
    
    args = parser.parse_args()
    
    # Настройка конфигурации
    setup_config()
    
    logger.info("Запуск подготовки датасета изображений...")
    
    try:
        # Создаем подготовщик
        preparer = ImageDatasetPreparer(args.manifest)
        
        # Копируем изображения в структурированную директорию
        logger.info("Копирование изображений...")
        copied_count = preparer.copy_images_to_dataset_structure()
        
        if copied_count == 0:
            logger.warning("Изображения не скопированы. Проверьте пути в манифесте.")
        
        # Аугментация
        if not args.copy_only and not args.no_augmentation:
            logger.info("Аугментация изображений...")
            preparer.augment_images(augment_train_only=True)
        
        # Создание маппинга категорий
        if args.create_mapping:
            logger.info("Создание маппинга категорий...")
            preparer.create_category_mapping()
        
        # Генерация отчета
        report_path = preparer.generate_dataset_report()
        
        logger.info("Подготовка датасета завершена!")
        
        # Вывод сводной информации
        print("\n" + "=" * 80)
        print("СВОДКА ПО ДАТАСЕТУ:")
        print("=" * 80)
        
        df = preparer.manifest_df
        print(f"Всего записей в манифесте: {len(df)}")
        
        if 'split' in df.columns:
            print("\nРаспределение по сплитам:")
            for split, count in df['split'].value_counts().items():
                print(f"  {split}: {count}")
        
        if 'recipe_type' in df.columns:
            print("\nУникальных типов рецептов:", df['recipe_type'].nunique())
        
        print(f"\nИзображения скопированы в: data/processed/images/")
        if report_path:
            print(f"Отчет сохранен: {report_path}")
        
        print("\nСледующие шаги:")
        print("1. Проверьте структуру директорий в data/processed/images/")
        print("2. Запустите обучение: python scripts/train_model.py")
        print("3. Используйте create_data_manifest.py для обновления манифеста")
        
    except Exception as e:
        logger.error(f"Ошибка при подготовке датасета: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
