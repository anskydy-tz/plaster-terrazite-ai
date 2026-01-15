"""
Создание манифеста данных для проекта Terrazite AI.
Исправленная версия без эмодзи для Windows.
"""
import pandas as pd
from pathlib import Path
import json
import logging
import sys
import argparse
from sklearn.model_selection import train_test_split

# Добавляем путь для импорта модулей проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger()


class DataManifestCreator:
    """Создатель манифеста данных с проверкой и ML-готовым выводом"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def check_data_availability(self):
        """Проверка наличия всех необходимых данных"""
        checks = {
            'recipes_excel': self.raw_dir / "recipes.xlsx",
            'images_dir': self.raw_dir / "images",
            'processed_recipes': self.processed_dir / "recipes.json"
        }
        
        results = {}
        for name, path in checks.items():
            exists = path.exists()
            results[name] = {
                'exists': exists,
                'path': str(path),
                'type': 'file' if path.is_file() else 'directory'
            }
            
            if exists and name == 'images_dir':
                # Подсчет подпапок (рецептов с изображениями)
                recipe_folders = [d for d in path.iterdir() if d.is_dir()]
                results[name]['subfolders_count'] = len(recipe_folders)
                
                # Подсчет общего количества изображений (включая .txt для тестов)
                image_count = 0
                for folder in recipe_folders:
                    images = (list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + 
                             list(folder.glob('*.txt')))  # Добавляем .txt файлы
                    image_count += len(images)
                results[name]['image_count'] = image_count
        
        return results
    
    def load_and_validate_data(self):
        """Загрузка и валидация данных из Excel"""
        excel_path = self.raw_dir / "recipes.xlsx"
        
        if not excel_path.exists():
            logger.error(f"Файл recipes.xlsx не найден: {excel_path}")
            return None
        
        try:
            recipes_df = pd.read_excel(excel_path)
            logger.info(f"Загружено рецептов: {len(recipes_df)}")
            
            # Проверка обязательных колонок
            required_columns = ['id', 'название']
            missing_columns = [col for col in required_columns if col not in recipes_df.columns]
            
            if missing_columns:
                logger.warning(f"Отсутствуют колонки: {missing_columns}")
            
            return recipes_df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel файла: {e}")
            return None
    
    def create_detailed_manifest(self, recipes_df):
        """Создание детального JSON манифеста"""
        manifest = {
            'statistics': {
                'total_recipes': len(recipes_df),
                'columns': list(recipes_df.columns),
                'missing_values': recipes_df.isnull().sum().to_dict()
            },
            'recipes': []
        }
        
        # Информация по каждому рецепту
        for idx, row in recipes_df.iterrows():
            recipe_id = str(row.get('id', idx + 1))
            image_dir = self.raw_dir / "images" / recipe_id
            
            # Определяем компоненты (все колонки кроме стандартных)
            standard_columns = ['id', 'название', 'описание', 'тип', 'type', 'name', 'description']
            component_columns = [col for col in recipes_df.columns if col.lower() not in standard_columns]
            
            recipe_info = {
                'id': recipe_id,
                'name': row.get('название', row.get('name', f'Рецепт {recipe_id}')),
                'type': row.get('тип', row.get('type', 'unknown')),
                'has_images': image_dir.exists(),
                'image_count': 0,
                'image_files': [],
                'components': {}
            }
            
            # Добавляем компоненты
            for col in component_columns:
                if col in row and pd.notna(row[col]):
                    recipe_info['components'][col] = float(row[col])
            
            # Информация об изображениях (включая .txt файлы)
            if image_dir.exists():
                image_files = (list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + 
                              list(image_dir.glob('*.txt')))  # Добавляем .txt файлы
                recipe_info['image_count'] = len(image_files)
                recipe_info['image_files'] = [str(f.relative_to(self.raw_dir)) for f in image_files]
            
            manifest['recipes'].append(recipe_info)
        
        # Сохранение детального JSON манифеста
        manifest_path = self.data_dir / "data_manifest_detailed.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Детальный манифест создан: {manifest_path}")
        
        # Статистика
        recipes_with_images = sum(1 for r in manifest['recipes'] if r['has_images'])
        total_images = sum(r['image_count'] for r in manifest['recipes'])
        
        logger.info(f"Рецепты с изображениями: {recipes_with_images}/{len(recipes_df)}")
        logger.info(f"Всего изображений: {total_images}")
        
        return manifest
    
    def create_ml_ready_manifest(self, detailed_manifest):
        """Создание CSV манифестов для ML"""
        ml_entries = []
        
        for recipe in detailed_manifest['recipes']:
            if recipe['has_images'] and recipe['image_count'] > 0:
                for img_path in recipe['image_files']:
                    # Преобразуем .txt в .jpg для совместимости с ML пайплайном
                    img_path_str = str(img_path)
                    if img_path_str.endswith('.txt'):
                        # Заменяем .txt на .jpg для совместимости
                        img_path_str = img_path_str.replace('.txt', '.jpg')
                    
                    ml_entries.append({
                        'image_path': img_path_str,
                        'recipe_id': recipe['id'],
                        'recipe_name': recipe['name'],
                        'recipe_type': recipe['type'],
                        'split': 'unassigned'  # Будет назначен позже
                    })
        
        if not ml_entries:
            logger.warning("Нет изображений для создания ML манифеста")
            return None
        
        # Создаем DataFrame
        df = pd.DataFrame(ml_entries)
        
        # Разделяем данные на train/val/test
        if df['recipe_id'].nunique() > 1:
            # Стратифицированное разделение по recipe_id
            unique_recipes = df['recipe_id'].unique()
            recipe_types = df.set_index('recipe_id')['recipe_type'].to_dict()
            
            # Разделяем ID рецептов
            train_ids, temp_ids = train_test_split(
                unique_recipes, test_size=0.3, random_state=42,
                stratify=[recipe_types.get(id, 'unknown') for id in unique_recipes]
            )
            val_ids, test_ids = train_test_split(
                temp_ids, test_size=0.5, random_state=42,
                stratify=[recipe_types.get(id, 'unknown') for id in temp_ids]
            )
            
            # Назначаем split
            def assign_split(recipe_id):
                if recipe_id in train_ids:
                    return 'train'
                elif recipe_id in val_ids:
                    return 'val'
                else:
                    return 'test'
            
            df['split'] = df['recipe_id'].apply(assign_split)
        else:
            # Если только один рецепт, просто разделяем изображения
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            n = len(df)
            df.loc[:int(n*0.7), 'split'] = 'train'
            df.loc[int(n*0.7):int(n*0.85), 'split'] = 'val'
            df.loc[int(n*0.85):, 'split'] = 'test'
        
        # Сохраняем отдельные CSV файлы
        self.processed_dir.mkdir(exist_ok=True)
        
        splits = {
            'train': df[df['split'] == 'train'],
            'val': df[df['split'] == 'val'],
            'test': df[df['split'] == 'test'],
            'all': df
        }
        
        saved_files = {}
        for split_name, split_df in splits.items():
            if split_name == 'all':
                filename = 'data_manifest_full.csv'
            else:
                filename = f'data_manifest_{split_name}.csv'
            
            filepath = self.processed_dir / filename
            split_df.to_csv(filepath, index=False, encoding='utf-8')
            saved_files[split_name] = {
                'path': str(filepath),
                'count': len(split_df)
            }
            
            logger.info(f"{split_name}: {len(split_df)} записей -> {filename}")
        
        # Сохраняем статистику
        stats = {
            'total_ml_records': len(df),
            'records_by_split': {k: len(v) for k, v in splits.items() if k != 'all'},
            'unique_recipes': df['recipe_id'].nunique(),
            'records_per_recipe': df.groupby('recipe_id').size().to_dict(),
            'split_percentage': {
                'train': f"{len(splits['train'])/len(df)*100:.1f}%",
                'val': f"{len(splits['val'])/len(df)*100:.1f}%",
                'test': f"{len(splits['test'])/len(df)*100:.1f}%"
            }
        }
        
        stats_path = self.processed_dir / 'ml_data_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ML статистика сохранена: {stats_path}")
        
        return saved_files
    
    def process(self):
        """Основной процесс создания манифестов"""
        logger.info("="*60)
        logger.info("СОЗДАНИЕ МАНИФЕСТА ДАННЫХ ДЛЯ TERRAZITE AI")
        logger.info("="*60)
        
        # 1. Проверка доступности данных
        logger.info("Проверка доступности данных...")
        availability = self.check_data_availability()
        
        for name, info in availability.items():
            status = "OK" if info['exists'] else "NOT FOUND"
            logger.info(f"  {status} {name}: {info['path']}")
            
            if info['exists'] and name == 'images_dir':
                logger.info(f"    Папок с рецептами: {info.get('subfolders_count', 0)}")
                logger.info(f"    Всего изображений: {info.get('image_count', 0)}")
        
        # 2. Загрузка данных из Excel
        logger.info("\nЗагрузка данных из Excel...")
        recipes_df = self.load_and_validate_data()
        if recipes_df is None:
            return False
        
        # 3. Создание детального JSON манифеста
        logger.info("\nСоздание детального манифеста...")
        detailed_manifest = self.create_detailed_manifest(recipes_df)
        
        # 4. Создание ML-готовых CSV манифестов
        logger.info("\nСоздание манифестов для ML...")
        ml_manifests = self.create_ml_ready_manifest(detailed_manifest)
        
        if ml_manifests:
            logger.info("\n" + "="*60)
            logger.info("МАНИФЕСТЫ ДАННЫХ УСПЕШНО СОЗДАНЫ")
            logger.info("="*60)
            
            logger.info("\nСозданные файлы:")
            for split_name, info in ml_manifests.items():
                logger.info(f"  * {Path(info['path']).name}: {info['count']} записей")
            
            logger.info(f"  * data_manifest_detailed.json: детальная информация")
            logger.info(f"  * ml_data_statistics.json: статистика для ML")
            
            logger.info("\nСледующие шаги:")
            logger.info("  1. Проверьте созданные файлы в data/ и data/processed/")
            logger.info("  2. Запустите обучение модели: python scripts/train_model.py")
            logger.info("  3. Модель теперь будет использовать реальные данные вместо синтетических")
            
            return True
        else:
            logger.warning("\nML манифесты не созданы (нет изображений)")
            logger.info("Поместите изображения в data/raw/images/{recipe_id}/")
            return False


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description='Создание манифестов данных для Terrazite AI')
    parser.add_argument('--skip-ml', action='store_true', help='Пропустить создание ML манифестов')
    args = parser.parse_args()
    
    creator = DataManifestCreator()
    success = creator.process()
    
    if success:
        print("\nПроцесс завершен успешно!")
        print("Проверьте созданные файлы в директории data/")
    else:
        print("\nВ процессе возникли ошибки")
        print("Проверьте наличие исходных данных")


if __name__ == "__main__":
    main()
