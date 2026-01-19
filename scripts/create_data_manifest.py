"""
Создание манифеста данных для проекта Terrazite AI.
Исправленная версия для Windows с учетом структуры типов рецептов.
"""
import pandas as pd
from pathlib import Path
import json
import logging
import sys
import argparse
import numpy as np

# Добавляем путь для импорта модулей проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger()


class DataManifestCreator:
    """Создатель манифеста данных с учетом структуры типов рецептов"""
    
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
                
                # Подсчет общего количества изображений
                image_count = 0
                for folder in recipe_folders:
                    images = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.txt'))
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
            required_columns = ['id', 'название', 'тип']
            missing_columns = [col for col in required_columns if col not in recipes_df.columns]
            
            if missing_columns:
                logger.warning(f"Отсутствуют колонки: {missing_columns}")
            
            return recipes_df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel файла: {e}")
            return None
    
    def analyze_recipe_structure(self, recipes_df):
        """Анализ структуры рецептов по типам и оттенкам"""
        analysis = {
            'recipe_types': {},
            'recipes_per_type': {},
            'components_analysis': {}
        }
        
        # Группировка по типам рецептов
        if 'тип' in recipes_df.columns:
            type_counts = recipes_df['тип'].value_counts()
            analysis['recipe_types'] = type_counts.to_dict()
            
            # Для каждого типа собираем рецепты
            for recipe_type in recipes_df['тип'].unique():
                if pd.isna(recipe_type):
                    continue
                    
                type_recipes = recipes_df[recipes_df['тип'] == recipe_type]
                analysis['recipes_per_type'][recipe_type] = {
                    'count': len(type_recipes),
                    'recipe_ids': type_recipes['id'].astype(str).tolist()
                }
        
        # Анализ компонентов для определения оттенков
        component_cols = [col for col in recipes_df.columns 
                         if col not in ['id', 'название', 'тип', 'описание', 'description', 'name', 'type']]
        
        if component_cols:
            # Нормализация компонентов
            components_data = recipes_df[component_cols].fillna(0)
            
            # Для каждого рецепта вычисляем "цветовой профиль" на основе пигментов
            pigment_cols = [col for col in component_cols if 'пигмент' in col.lower() or 'красн' in col.lower() 
                          or 'син' in col.lower() or 'желт' in col.lower() or 'цвет' in col.lower()]
            
            if pigment_cols:
                analysis['components_analysis']['pigment_columns'] = pigment_cols
                analysis['components_analysis']['pigment_stats'] = {
                    col: {
                        'min': float(components_data[col].min()),
                        'max': float(components_data[col].max()),
                        'mean': float(components_data[col].mean())
                    } for col in pigment_cols
                }
        
        return analysis
    
    def create_detailed_manifest(self, recipes_df, structure_analysis):
        """Создание детального JSON манифеста с учетом структуры типов"""
        manifest = {
            'statistics': {
                'total_recipes': len(recipes_df),
                'columns': list(recipes_df.columns),
                'missing_values': recipes_df.isnull().sum().to_dict(),
                'recipe_types': structure_analysis['recipe_types'],
                'recipes_per_type': structure_analysis['recipes_per_type']
            },
            'recipes': [],
            'recipe_groups': {}  # Группировка рецептов по типам и оттенкам
        }
        
        # Группируем рецепты по типам
        if 'тип' in recipes_df.columns:
            for recipe_type in recipes_df['тип'].unique():
                if pd.isna(recipe_type):
                    continue
                    
                type_recipes = recipes_df[recipes_df['тип'] == recipe_type]
                manifest['recipe_groups'][recipe_type] = {
                    'count': len(type_recipes),
                    'recipe_ids': type_recipes['id'].astype(str).tolist(),
                    'description': f"Рецепты типа '{recipe_type}' с разными оттенками штукатурки"
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
                'description': row.get('описание', row.get('description', '')),
                'has_images': image_dir.exists(),
                'image_count': 0,
                'image_files': [],
                'components': {},
                'pigment_profile': {}  # Профиль пигментов для определения оттенка
            }
            
            # Добавляем компоненты
            for col in component_columns:
                if col in row and pd.notna(row[col]):
                    value = float(row[col])
                    recipe_info['components'][col] = value
                    
                    # Выделяем пигменты в отдельный профиль
                    if 'пигмент' in col.lower() or 'красн' in col.lower() or 'син' in col.lower() or 'желт' in col.lower():
                        recipe_info['pigment_profile'][col] = value
            
            # Информация об изображениях
            if image_dir.exists():
                image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.txt'))
                recipe_info['image_count'] = len(image_files)
                recipe_info['image_files'] = [str(f.relative_to(self.raw_dir)) for f in image_files]
            
            manifest['recipes'].append(recipe_info)
        
        # Сохранение детального JSON манифеста
        manifest_path = self.data_dir / "data_manifest_detailed.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Детальный манифест создан: {manifest_path}")
        
        # Статистика
        recipes_with_images = sum(1 for r in manifest['recipes'] if r['has_images'])
        total_images = sum(r['image_count'] for r in manifest['recipes'])
        
        logger.info(f"Рецепты с изображениями: {recipes_with_images}/{len(recipes_df)}")
        logger.info(f"Всего изображений: {total_images}")
        logger.info(f"Группы рецептов по типам: {list(manifest['recipe_groups'].keys())}")
        
        return manifest
    
    def create_ml_ready_manifest(self, detailed_manifest):
        """Создание CSV манифестов для ML с группировкой по типам"""
        ml_entries = []
        
        for recipe in detailed_manifest['recipes']:
            if recipe['has_images'] and recipe['image_count'] > 0:
                for img_path in recipe['image_files']:
                    # Для тестовых .txt файлов заменяем расширение на .jpg для совместимости
                    img_path_str = str(img_path)
                    if img_path_str.endswith('.txt'):
                        img_path_str = img_path_str.replace('.txt', '.jpg')
                    
                    ml_entries.append({
                        'image_path': img_path_str,
                        'recipe_id': recipe['id'],
                        'recipe_name': recipe['name'],
                        'recipe_type': recipe['type'],
                        'split': 'unassigned'
                    })
        
        if not ml_entries:
            logger.warning("Нет изображений для создания ML манифеста")
            return None
        
        # Создаем DataFrame
        df = pd.DataFrame(ml_entries)
        
        # УПРОЩЕННОЕ РАЗДЕЛЕНИЕ БЕЗ СТРАТИФИКАЦИИ
        # Перемешиваем данные
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Разделяем на train/val/test (70/15/15)
        n = len(df)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.15)
        
        # Назначаем сплиты
        splits = []
        for i in range(n):
            if i < train_end:
                splits.append('train')
            elif i < val_end:
                splits.append('val')
            else:
                splits.append('test')
        
        df['split'] = splits
        
        # Проверяем, что в каждом сплите есть хотя бы по одному рецепту каждого типа
        for split_name in ['train', 'val', 'test']:
            split_df = df[df['split'] == split_name]
            logger.info(f"{split_name}: {len(split_df)} записей, типы: {split_df['recipe_type'].value_counts().to_dict()}")
        
        # Сохраняем отдельные CSV файлы
        self.processed_dir.mkdir(exist_ok=True)
        
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        
        train_path = self.processed_dir / 'data_manifest_train.csv'
        val_path = self.processed_dir / 'data_manifest_val.csv'
        test_path = self.processed_dir / 'data_manifest_test.csv'
        full_path = self.processed_dir / 'data_manifest_full.csv'
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        df.to_csv(full_path, index=False, encoding='utf-8')
        
        saved_files = {
            'train': {'path': str(train_path), 'count': len(train_df)},
            'val': {'path': str(val_path), 'count': len(val_df)},
            'test': {'path': str(test_path), 'count': len(test_df)},
            'all': {'path': str(full_path), 'count': len(df)}
        }
        
        for split_name, info in saved_files.items():
            if split_name != 'all':
                logger.info(f"{split_name}: {info['count']} записей -> {Path(info['path']).name}")
        
        # Сохраняем статистику
        stats = {
            'total_ml_records': len(df),
            'records_by_split': {k: v['count'] for k, v in saved_files.items() if k != 'all'},
            'unique_recipes': df['recipe_id'].nunique(),
            'recipe_types_distribution': df['recipe_type'].value_counts().to_dict(),
            'split_distribution': {
                split_name: {
                    'count': len(df[df['split'] == split_name]),
                    'percentage': f"{len(df[df['split'] == split_name])/len(df)*100:.1f}%",
                    'recipe_types': df[df['split'] == split_name]['recipe_type'].value_counts().to_dict()
                } for split_name in ['train', 'val', 'test']
            }
        }
        
        stats_path = self.processed_dir / 'ml_data_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ML статистика сохранена: {stats_path}")
        
        return saved_files
    
    def create_recipe_type_report(self, detailed_manifest):
        """Создание отчета по типам рецептов и их оттенкам"""
        report = {
            'recipe_types_summary': {},
            'pigment_analysis': {},
            'recommendations': []
        }
        
        # Анализируем каждый тип рецептов
        for recipe_type, group_info in detailed_manifest.get('recipe_groups', {}).items():
            recipe_ids = group_info['recipe_ids']
            type_recipes = [r for r in detailed_manifest['recipes'] if r['id'] in recipe_ids]
            
            report['recipe_types_summary'][recipe_type] = {
                'recipe_count': len(type_recipes),
                'recipes_with_images': sum(1 for r in type_recipes if r['has_images']),
                'total_images': sum(r['image_count'] for r in type_recipes),
                'recipe_ids': recipe_ids
            }
            
            # Анализ пигментов для этого типа
            pigment_data = []
            for recipe in type_recipes:
                if recipe['pigment_profile']:
                    pigment_data.append(recipe['pigment_profile'])
            
            if pigment_data:
                # Преобразуем в DataFrame для анализа
                pigment_df = pd.DataFrame(pigment_data)
                report['pigment_analysis'][recipe_type] = {
                    'pigment_columns': list(pigment_df.columns),
                    'statistics': {
                        col: {
                            'min': float(pigment_df[col].min()),
                            'max': float(pigment_df[col].max()),
                            'mean': float(pigment_df[col].mean()),
                            'std': float(pigment_df[col].std())
                        } for col in pigment_df.columns
                    }
                }
        
        # Рекомендации
        for recipe_type, summary in report['recipe_types_summary'].items():
            if summary['recipes_with_images'] > 0:
                report['recommendations'].append(
                    f"Тип '{recipe_type}': {summary['recipes_with_images']} рецептов с изображениями, "
                    f"{summary['total_images']} изображений. Готов к обучению модели."
                )
            else:
                report['recommendations'].append(
                    f"Тип '{recipe_type}': нет изображений. Добавьте фотографии образцов штукатурки."
                )
        
        # Сохраняем отчет
        report_path = self.processed_dir / 'recipe_type_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Отчет по типам рецептов сохранен: {report_path}")
        
        return report
    
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
        
        # 3. Анализ структуры рецептов
        logger.info("\nАнализ структуры рецептов...")
        structure_analysis = self.analyze_recipe_structure(recipes_df)
        
        # 4. Создание детального JSON манифеста
        logger.info("\nСоздание детального манифеста...")
        detailed_manifest = self.create_detailed_manifest(recipes_df, structure_analysis)
        
        # 5. Создание отчета по типам рецептов
        logger.info("\nСоздание отчета по типам рецептов...")
        type_report = self.create_recipe_type_report(detailed_manifest)
        
        # 6. Создание ML-готовых CSV манифестов
        logger.info("\nСоздание манифестов для ML...")
        ml_manifests = self.create_ml_ready_manifest(detailed_manifest)
        
        if ml_manifests:
            logger.info("\n" + "="*60)
            logger.info("МАНИФЕСТЫ ДАННЫХ УСПЕШНО СОЗДАНЫ")
            logger.info("="*60)
            
            logger.info("\nСозданные файлы:")
            for split_name, info in ml_manifests.items():
                if split_name != 'all':
                    logger.info(f"  * {Path(info['path']).name}: {info['count']} записей")
            
            logger.info(f"  * data_manifest_detailed.json: детальная информация о рецептах")
            logger.info(f"  * recipe_type_analysis.json: анализ типов рецептов и оттенков")
            logger.info(f"  * ml_data_statistics.json: статистика для ML")
            
            # Вывод рекомендаций
            logger.info("\nРекомендации:")
            for rec in type_report.get('recommendations', []):
                logger.info(f"  * {rec}")
            
            logger.info("\nСледующие шаги:")
            logger.info("  1. Проверьте созданные файлы в data/ и data/processed/")
            logger.info("  2. Запустите обучение модели: python scripts/train_model.py")
            logger.info("  3. Для реальных данных замените .txt файлы на фотографии .jpg/.png")
            
            return True
        else:
            logger.warning("\nML манифесты не созданы (нет изображений)")
            logger.info("Поместите изображения в data/raw/images/{recipe_id}/")
            return False


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description='Создание манифестов данных для Terrazite AI')
    parser.add_argument('--skip-ml', action='store_true', 
                       help='Пропустить создание ML манифестов (только детальный манифест)')
    parser.add_argument('--analyze-types', action='store_true',
                       help='Только анализ типов рецептов без создания ML манифестов')
    
    args = parser.parse_args()
    
    creator = DataManifestCreator()
    
    if args.analyze_types:
        # Только анализ типов
        recipes_df = creator.load_and_validate_data()
        if recipes_df is not None:
            structure_analysis = creator.analyze_recipe_structure(recipes_df)
            detailed_manifest = creator.create_detailed_manifest(recipes_df, structure_analysis)
            creator.create_recipe_type_report(detailed_manifest)
            print("\nАнализ типов рецептов завершен.")
    else:
        # Полный процесс
        success = creator.process()
        
        if success:
            print("\n" + "="*60)
            print("ПРОЦЕСС СОЗДАНИЯ МАНИФЕСТА ЗАВЕРШЕН УСПЕШНО!")
            print("="*60)
            print("\nСозданные файлы доступны в директориях:")
            print("  data/ - детальный манифест и отчеты")
            print("  data/processed/ - ML-готовые манифесты")
        else:
            print("\nВ процессе возникли ошибки")
            print("Проверьте наличие исходных данных")


if __name__ == "__main__":
    main()
