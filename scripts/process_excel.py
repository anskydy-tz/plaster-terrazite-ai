#!/usr/bin/env python3
"""
Скрипт для первичной обработки Excel файла с рецептами терразитовой штукатурки.
Создает базовые JSON файлы и маппинги для дальнейшей обработки скриптами create_data_manifest.py и prepare_image_dataset.py.
"""
import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from src.data.loader import RecipeLoader
from src.utils.config import setup_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_excel_file(excel_path: str, 
                      output_dir: str = "data/processed",
                      config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Базовая обработка Excel файла с рецептами.
    
    Args:
        excel_path: Путь к Excel файлу
        output_dir: Директория для сохранения результатов
        config_path: Путь к файлу конфигурации (опционально)
        
    Returns:
        Словарь с результатами обработки
    """
    results = {
        'excel_path': excel_path,
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'errors': []
    }
    
    try:
        # Проверяем существование файла
        if not Path(excel_path).exists():
            raise FileNotFoundError(f"Excel файл не найден: {excel_path}")
        
        logger.info("=" * 60)
        logger.info("НАЧАЛО ОБРАБОТКИ EXCEL ФАЙЛА")
        logger.info("=" * 60)
        logger.info(f"Файл: {excel_path}")
        
        # Настройка конфигурации
        if config_path:
            setup_config(config_path)
            logger.info(f"Конфигурация загружена: {config_path}")
        
        # Создаем загрузчик рецептов
        logger.info("\n📂 Загрузка данных из Excel...")
        loader = RecipeLoader(excel_path)
        
        # Загружаем Excel
        df = loader.load_excel()
        logger.info(f"  Загружено рецептов: {len(df)}")
        
        # Получаем все рецепты
        logger.info("\n🔍 Парсинг рецептов...")
        recipes = loader.get_all_recipes()
        logger.info(f"  Успешно распарсено: {len(recipes)}")
        
        # Получаем статистику
        logger.info("\n📊 Анализ статистики...")
        stats = loader.get_component_statistics()
        
        # Получаем информацию о компонентах
        component_features = loader.component_features
        unique_components = component_features.get('total_components', 0)
        component_groups = component_features.get('component_groups', {})
        
        logger.info(f"  Уникальных компонентов: {unique_components}")
        logger.info(f"  Групп компонентов: {len(component_groups)}")
        
        # Распределение по категориям
        category_stats = stats.get('categories', {})
        logger.info("\n📈 Распределение по категориям:")
        for category, count in category_stats.items():
            percentage = (count / len(recipes)) * 100 if recipes else 0
            logger.info(f"  • {category}: {count} ({percentage:.1f}%)")
        
        # Сохраняем результаты
        output_path = save_processed_data(
            recipes, stats, component_features, 
            loader.categories, output_dir, excel_path
        )
        
        results['output_path'] = str(output_path)
        results['total_recipes'] = len(recipes)
        results['categories'] = category_stats
        results['unique_components'] = unique_components
        results['component_groups'] = list(component_groups.keys())
        results['success'] = True
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке: {e}")
        results['errors'].append(str(e))
        results['success'] = False
    
    return results


def save_processed_data(recipes: list, 
                       stats: Dict[str, Any],
                       component_features: Dict[str, Any],
                       categories: list,
                       output_dir: str,
                       excel_path: str) -> Path:
    """
    Сохранение обработанных данных в JSON.
    
    Args:
        recipes: Список рецептов
        stats: Статистика
        component_features: Информация о компонентах
        categories: Список категорий
        output_dir: Директория для сохранения
        excel_path: Путь к исходному Excel
        
    Returns:
        Путь к сохраненному файлу
    """
    output_path = Path(output_dir) / "recipes_processed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Подготовка данных для сохранения
    output_data = {
        'metadata': {
            'source_file': excel_path,
            'processing_date': datetime.now().isoformat(),
            'total_recipes': len(recipes),
            'categories': categories,
            'unique_components': component_features.get('total_components', 0),
            'component_groups': list(component_features.get('component_groups', {}).keys()),
            'category_distribution': stats.get('categories', {})
        },
        'recipes': [],
        'component_mapping': {
            'component_to_idx': component_features.get('component_to_idx', {}),
            'idx_to_component': component_features.get('idx_to_component', {}),
            'component_groups': component_features.get('component_groups', {})
        }
    }
    
    # Добавляем рецепты (только базовую информацию)
    for recipe in recipes:
        recipe_data = {
            'name': recipe.name,
            'category': recipe.category,
            'component_count': len(recipe.components),
            'total_weight': round(sum(recipe.components.values()), 2),
            'components': recipe.components  # Полный словарь компонентов
        }
        output_data['recipes'].append(recipe_data)
    
    # Сохраняем JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n💾 Данные сохранены: {output_path}")
    
    # Также сохраняем отдельно маппинг компонентов для удобства
    mapping_path = Path(output_dir) / "component_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(component_features.get('idx_to_component', {}), f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Маппинг компонентов: {mapping_path}")
    
    return output_path


def print_summary(results: Dict[str, Any]) -> None:
    """
    Вывод краткой сводки результатов.
    
    Args:
        results: Результаты обработки
    """
    print("\n" + "=" * 60)
    print("📋 СВОДКА ПО ОБРАБОТКЕ")
    print("=" * 60)
    
    if not results['success']:
        print("\n❌ ОБРАБОТКА ЗАВЕРШИЛАСЬ С ОШИБКАМИ")
        for error in results.get('errors', []):
            print(f"  • {error}")
        return
    
    print(f"\n✅ Статус: УСПЕШНО")
    print(f"📁 Исходный файл: {results['excel_path']}")
    print(f"📊 Всего рецептов: {results['total_recipes']}")
    print(f"🔢 Уникальных компонентов: {results['unique_components']}")
    print(f"📦 Групп компонентов: {len(results['component_groups'])}")
    
    print("\n📈 Распределение по категориям:")
    for category, count in results.get('categories', {}).items():
        percentage = (count / results['total_recipes']) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {category:12} {count:3} ({percentage:5.1f}%) {bar}")
    
    print(f"\n💾 Результаты сохранены: {results['output_path']}")


def print_next_steps() -> None:
    """Вывод следующих шагов."""
    print("\n" + "=" * 60)
    print("🎯 СЛЕДУЮЩИЕ ШАГИ")
    print("=" * 60)
    
    print("\n1️⃣  СОЗДАНИЕ МАНИФЕСТОВ ДАННЫХ:")
    print("   python scripts/create_data_manifest.py")
    print("   → Создает train/val/test манифесты на основе рецептов")
    
    print("\n2️⃣  ПОДГОТОВКА ДАТАСЕТА:")
    print("   python scripts/prepare_image_dataset.py --create-mapping")
    print("   → Копирует и аугментирует изображения, создает структуру датасета")
    
    print("\n3️⃣  ОБУЧЕНИЕ МОДЕЛИ:")
    print("   python scripts/train_model.py --plot")
    print("   → Запускает обучение с визуализацией")
    
    print("\n4️⃣  ТЕСТИРОВАНИЕ:")
    print("   python test_model_basic.py")
    print("   python test_full_pipeline.py")
    
    print("\n5️⃣  ЗАПУСК СИСТЕМЫ:")
    print("   # API сервер")
    print("   uvicorn src.api.main:app --reload")
    print("   # Веб-интерфейс")
    print("   streamlit run streamlit_app.py")
    
    print("\n📌 Для быстрого тестового прогона:")
    print("   python scripts/process_excel.py --quick")


def create_sample_manifest(results: Dict[str, Any], output_dir: str) -> None:
    """
    Создание простого CSV манифеста для совместимости с prepare_image_dataset.py.
    
    Args:
        results: Результаты обработки
        output_dir: Директория для сохранения
    """
    try:
        import pandas as pd
        
        # Загружаем обработанные данные
        with open(results['output_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Создаем простой манифест
        manifest_data = []
        for recipe in data['recipes']:
            # Добавляем запись для рецепта (без изображений)
            manifest_data.append({
                'recipe_id': hash(recipe['name']) % 10000,
                'recipe_name': recipe['name'],
                'recipe_type': recipe['category'],
                'split': 'train'  # По умолчанию все в train
            })
        
        if manifest_data:
            df = pd.DataFrame(manifest_data)
            manifest_path = Path(output_dir) / "basic_recipe_manifest.csv"
            df.to_csv(manifest_path, index=False, encoding='utf-8')
            logger.info(f"📋 Базовый манифест создан: {manifest_path}")
            
    except Exception as e:
        logger.debug(f"Не удалось создать базовый манифест: {e}")


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description='Первичная обработка Excel файла с рецептами терразитовой штукатурки',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/process_excel.py                          # Обработка файла по умолчанию
  python scripts/process_excel.py --excel my_recipes.xlsx # Другой файл
  python scripts/process_excel.py --output ./my_data      # Другая директория
  python scripts/process_excel.py --quick                  # Быстрый тестовый прогон
  python scripts/process_excel.py --no-summary             # Без вывода сводки
        """
    )
    
    parser.add_argument('--excel', type=str, default='data/raw/recipes.xlsx',
                       help='Путь к Excel файлу с рецептами (по умолчанию: data/raw/recipes.xlsx)')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Директория для сохранения результатов (по умолчанию: data/processed)')
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к файлу конфигурации (опционально)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Не выводить сводку')
    parser.add_argument('--quick', action='store_true',
                       help='Быстрый тестовый прогон (без полного анализа)')
    
    args = parser.parse_args()
    
    # Быстрый тестовый прогон
    if args.quick:
        logger.info("⚡ Режим быстрого тестирования")
        excel_path = args.excel
        if not Path(excel_path).exists():
            logger.warning(f"Файл {excel_path} не найден, создаем тестовые данные...")
            # Пытаемся создать тестовый Excel если его нет
            try:
                from create_test_excel import create_test_excel
                test_excel_path = create_test_excel()
                if test_excel_path:
                    excel_path = test_excel_path
                    logger.info(f"✅ Создан тестовый файл: {excel_path}")
            except ImportError:
                logger.warning("Не удалось создать тестовый файл")
    
    # Обработка файла
    results = process_excel_file(
        excel_path=args.excel,
        output_dir=args.output,
        config_path=args.config
    )
    
    # Вывод сводки
    if not args.no_summary:
        print_summary(results)
    
    # Создание базового манифеста
    if results['success']:
        create_sample_manifest(results, args.output)
    
    # Вывод следующих шагов
    if results['success'] and not args.no_summary:
        print_next_steps()
    
    # Возвращаем код завершения
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
