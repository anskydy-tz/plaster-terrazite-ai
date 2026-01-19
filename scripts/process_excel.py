#!/usr/bin/env python3
"""
Скрипт для обработки Excel файла с рецептами терразитовой штукатурки
Интегрирован с ComponentAnalyzer для анализа компонентов по категориям
"""
import sys
import os
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.data.loader import RecipeLoader
from src.data.component_analyzer import ComponentAnalyzer
from src.utils.config import setup_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_excel_file(excel_path: str, 
                      output_dir: str = "data/processed",
                      analyze_components: bool = True,
                      generate_report: bool = True) -> Dict[str, Any]:
    """
    Обработка Excel файла с рецептами
    
    Args:
        excel_path: Путь к Excel файлу
        output_dir: Директория для сохранения результатов
        analyze_components: Анализировать ли компоненты
        generate_report: Генерировать ли отчет
        
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
        
        logger.info(f"Начинаю обработку Excel файла: {excel_path}")
        
        # Создаем анализатор компонентов
        analyzer = ComponentAnalyzer(excel_path)
        
        # Загружаем и анализируем Excel
        logger.info("Загрузка и анализ Excel файла...")
        analyzer.load_excel()
        
        if analyze_components:
            logger.info("Анализ компонентов...")
            analysis_results = analyzer.analyze_components()
            results['analysis'] = {
                'total_recipes': len(analyzer.df),
                'categories': analysis_results['category_stats'],
                'unique_components': len(analyzer.get_component_features()['component_to_idx'])
            }
        
        # Создаем загрузчик рецептов
        loader = RecipeLoader(excel_path, analyzer)
        
        # Получаем все рецепты
        logger.info("Парсинг рецептов...")
        recipes = loader.get_all_recipes()
        
        # Получаем статистику
        stats = loader.get_component_statistics()
        
        # Сохраняем результаты в JSON
        output_path = Path(output_dir) / "recipes_processed.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Подготовка данных для сохранения
        output_data = {
            'metadata': {
                'source_file': excel_path,
                'processing_date': datetime.now().isoformat(),
                'total_recipes': len(recipes),
                'categories': {cat: stats['categories'][cat] for cat in stats['categories']},
                'component_groups': loader.component_groups
            },
            'recipes': []
        }
        
        # Добавляем рецепты
        for recipe in recipes:
            recipe_data = {
                'name': recipe.name,
                'category': recipe.category,
                'components': recipe.components,
                'component_count': len(recipe.components),
                'total_weight': sum(recipe.components.values())
            }
            output_data['recipes'].append(recipe_data)
        
        # Сохраняем JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        results['json_path'] = str(output_path)
        results['total_recipes'] = len(recipes)
        results['categories'] = stats['categories']
        
        # Генерация отчетов
        if generate_report:
            logger.info("Генерация отчетов...")
            
            # Текстовый отчет
            report_path = Path("reports") / f"excel_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ОТЧЕТ ОБ ОБРАБОТКЕ EXCEL ФАЙЛА С РЕЦЕПТАМИ\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Файл: {excel_path}\n")
                f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Всего рецептов: {len(recipes)}\n\n")
                
                f.write("РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:\n")
                f.write("-" * 40 + "\n")
                for category, count in stats['categories'].items():
                    f.write(f"{category}: {count} рецептов ({count/len(recipes)*100:.1f}%)\n")
                
                f.write("\nСТАТИСТИКА КОМПОНЕНТОВ:\n")
                f.write("-" * 40 + "\n")
                
                # Топ-10 наиболее часто используемых компонентов
                component_freq = stats['component_frequency']
                top_components = sorted(component_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                
                f.write("\nТоп-10 наиболее частых компонентов:\n")
                for component, freq in top_components:
                    percentage = (freq / len(recipes)) * 100
                    f.write(f"  {component[:50]}: {freq} рецептов ({percentage:.1f}%)\n")
                
                f.write("\nГРУППЫ КОМПОНЕНТОВ:\n")
                for group_name, components in loader.component_groups.items():
                    group_usage = sum(1 for comp in components if comp in component_freq and component_freq[comp] > 0)
                    if group_usage > 0:
                        f.write(f"  {group_name}: {group_usage} компонентов\n")
                
                f.write("\nПРИМЕРЫ РЕЦЕПТОВ:\n")
                f.write("-" * 40 + "\n")
                for i, recipe in enumerate(recipes[:3]):  # Показываем первые 3 рецепта
                    f.write(f"\n{i+1}. {recipe.name} ({recipe.category})\n")
                    f.write("   Компоненты:\n")
                    for component, value in list(recipe.components.items())[:5]:  # Первые 5 компонентов
                        f.write(f"     - {component}: {value} кг\n")
                    if len(recipe.components) > 5:
                        f.write(f"     ... и еще {len(recipe.components) - 5} компонентов\n")
            
            results['report_path'] = str(report_path)
            
            # Визуализации
            if analyze_components:
                logger.info("Создание визуализаций...")
                viz_path = analyzer.visualize_analysis()
                results['visualization_path'] = str(viz_path)
        
        # Сохранение векторизованных данных для ML
        ml_data_path = Path(output_dir) / "ml_ready_data.json"
        ml_data = {
            'component_mapping': loader.component_features,
            'category_mapping': {cat: idx for idx, cat in enumerate(loader.categories)},
            'recipes_vectorized': []
        }
        
        for recipe in recipes:
            recipe_vector = loader.vectorize_components(recipe.components)
            ml_data['recipes_vectorized'].append({
                'name': recipe.name,
                'category': recipe.category,
                'vector': recipe_vector.tolist(),
                'components': recipe.components
            })
        
        with open(ml_data_path, 'w', encoding='utf-8') as f:
            json.dump(ml_data, f, ensure_ascii=False, indent=2)
        
        results['ml_data_path'] = str(ml_data_path)
        results['success'] = True
        
        logger.info(f"Обработка завершена успешно!")
        logger.info(f"  Обработано рецептов: {len(recipes)}")
        logger.info(f"  Категории: {', '.join(stats['categories'].keys())}")
        logger.info(f"  JSON сохранен: {output_path}")
        
        if 'report_path' in results:
            logger.info(f"  Отчет сохранен: {results['report_path']}")
        
        # Вывод краткой статистики
        print("\n" + "=" * 80)
        print("КРАТКАЯ СТАТИСТИКА:")
        print("=" * 80)
        print(f"Всего рецептов: {len(recipes)}")
        print("\nРаспределение по категориям:")
        for category, count in stats['categories'].items():
            percentage = (count / len(recipes)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\nУникальных компонентов:", len(loader.component_features['component_to_idx']))
        print(f"Данные для ML сохранены: {ml_data_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке Excel файла: {e}")
        results['errors'].append(str(e))
        results['success'] = False
    
    return results


def compare_with_existing_data(new_data_path: str, 
                              existing_data_path: str = "data/processed/recipes_processed.json") -> Dict[str, Any]:
    """
    Сравнение новых данных с существующими
    
    Args:
        new_data_path: Путь к новым данным
        existing_data_path: Путь к существующим данным
        
    Returns:
        Словарь с результатами сравнения
    """
    comparison = {
        'new_recipes': 0,
        'updated_recipes': 0,
        'removed_recipes': 0,
        'changes': []
    }
    
    try:
        if not Path(existing_data_path).exists():
            logger.info(f"Существующие данные не найдены: {existing_data_path}")
            return comparison
        
        # Загружаем данные
        with open(new_data_path, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        
        with open(existing_data_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Создаем словари для сравнения
        new_recipes = {r['name']: r for r in new_data.get('recipes', [])}
        existing_recipes = {r['name']: r for r in existing_data.get('recipes', [])}
        
        # Находим новые рецепты
        new_recipe_names = set(new_recipes.keys()) - set(existing_recipes.keys())
        comparison['new_recipes'] = len(new_recipe_names)
        
        # Находим удаленные рецепты
        removed_recipe_names = set(existing_recipes.keys()) - set(new_recipes.keys())
        comparison['removed_recipes'] = len(removed_recipe_names)
        
        # Проверяем изменения в существующих рецептах
        for name in set(new_recipes.keys()) & set(existing_recipes.keys()):
            new_recipe = new_recipes[name]
            existing_recipe = existing_recipes[name]
            
            # Проверяем изменения в компонентах
            new_components = set(new_recipe.get('components', {}).items())
            existing_components = set(existing_recipe.get('components', {}).items())
            
            if new_components != existing_components:
                comparison['updated_recipes'] += 1
                
                changes = {
                    'recipe': name,
                    'category_changed': new_recipe.get('category') != existing_recipe.get('category'),
                    'component_changes': {
                        'added': dict(new_components - existing_components),
                        'removed': dict(existing_components - new_components)
                    }
                }
                comparison['changes'].append(changes)
        
        logger.info(f"Сравнение завершено:")
        logger.info(f"  Новых рецептов: {comparison['new_recipes']}")
        logger.info(f"  Обновленных рецептов: {comparison['updated_recipes']}")
        logger.info(f"  Удаленных рецептов: {comparison['removed_recipes']}")
        
    except Exception as e:
        logger.error(f"Ошибка при сравнении данных: {e}")
    
    return comparison


def create_sample_dataset(output_path: str = "data/processed/sample_dataset.csv",
                        num_samples: int = 100) -> str:
    """
    Создание выборки для тестирования
    
    Args:
        output_path: Путь для сохранения выборки
        num_samples: Количество образцов
        
    Returns:
        Путь к созданному файлу
    """
    try:
        # Загружаем обработанные данные
        processed_path = Path("data/processed/recipes_processed.json")
        if not processed_path.exists():
            logger.warning("Обработанные данные не найдены. Сначала выполните process_excel_file()")
            return ""
        
        with open(processed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Создаем DataFrame
        samples = []
        for recipe in data['recipes'][:num_samples]:
            sample = {
                'recipe_name': recipe['name'],
                'category': recipe['category'],
                'component_count': recipe['component_count'],
                'total_weight': recipe['total_weight']
            }
            
            # Добавляем топ-5 компонентов
            components = sorted(recipe['components'].items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (component, value) in enumerate(components):
                sample[f'component_{i+1}_name'] = component
                sample[f'component_{i+1}_value'] = value
            
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Выборка создана: {output_path} ({len(df)} записей)")
        return output_path
        
    except Exception as e:
        logger.error(f"Ошибка при создании выборки: {e}")
        return ""


def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description='Обработка Excel файла с рецептами терразитовой штукатурки')
    parser.add_argument('--excel', type=str, default='data/raw/recipes.xlsx',
                       help='Путь к Excel файлу с рецептами (по умолчанию: data/raw/recipes.xlsx)')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Директория для сохранения результатов (по умолчанию: data/processed)')
    parser.add_argument('--no-analyze', action='store_true',
                       help='Не анализировать компоненты')
    parser.add_argument('--no-report', action='store_true',
                       help='Не генерировать отчет')
    parser.add_argument('--compare', action='store_true',
                       help='Сравнить с существующими данными')
    parser.add_argument('--sample', type=int, default=0,
                       help='Создать выборку указанного размера')
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    # Настройка конфигурации
    if args.config:
        setup_config(args.config)
    
    # Обработка Excel файла
    results = process_excel_file(
        excel_path=args.excel,
        output_dir=args.output,
        analyze_components=not args.no_analyze,
        generate_report=not args.no_report
    )
    
    if not results['success']:
        logger.error("Обработка завершилась с ошибками:")
        for error in results.get('errors', []):
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Сравнение с существующими данными
    if args.compare and 'json_path' in results:
        comparison = compare_with_existing_data(results['json_path'])
        
        if comparison['new_recipes'] > 0 or comparison['updated_recipes'] > 0:
            logger.info("Обнаружены изменения в данных:")
            logger.info(f"  Новых рецептов: {comparison['new_recipes']}")
            logger.info(f"  Обновленных рецептов: {comparison['updated_recipes']}")
            
            # Сохраняем отчет об изменениях
            changes_path = Path(args.output) / f"changes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(changes_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            logger.info(f"Отчет об изменениях сохранен: {changes_path}")
    
    # Создание выборки
    if args.sample > 0:
        sample_path = Path(args.output) / f"sample_dataset_{args.sample}.csv"
        create_sample_dataset(str(sample_path), args.sample)
    
    # Обновление конфигурации на основе анализа
    try:
        from src.utils.config import config
        config.update_from_excel(args.excel)
        logger.info("Конфигурация обновлена на основе анализа Excel файла")
    except Exception as e:
        logger.warning(f"Не удалось обновить конфигурацию: {e}")
    
    print("\n" + "=" * 80)
    print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 80)
    print(f"Исходный файл: {results['excel_path']}")
    print(f"Обработано рецептов: {results['total_recipes']}")
    print(f"Категории: {', '.join(results['categories'].keys())}")
    
    if 'json_path' in results:
        print(f"Данные сохранены: {results['json_path']}")
    
    if 'ml_data_path' in results:
        print(f"Данные для ML: {results['ml_data_path']}")
    
    if 'report_path' in results:
        print(f"Отчет: {results['report_path']}")
    
    print("\nСледующие шаги:")
    print("1. Соберите фотографии образцов для каждого рецепта")
    print("2. Разместите их в data/raw/images/{recipe_name}/")
    print("3. Запустите обучение модели: python scripts/train_model.py")
    print("4. Запустите API сервер: uvicorn src.api.main:app --reload")
    print("5. Откройте интерфейс: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
