"""
Обработчик Excel файла с рецептами терразитовой штукатурки
Преобразует Excel в структурированный JSON формат
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Добавляем путь для импорта модулей проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import RecipeLoader


class ExcelProcessor:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = None
        self.processed_data = []
        self.recipe_loader = None
        
    def load_excel(self):
        """Загрузка Excel файла с использованием RecipeLoader"""
        try:
            # Используем новый RecipeLoader для загрузки
            self.recipe_loader = RecipeLoader(self.excel_path)
            self.df = self.recipe_loader.load_excel()
            print(f"✅ Загружен файл: {os.path.basename(self.excel_path)}")
            print(f"📊 Размер данных: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
            
            # Парсим компоненты
            components = self.recipe_loader.parse_components()
            print(f"🔧 Распарсено {len(components)} рецептов с компонентами")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки Excel: {e}")
            return False
    
    def clean_column_names(self, df):
        """Очистка названий столбцов"""
        df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
        return df
    
    def extract_recipes(self):
        """Извлечение рецептов из Excel с использованием RecipeLoader"""
        if self.recipe_loader is None or self.recipe_loader.components is None:
            print("❌ Данные не загружены. Сначала вызовите load_excel()")
            return []
        
        recipes = []
        components = self.recipe_loader.components
        
        for recipe_id, comp_dict in components.items():
            # Находим строку в DataFrame
            recipe_row = self.df[self.df['id'].astype(str) == str(recipe_id)]
            
            if recipe_row.empty:
                print(f"⚠️  Не найдена строка для id: {recipe_id}")
                continue
            
            recipe_row = recipe_row.iloc[0]
            
            recipe = {
                'id': str(recipe_id),
                'name': recipe_row.get('Название', ''),
                'type': recipe_row.get('Тип', 'unknown'),
                'components': comp_dict,
                'total_weight': sum(comp_dict.values())
            }
            
            recipes.append(recipe)
        
        print(f"✅ Извлечено {len(recipes)} рецептов")
        return recipes
    
    def analyze_recipes(self, recipes):
        """Анализ рецептов"""
        analysis = {
            'total_recipes': len(recipes),
            'types_count': {},
            'component_stats': {}
        }
        
        # Подсчет типов рецептов
        types = [r['type'] for r in recipes]
        type_counts = pd.Series(types).value_counts()
        analysis['types_count'] = type_counts.to_dict()
        
        # Анализ компонентов
        all_components = set()
        component_totals = {}
        
        for recipe in recipes:
            all_components.update(recipe['components'].keys())
            for component, value in recipe['components'].items():
                component_totals[component] = component_totals.get(component, 0) + value
        
        analysis['unique_components'] = len(all_components)
        analysis['component_totals'] = dict(sorted(
            component_totals.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])  # Только топ-10
        
        print(f"📊 Всего уникальных компонентов: {len(all_components)}")
        print(f"📊 Типы рецептов: {analysis['types_count']}")
        
        return analysis
    
    def save_to_json(self, recipes, output_path):
        """Сохранение в JSON с использованием RecipeLoader"""
        try:
            # Используем метод RecipeLoader для сохранения
            self.recipe_loader.save_to_json(output_path)
            print(f"💾 Данные сохранены: {output_path}")
            
            # Создаем summary
            summary = {
                'total_recipes': len(recipes),
                'components_count': len(recipes[0]['components']) if recipes else 0,
                'file_size': os.path.getsize(output_path)
            }
            
            return summary
        except Exception as e:
            print(f"❌ Ошибка сохранения JSON: {e}")
            return None
    
    def create_visualization(self, recipes, output_dir):
        """Создание визуализаций"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Распределение типов рецептов
        types = [r['type'] for r in recipes]
        type_counts = pd.Series(types).value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Круговая диаграмма типов
        axes[0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0].set_title('Распределение типов рецептов')
        
        # 2. Топ-10 компонентов
        component_totals = {}
        for recipe in recipes:
            for component, value in recipe['components'].items():
                if value > 0:
                    component_totals[component] = component_totals.get(component, 0) + value
        
        # Сортируем по общему весу
        top_components = sorted(component_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        components_names = [c[0][:20] + '...' if len(c[0]) > 20 else c[0] for c in top_components]
        components_values = [c[1] for c in top_components]
        
        axes[1].barh(components_names, components_values)
        axes[1].set_xlabel('Общий вес (кг)')
        axes[1].set_title('Топ-10 компонентов по использованию')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'recipe_analysis.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Визуализации сохранены: {viz_path}")
        
        return viz_path
    
    def process(self, output_dir='data/processed'):
        """Основной процесс обработки"""
        print("=" * 50)
        print("🔄 НАЧАЛО ОБРАБОТКИ EXCEL ФАЙЛА")
        print("=" * 50)
        
        # 1. Загрузка с использованием RecipeLoader
        if not self.load_excel():
            return None
        
        # 2. Извлечение рецептов
        recipes = self.extract_recipes()
        if not recipes:
            return None
        
        # 3. Анализ
        analysis = self.analyze_recipes(recipes)
        
        # 4. Сохранение с использованием RecipeLoader
        json_path = os.path.join(output_dir, 'recipes.json')
        summary = self.save_to_json(recipes, json_path)
        
        if not summary:
            return None
        
        # 5. Визуализация
        viz_path = self.create_visualization(recipes, output_dir)
        
        print("\n" + "=" * 50)
        print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 50)
        print(f"📁 Рецептов обработано: {summary['total_recipes']}")
        print(f"📁 Компонентов в каждом: {summary['components_count']}")
        print(f"💾 Размер файла: {summary['file_size'] / 1024:.1f} KB")
        print(f"📈 Визуализации: {viz_path}")
        
        return {
            'recipes': recipes,
            'json_path': json_path,
            'viz_path': viz_path,
            'analysis': analysis
        }


def main():
    """Точка входа"""
    import sys
    
    # Определяем путь к Excel файлу
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    else:
        # Ищем Excel файл в data/raw/
        raw_dir = Path('data/raw')
        excel_files = list(raw_dir.glob('*.xlsx')) + list(raw_dir.glob('*.xls'))
        
        if not excel_files:
            print("❌ Не найден Excel файл в data/raw/")
            print("📂 Поместите файл с рецептами в папку data/raw/")
            print("📂 Или укажите путь к файлу как аргумент командной строки")
            return
        
        excel_path = excel_files[0]
    
    print(f"📄 Используется файл: {excel_path}")
    
    # Создаем процессор и обрабатываем
    processor = ExcelProcessor(excel_path)
    result = processor.process()
    
    if result:
        print("\n🎉 Файл успешно обработан!")
        print(f"📄 JSON файл: {result['json_path']}")
        print(f"📊 Графики: {result['viz_path']}")
        
        # Сохраняем также через RecipeLoader для ML пайплайна
        try:
            # Создаем новый RecipeLoader для полного пайплайна
            recipe_loader = RecipeLoader(excel_path)
            recipe_loader.process_pipeline(output_path=result['json_path'])
            print("🔧 Рецепты также сохранены через RecipeLoader для ML пайплайна")
        except Exception as e:
            print(f"⚠️  Дополнительное сохранение не удалось: {e}")


if __name__ == "__main__":
    main()
