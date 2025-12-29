"""
Обработчик Excel файла с рецептами терразитовой штукатурки
Преобразует Excel в структурированный JSON формат
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

class ExcelProcessor:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = None
        self.processed_data = []
        
    def load_excel(self):
        """Загрузка Excel файла"""
        try:
            self.df = pd.read_excel(self.excel_path, sheet_name=0)
            print(f"✅ Загружен файл: {os.path.basename(self.excel_path)}")
            print(f"📊 Размер данных: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки Excel: {e}")
            return False
    
    def clean_column_names(self, df):
        """Очистка названий столбцов"""
        df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
        return df
    
    def extract_recipes(self):
        """Извлечение рецептов из Excel"""
        if self.df is None:
            return []
        
        # Очищаем названия столбцов
        self.df = self.clean_column_names(self.df)
        
        # Находим столбец с названиями рецептов
        recipe_col = None
        for col in self.df.columns:
            if 'наименование' in col.lower() or 'рецепт' in col.lower():
                recipe_col = col
                break
        
        if recipe_col is None:
            print("❌ Не найден столбец с названиями рецептов")
            return []
        
        recipes = []
        
        for idx, row in self.df.iterrows():
            recipe_name = row[recipe_col]
            
            # Пропускаем пустые строки
            if pd.isna(recipe_name):
                continue
            
            recipe = {
                'id': f"REC_{idx:03d}",
                'name': str(recipe_name).strip(),
                'type': 'терразит' if 'терразит' in str(recipe_name).lower() else 'шовный',
                'components': {},
                'total_weight': 1000  # Все рецепты на 1000 кг
            }
            
            # Извлекаем компоненты
            for col in self.df.columns:
                if col == recipe_col:
                    continue
                
                value = row[col]
                if pd.isna(value):
                    value = 0
                
                # Преобразуем в число
                try:
                    value = float(value)
                except:
                    value = 0
                
                recipe['components'][col] = value
            
            recipes.append(recipe)
        
        print(f"✅ Извлечено {len(recipes)} рецептов")
        return recipes
    
    def analyze_recipes(self, recipes):
        """Анализ рецептов"""
        analysis = {
            'total_recipes': len(recipes),
            'terrazite_count': sum(1 for r in recipes if r['type'] == 'терразит'),
            'seam_count': sum(1 for r in recipes if r['type'] == 'шовный'),
            'component_stats': {}
        }
        
        # Анализ компонентов
        all_components = set()
        for recipe in recipes:
            all_components.update(recipe['components'].keys())
        
        print(f"📊 Всего уникальных компонентов: {len(all_components)}")
        
        return analysis
    
    def save_to_json(self, recipes, output_path):
        """Сохранение в JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Данные сохранены: {output_path}")
        
        # Создаем summary
        summary = {
            'total_recipes': len(recipes),
            'components_count': len(recipes[0]['components']) if recipes else 0,
            'file_size': os.path.getsize(output_path)
        }
        
        return summary
    
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
        
        # 1. Загрузка
        if not self.load_excel():
            return None
        
        # 2. Извлечение рецептов
        recipes = self.extract_recipes()
        if not recipes:
            return None
        
        # 3. Анализ
        analysis = self.analyze_recipes(recipes)
        
        # 4. Сохранение
        json_path = os.path.join(output_dir, 'recipes.json')
        summary = self.save_to_json(recipes, json_path)
        
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
            print("📂 Поместите файл 'Рецептуры терразит.xlsx' в папку data/raw/")
            return
        
        excel_path = excel_files[0]
    
    # Создаем процессор и обрабатываем
    processor = ExcelProcessor(excel_path)
    result = processor.process()
    
    if result:
        print("\n🎉 Файл успешно обработан!")
        print(f"📄 JSON файл: {result['json_path']}")
        print(f"📊 Графики: {result['viz_path']}")

if __name__ == "__main__":
    main()
