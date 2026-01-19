"""
Анализатор компонентов терразитовых составов на основе реальной базы рецептов
"""
import pandas as pd
import json
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple
import numpy as np

class ComponentAnalyzer:
    """
    Класс для анализа компонентов из базы рецептов Excel
    """
    
    # Категории составов на основе анализа Excel
    RECIPE_CATEGORIES = {
        'Терразит': ['Терразит'],
        'Шовный': ['Шовный', 'TZ Шовный'],
        'Мастика': ['Мастика'],
        'Терраццо': ['Терраццо'],
        'Ретушь': ['Ретушь']
    }
    
    # Группы компонентов для классификации
    COMPONENT_GROUPS = {
        'Вяжущие': [
            'Цемент белый ПЦ500',
            'Цемент серый ПЦ500, кг', 
            'Известь гашеная, кг'
        ],
        'Наполнители_песок': [
            'Песок лужский фр.0-0,63мм, кг',
            'Песок кварцевый белый фр.0,2-0,63 мм, кг',
            'Песок кварцевый белый фр.0,4-1,25 мм, кг',
            'Песок  кварцевый белый фр.0,63-1,5 мм, кг',
            'Песок карьерный фр.0,63-2,5 мм, кг',
            'Песок кварцевый белый фр.1,0-3,0 мм, кг',
            'Песок карьерный фр.2,5-5,0 мм, кг'
        ],
        'Наполнители_минеральные': [
            'Доломитовая мука, кг',
            'Микрокальцит МК100 фр.0,1 мм, кг'
        ],
        'Пигменты': [
            'Пигмент светло красный S110, кг',
            'Пигмент красный S130, кг',
            'Пигмент желтый S313, кг',
            'Пигмент оранжевый S960, кг',
            'Пигмент зеленый S5605, кг',
            'Пигмент утрамарин синий, кг',
            'Пигмент светло коричневый S610, кг',
            'Пигмент темно корчневый S686 (S868), кг',
            'Пигмент черный S722 (S723), кг'
        ],
        'Мрамор_белый': [
            'Мрамор белый фр.0,2-0,5 мм, кг',
            'Мрамор белый фр.0,5-1,0 мм, кг',
            'Мрамор белый фр.1,0-1,5 мм, кг',
            'Мрамор белый фр.1,5-2,0 мм, кг',
            'Мрамор белый фр.2,0-3,0 мм, кг',
            'Мрамор белый фр.3,0-5,0 мм, кг',
            'Мрамор белый фр.2,0-7,0 мм, кг'
        ],
        'Мрамор_цветной': [
            'Мрамор черный фр.1,0-3,0мм, кг',
            'Мрамор черный фр.3,0-5,0мм, кг',
            'Мрамор серый фр.1,0-3,0мм, кг',
            'Мрамор серый фр.2,0-3,0мм, кг'
        ],
        'Декоративные_наполнители': [
            'Известняк фр.2.0-3.0мм',
            'Известняк фр.2,0-6,0мм, кг',
            'Купершлак (габродиабаз) фр.0.5-2.5 мм, кг',
            'Мрамор красный кардинал ред фр.1,0-4,0мм, кг',
            'Мрамор красный кардинал ред фр.2,0-3,0 мм, кг',
            'Гранитный отсев фр.2,5-5,0мм, кг',
            'Златолит фр.1,0-3,0 мм, кг',
            'Златолит фр.5,0-10,0 мм, кг',
            'Златолит фр.10,0-20,0 мм, кг',
            'Фельзит фр.4,0-5,0 мм, кг',
            'Мрамор черный (шунгит) фр.5,0-8,0 мм, кг',
            'Мрамор черный (шунгит) фр.5,0-20,0 мм, кг',
            'Змеевик зеленый фр.5,0-10,0 мм, кг',
            'Яшма желтая фр. 2,0-5,0 мм, кг',
            'Фельзит коричневый фр.5,0-10,0мм, кг',
            'Корунд (стекло) фр.1,0-3,0 мм, кг',
            'Слюда фр.2,5-5,0мм, кг'
        ],
        'Добавки': [
            'Пластификатор С-3, (Reamin, РС101, Melflux 5581, Flux3  (терраццо)), кг',
            'Метилцеллюлоза  20000-45000 мПа (HPMC C 712, Walocel MKX20000PP20, Culminal 4053, Wekcelo 75(150), Wekcelo 400 (терраццо)), кг',
            'РПП Полипласт (Dairen 1400, Vinnapas 4023, Vinavil 5603, WWJF - 8020, ОРП 7085, Elotex) кг',
            'Крахмал картофельный (эфир крахмала Casucol, Berolan ST801, Amitrolit 8850), кг',
            'Порообразователь Ufapore, Esapon (любой кроме альфаолефинсульфоната ), кг',
            'Формиат кальция, кг'
        ]
    }
    
    def __init__(self, excel_path: str = None):
        """
        Инициализация анализатора
        
        Args:
            excel_path: Путь к файлу Excel с рецептами
        """
        self.excel_path = excel_path
        self.df = None
        self.analysis_results = {}
        
    def load_excel(self, excel_path: str = None):
        """
        Загрузка данных из Excel файла
        
        Args:
            excel_path: Путь к файлу Excel (если не указан при инициализации)
        """
        if excel_path:
            self.excel_path = excel_path
            
        if not self.excel_path:
            raise ValueError("Не указан путь к файлу Excel")
            
        print(f"Загрузка Excel файла: {self.excel_path}")
        self.df = pd.read_excel(self.excel_path, header=0)
        
        # Удаление строки с итоговой суммой
        self.df = self.df[self.df.iloc[:, 0] != 'Общая сумма компонетов в рецепте, кг']
        
        # Первый столбец - название рецепта
        recipe_names = self.df.iloc[:, 0].astype(str)
        
        # Определяем категорию рецепта
        categories = []
        for name in recipe_names:
            category = 'Неизвестно'
            for cat, prefixes in self.RECIPE_CATEGORIES.items():
                for prefix in prefixes:
                    if name.startswith(prefix):
                        category = cat
                        break
                if category != 'Неизвестно':
                    break
            categories.append(category)
        
        self.df['category'] = categories
        self.df['recipe_name'] = recipe_names
        
        print(f"Загружено {len(self.df)} рецептов")
        print("Категории рецептов:", dict(Counter(categories)))
        
    def analyze_components(self):
        """
        Анализ компонентов по категориям рецептов
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала выполните load_excel()")
        
        results = {
            'category_stats': {},
            'component_frequency': defaultdict(lambda: defaultdict(int)),
            'average_composition': {},
            'unique_components_by_category': {},
            'component_groups_by_category': {}
        }
        
        # Анализ по категориям
        for category in self.RECIPE_CATEGORIES.keys():
            category_df = self.df[self.df['category'] == category]
            
            if len(category_df) == 0:
                continue
                
            # Статистика по категории
            results['category_stats'][category] = {
                'count': len(category_df),
                'recipes': category_df['recipe_name'].tolist()
            }
            
            # Частота использования компонентов
            component_columns = self.df.columns[1:-2]  # Все колонки компонентов
            
            for component in component_columns:
                non_zero = category_df[component].apply(
                    lambda x: x if isinstance(x, (int, float)) and not pd.isna(x) and x != 0 else None
                ).dropna()
                
                if len(non_zero) > 0:
                    results['component_frequency'][category][component] = len(non_zero)
            
            # Средний состав
            avg_composition = {}
            for component in component_columns:
                values = category_df[component].apply(
                    lambda x: x if isinstance(x, (int, float)) and not pd.isna(x) else 0
                )
                if values.sum() > 0:
                    avg_composition[component] = values.mean()
            
            results['average_composition'][category] = avg_composition
            
            # Уникальные компоненты категории
            unique_comps = []
            for component in component_columns:
                if component in results['component_frequency'][category]:
                    # Проверяем, используется ли компонент преимущественно в этой категории
                    other_cat_usage = sum(
                        1 for cat in self.RECIPE_CATEGORIES.keys() 
                        if cat != category and component in results['component_frequency'][cat]
                    )
                    if other_cat_usage == 0:
                        unique_comps.append(component)
            
            results['unique_components_by_category'][category] = unique_comps
            
            # Анализ по группам компонентов
            group_usage = {}
            for group_name, group_components in self.COMPONENT_GROUPS.items():
                group_count = 0
                for comp in group_components:
                    if comp in results['component_frequency'][category]:
                        group_count += results['component_frequency'][category][comp]
                group_usage[group_name] = group_count
            
            results['component_groups_by_category'][category] = group_usage
        
        self.analysis_results = results
        return results
    
    def generate_report(self, output_dir: str = "reports"):
        """
        Генерация отчета по анализу
        
        Args:
            output_dir: Директория для сохранения отчетов
        """
        if not self.analysis_results:
            self.analyze_components()
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Сохранение JSON отчета
        report_path = Path(output_dir) / "component_analysis.json"
        
        # Сериализуемый отчет
        serializable_report = {
            'category_stats': self.analysis_results['category_stats'],
            'component_frequency': {
                cat: dict(comps) 
                for cat, comps in self.analysis_results['component_frequency'].items()
            },
            'unique_components_by_category': self.analysis_results['unique_components_by_category'],
            'component_groups_by_category': self.analysis_results['component_groups_by_category']
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        print(f"Отчет сохранен: {report_path}")
        
        # Генерация текстового отчета
        text_report_path = Path(output_dir) / "component_analysis.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("АНАЛИЗ КОМПОНЕНТОВ ТЕРРАЗИТОВЫХ СОСТАВОВ\n")
            f.write("=" * 80 + "\n\n")
            
            # Статистика по категориям
            f.write("СТАТИСТИКА ПО КАТЕГОРИЯМ:\n")
            f.write("-" * 40 + "\n")
            for category, stats in self.analysis_results['category_stats'].items():
                f.write(f"\n{category}:\n")
                f.write(f"  Количество рецептов: {stats['count']}\n")
            
            # Уникальные компоненты по категориям
            f.write("\n\nУНИКАЛЬНЫЕ КОМПОНЕНТЫ ПО КАТЕГОРИЯМ:\n")
            f.write("-" * 40 + "\n")
            for category, components in self.analysis_results['unique_components_by_category'].items():
                if components:
                    f.write(f"\n{category}:\n")
                    for comp in components[:10]:  # Показываем первые 10
                        f.write(f"  - {comp}\n")
                    if len(components) > 10:
                        f.write(f"  ... и еще {len(components) - 10} компонентов\n")
            
            # Группы компонентов
            f.write("\n\nИСПОЛЬЗОВАНИЕ ГРУПП КОМПОНЕНТОВ:\n")
            f.write("-" * 40 + "\n")
            for category, groups in self.analysis_results['component_groups_by_category'].items():
                f.write(f"\n{category}:\n")
                for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        f.write(f"  {group}: {count} использований\n")
        
        print(f"Текстовый отчет сохранен: {text_report_path}")
        
        return report_path
    
    def visualize_analysis(self, output_dir: str = "reports/visualizations"):
        """
        Визуализация результатов анализа
        
        Args:
            output_dir: Директория для сохранения визуализаций
        """
        if not self.analysis_results:
            self.analyze_components()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Распределение рецептов по категориям
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # График распределения
        categories = list(self.analysis_results['category_stats'].keys())
        counts = [stats['count'] for stats in self.analysis_results['category_stats'].values()]
        
        axes[0, 0].bar(categories, counts, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[0, 0].set_title('Распределение рецептов по категориям')
        axes[0, 0].set_ylabel('Количество рецептов')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Топ-10 компонентов по частоте использования
        all_components = defaultdict(int)
        for category, components in self.analysis_results['component_frequency'].items():
            for comp, freq in components.items():
                all_components[comp] += freq
        
        top_components = sorted(all_components.items(), key=lambda x: x[1], reverse=True)[:10]
        comp_names = [c[0][:30] + '...' if len(c[0]) > 30 else c[0] for c in top_components]
        comp_freqs = [c[1] for c in top_components]
        
        axes[0, 1].barh(range(len(comp_names)), comp_freqs)
        axes[0, 1].set_yticks(range(len(comp_names)))
        axes[0, 1].set_yticklabels(comp_names)
        axes[0, 1].set_title('Топ-10 наиболее частых компонентов')
        axes[0, 1].set_xlabel('Количество использований')
        
        # 3. Использование групп компонентов по категориям
        categories = list(self.analysis_results['component_groups_by_category'].keys())
        group_names = list(self.COMPONENT_GROUPS.keys())
        
        data_matrix = []
        for category in categories:
            row = []
            for group in group_names:
                row.append(self.analysis_results['component_groups_by_category'][category].get(group, 0))
            data_matrix.append(row)
        
        im = axes[1, 0].imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_xticks(range(len(group_names)))
        axes[1, 0].set_xticklabels(group_names, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(len(categories)))
        axes[1, 0].set_yticklabels(categories)
        axes[1, 0].set_title('Использование групп компонентов по категориям')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Круговая диаграмма категорий
        axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', 
                      colors=['blue', 'green', 'orange', 'red', 'purple'])
        axes[1, 1].set_title('Процентное распределение категорий')
        
        plt.tight_layout()
        viz_path = Path(output_dir) / "component_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Визуализации сохранены в: {viz_path}")
        
        return viz_path
    
    def get_category_mapping(self) -> Dict[str, List[str]]:
        """
        Получение маппинга категорий для использования в ML модели
        
        Returns:
            Словарь с маппингом категорий и их признаков
        """
        mapping = {
            'category_labels': list(self.RECIPE_CATEGORIES.keys()),
            'component_groups': self.COMPONENT_GROUPS,
            'category_prefixes': self.RECIPE_CATEGORIES
        }
        
        return mapping
    
    def get_component_features(self) -> Dict:
        """
        Получение признаков компонентов для векторизации
        
        Returns:
            Словарь с признаками компонентов
        """
        if not self.analysis_results:
            self.analyze_components()
        
        # Собираем все уникальные компоненты
        all_components = set()
        for category, components in self.analysis_results['component_frequency'].items():
            all_components.update(components.keys())
        
        # Создаем маппинг компонент -> индекс
        component_to_idx = {comp: idx for idx, comp in enumerate(sorted(all_components))}
        
        # Группируем компоненты
        component_groups = {}
        for comp in all_components:
            for group_name, group_list in self.COMPONENT_GROUPS.items():
                if comp in group_list:
                    component_groups[comp] = group_name
                    break
            else:
                component_groups[comp] = 'other'
        
        features = {
            'component_to_idx': component_to_idx,
            'idx_to_component': {v: k for k, v in component_to_idx.items()},
            'component_groups': component_groups,
            'total_components': len(all_components)
        }
        
        return features


def main():
    """Основная функция для анализа компонентов"""
    analyzer = ComponentAnalyzer()
    
    try:
        # Попытка загрузить реальный Excel файл
        excel_paths = [
            "data/raw/recipes.xlsx",
            "Рецептуры терразит.xlsx",
            "../data/raw/recipes.xlsx"
        ]
        
        excel_loaded = False
        for path in excel_paths:
            if Path(path).exists():
                analyzer.load_excel(path)
                excel_loaded = True
                break
        
        if not excel_loaded:
            print("Предупреждение: Excel файл не найден. Создаю тестовые данные...")
            # Создаем тестовый DataFrame
            test_data = {
                'Наименование рецепта/компоненты в кг на 1000 кг': [
                    'Терразит К62А', 'Шовный МШ1', 'Мастика К1', 
                    'Терраццо Ц1М', 'Ретушь 1'
                ],
                'Цемент белый ПЦ500': [100, 230, 90, 349, 320],
                'Цемент серый ПЦ500, кг': [150, 15, 90, 0, 0],
                'category': ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь'],
                'recipe_name': ['Терразит К62А', 'Шовный МШ1', 'Мастика К1', 
                              'Терраццо Ц1М', 'Ретушь 1']
            }
            analyzer.df = pd.DataFrame(test_data)
        
        # Выполняем анализ
        results = analyzer.analyze_components()
        
        # Генерируем отчет
        report_path = analyzer.generate_report()
        
        # Создаем визуализации
        viz_path = analyzer.visualize_analysis()
        
        # Получаем признаки для ML
        features = analyzer.get_component_features()
        
        print("\n" + "="*80)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
        print("="*80)
        print(f"Обнаружено категорий: {len(results['category_stats'])}")
        print(f"Обнаружено уникальных компонентов: {features['total_components']}")
        print(f"Отчеты сохранены в директории: reports/")
        
        # Сохраняем признаки для использования в ML модели
        features_path = "data/processed/component_features.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        print(f"Признаки компонентов сохранены: {features_path}")
        
    except Exception as e:
        print(f"Ошибка при анализе: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
