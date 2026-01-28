"""
Модуль для загрузки и обработки данных
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

# Исправляем импорт с абсолютного пути
from src.utils.logger import setup_logger
from src.data.component_analyzer import ComponentAnalyzer  # Импортируем анализатор

logger = setup_logger(__name__)


@dataclass
class RecipeData:
    """Контейнер для данных рецепта"""
    name: str
    category: str
    components: Dict[str, float]
    image_paths: List[str] = field(default_factory=list)
    features: np.ndarray = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLoader:
    """Основной загрузчик данных"""
    
    @staticmethod
    def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Загрузка и предобработка изображения"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)
            return np.array(image)
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {image_path}: {e}")
            return np.zeros((*target_size, 3), dtype=np.uint8)
    
    @staticmethod
    def load_recipe_json(json_path: str) -> Dict[str, Any]:
        """Загрузка JSON файла с рецептами"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки JSON {json_path}: {e}")
            return {}


class RecipeLoader:
    """Загрузчик рецептов из Excel с поддержкой анализа компонентов"""
    
    def __init__(self, excel_path: str = None, component_analyzer: ComponentAnalyzer = None):
        """
        Инициализация загрузчика рецептов
        
        Args:
            excel_path: Путь к Excel файлу с рецептами
            component_analyzer: Анализатор компонентов (если None, создается новый)
        """
        self.excel_path = excel_path
        self.df = None
        self.recipes = []
        self.categories = []
        self.component_features = None
        
        # Инициализация анализатора компонентов
        if component_analyzer is None:
            self.analyzer = ComponentAnalyzer(excel_path)
        else:
            self.analyzer = component_analyzer
        
        # Загружаем компоненты и их группы, исключая воду
        self._load_component_config()
        
    def _load_component_config(self):
        """Загрузка конфигурации компонентов из анализатора с фильтрацией воды"""
        try:
            # Получаем маппинг категорий
            category_mapping = self.analyzer.get_category_mapping()
            self.categories = category_mapping['category_labels']
            self.component_groups = category_mapping['component_groups']
            
            # Получаем признаки компонентов для ML и фильтруем воду
            raw_features = self.analyzer.get_component_features()
            
            # Фильтруем компоненты с водой
            component_to_idx = {}
            idx_to_component = {}
            
            for component, idx in raw_features.get('component_to_idx', {}).items():
                if 'вода' not in component.lower():
                    # Перенумеровываем индексы
                    new_idx = len(component_to_idx)
                    component_to_idx[component] = new_idx
                    idx_to_component[new_idx] = component
                else:
                    logger.debug(f"Исключен компонент с водой: {component}")
            
            # Обновляем component_groups, удаляя компоненты с водой
            filtered_groups = {}
            for group_name, components in raw_features.get('component_groups', {}).items():
                filtered_components = [c for c in components if 'вода' not in c.lower()]
                if filtered_components:
                    filtered_groups[group_name] = filtered_components
            
            self.component_features = {
                'component_to_idx': component_to_idx,
                'idx_to_component': idx_to_component,
                'component_groups': filtered_groups,
                'total_components': len(component_to_idx)
            }
            
            logger.info(f"Загружено категорий: {len(self.categories)}")
            logger.info(f"Загружено групп компонентов: {len(self.component_features['component_groups'])}")
            logger.info(f"Общее количество уникальных компонентов (без воды): {self.component_features['total_components']}")
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфигурацию компонентов: {e}")
            # Значения по умолчанию
            self.categories = ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь']
            self.component_groups = {}
            self.component_features = {
                'component_to_idx': {},
                'idx_to_component': {},
                'component_groups': {},
                'total_components': 0
            }
    
    def _parse_float(self, value: Any) -> float:
        """Парсинг числовых значений"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                # Убираем пробелы и заменяем запятые на точки
                cleaned = value.replace(',', '.').strip()
                return float(cleaned) if cleaned else 0.0
            except:
                return 0.0
        else:
            return 0.0
    
    def load_excel(self, excel_path: str = None) -> pd.DataFrame:
        """
        Загрузка данных из Excel файла с анализом категорий
        
        Args:
            excel_path: Путь к Excel файлу
            
        Returns:
            DataFrame с данными
        """
        if excel_path:
            self.excel_path = excel_path
        
        if not self.excel_path:
            raise ValueError("Не указан путь к Excel файлу")
        
        logger.info(f"Загрузка Excel файла: {self.excel_path}")
        
        try:
            # Загрузка Excel
            self.df = pd.read_excel(self.excel_path, header=0)
            
            # Удаление строки с итоговой суммой
            self.df = self.df[self.df.iloc[:, 0] != 'Общая сумма компонетов в рецепте, кг']
            
            # Первый столбец - название рецепта
            recipe_names = self.df.iloc[:, 0].astype(str)
            
            # Определяем категорию рецепта на основе анализатора
            categories = []
            for name in recipe_names:
                category = 'Неизвестно'
                for cat, prefixes in self.analyzer.RECIPE_CATEGORIES.items():
                    for prefix in prefixes:
                        if name.startswith(prefix):
                            category = cat
                            break
                    if category != 'Неизвестно':
                        break
                categories.append(category)
            
            self.df['category'] = categories
            self.df['recipe_name'] = recipe_names
            
            logger.info(f"Загружено рецептов: {len(self.df)}")
            logger.info(f"Распределение по категориям: {dict(pd.Series(categories).value_counts())}")
            
            # Анализ компонентов
            self.analyzer.df = self.df
            self.analyzer.analyze_components()
            
            # Обновляем признаки компонентов после анализа (исключая воду)
            self._load_component_config()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel: {e}")
            raise
    
    def parse_components(self, row: pd.Series) -> Dict[str, float]:
        """
        Парсинг компонентов из строки DataFrame
        
        Args:
            row: Строка DataFrame
            
        Returns:
            Словарь компонентов
        """
        components = {}
        
        # Пропускаем первые 2 колонки (название и категория) и последнюю (итог)
        component_columns = self.df.columns[1:-1] if 'category' in self.df.columns else self.df.columns[1:]
        
        for col in component_columns:
            if col not in ['recipe_name', 'category']:
                value = self._parse_float(row[col])
                if value > 0:
                    components[col] = value
        
        return components
    
    def vectorize_components(self, components: Dict[str, float]) -> np.ndarray:
        """
        Векторизация компонентов для ML модели (исключая воду)
        
        Args:
            components: Словарь компонентов
            
        Returns:
            Вектор признаков без водных компонентов
        """
        if not self.component_features or 'component_to_idx' not in self.component_features:
            raise ValueError("Признаки компонентов не загружены. Сначала выполните load_excel()")
        
        vector_size = self.component_features['total_components']
        vector = np.zeros(vector_size, dtype=np.float32)
        
        component_to_idx = self.component_features['component_to_idx']
        
        for component, value in components.items():
            # Пропускаем компоненты с водой
            if 'вода' in component.lower():
                logger.debug(f"Пропущен компонент с водой при векторизации: {component}")
                continue
                
            if component in component_to_idx:
                idx = component_to_idx[component]
                vector[idx] = value / 1000.0  # Нормализация к тоннам
            else:
                logger.warning(f"Компонент не найден в маппинге: {component}")
        
        return vector
    
    def get_recipe_by_category(self, category: str) -> List[RecipeData]:
        """
        Получение рецептов по категории
        
        Args:
            category: Категория рецепта
            
        Returns:
            Список объектов RecipeData
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала выполните load_excel()")
        
        category_df = self.df[self.df['category'] == category]
        recipes = []
        
        for _, row in category_df.iterrows():
            recipe = RecipeData(
                name=row['recipe_name'],
                category=category,
                components=self.parse_components(row),
                metadata={'excel_row': _}
            )
            recipes.append(recipe)
        
        return recipes
    
    def get_all_recipes(self, include_components: bool = True) -> List[RecipeData]:
        """
        Получение всех рецептов
        
        Args:
            include_components: Включать ли компоненты в данные
            
        Returns:
            Список объектов RecipeData
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала выполните load_excel()")
        
        recipes = []
        
        for _, row in self.df.iterrows():
            recipe = RecipeData(
                name=row['recipe_name'],
                category=row['category'],
                components=self.parse_components(row) if include_components else {},
                metadata={'excel_row': _}
            )
            recipes.append(recipe)
        
        self.recipes = recipes
        return recipes
    
    def find_similar_recipes(self, target_components: Dict[str, float], 
                            top_k: int = 5) -> List[Tuple[RecipeData, float]]:
        """
        Поиск похожих рецептов по компонентам
        
        Args:
            target_components: Целевые компоненты
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список кортежей (рецепт, оценка сходства)
        """
        if not self.recipes:
            self.get_all_recipes()
        
        # Векторизация целевых компонентов (исключая воду)
        target_vector = self.vectorize_components(target_components)
        
        similarities = []
        
        for recipe in self.recipes:
            # Векторизация компонентов рецепта (исключая воду)
            recipe_vector = self.vectorize_components(recipe.components)
            
            # Косинусное сходство
            dot_product = np.dot(target_vector, recipe_vector)
            norm_target = np.linalg.norm(target_vector)
            norm_recipe = np.linalg.norm(recipe_vector)
            
            if norm_target > 0 and norm_recipe > 0:
                similarity = dot_product / (norm_target * norm_recipe)
            else:
                similarity = 0.0
            
            similarities.append((recipe, similarity))
        
        # Сортировка по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по компонентам
        
        Returns:
            Словарь со статистикой
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала выполните load_excel()")
        
        stats = {
            'total_recipes': len(self.df),
            'categories': {},
            'component_frequency': {},
            'average_composition_by_category': {}
        }
        
        # Статистика по категориям
        for category in self.categories:
            category_df = self.df[self.df['category'] == category]
            stats['categories'][category] = len(category_df)
        
        # Частота использования компонентов
        component_columns = [col for col in self.df.columns 
                           if col not in ['recipe_name', 'category'] 
                           and not str(col).startswith('Unnamed')]
        
        for component in component_columns:
            if component in self.df.columns:
                non_zero = self.df[component].apply(
                    lambda x: 1 if self._parse_float(x) > 0 else 0
                ).sum()
                stats['component_frequency'][component] = non_zero
        
        # Средний состав по категориям
        for category in self.categories:
            category_df = self.df[self.df['category'] == category]
            avg_composition = {}
            
            for component in component_columns:
                if component in category_df.columns:
                    avg_value = category_df[component].apply(self._parse_float).mean()
                    if avg_value > 0:
                        avg_composition[component] = avg_value
            
            stats['average_composition_by_category'][category] = avg_composition
        
        return stats
    
    def save_to_json(self, output_path: str = "data/processed/recipes_with_categories.json"):
        """
        Сохранение рецептов с категориями в JSON
        
        Args:
            output_path: Путь для сохранения
        """
        if not self.recipes:
            self.get_all_recipes()
        
        output = {
            'metadata': {
                'total_recipes': len(self.recipes),
                'categories': self.categories,
                'component_groups': self.component_groups
            },
            'recipes': []
        }
        
        for recipe in self.recipes:
            recipe_data = {
                'name': recipe.name,
                'category': recipe.category,
                'components': recipe.components,
                'vectorized': self.vectorize_components(recipe.components).tolist(),
                'metadata': recipe.metadata
            }
            output['recipes'].append(recipe_data)
        
        # Создание директории, если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Рецепты сохранены в: {output_path}")
        
        # Также сохраняем анализ компонентов
        analysis_path = Path(output_path).parent / "component_analysis.json"
        self.analyzer.generate_report(str(analysis_path.parent))
        
        return output_path
    
    def process_pipeline(self, excel_path: str = None, 
                        output_json: str = "data/processed/recipes_with_categories.json") -> Dict[str, Any]:
        """
        Полный пайплайн обработки данных
        
        Args:
            excel_path: Путь к Excel файлу
            output_json: Путь для сохранения JSON
            
        Returns:
            Словарь с результатами обработки
        """
        # Загрузка Excel
        self.load_excel(excel_path)
        
        # Получение всех рецептов
        recipes = self.get_all_recipes()
        
        # Сохранение в JSON
        json_path = self.save_to_json(output_json)
        
        # Получение статистики
        stats = self.get_component_statistics()
        
        # Генерация отчета
        report_path = self.analyzer.generate_report("reports")
        
        result = {
            'excel_path': self.excel_path,
            'json_path': json_path,
            'report_path': report_path,
            'total_recipes': len(recipes),
            'categories': stats['categories'],
            'component_features': self.component_features
        }
        
        logger.info(f"Пайплайн обработки завершен. Обработано рецептов: {len(recipes)}")
        
        return result


class TerraziteDataset(Dataset):
    """Датасет для терразитовых составов с поддержкой категорий и фиксированными векторами компонентов (без воды)"""
    
    def __init__(self, recipes_data: List[RecipeData], 
                 image_dir: str = None,
                 transform=None,
                 include_components: bool = True,
                 component_mapping: Optional[Dict] = None):
        """
        Инициализация датасета
        
        Args:
            recipes_data: Список объектов RecipeData
            image_dir: Директория с изображениями
            transform: Трансформации для изображений
            include_components: Включать ли компоненты в данные
            component_mapping: Маппинг компонентов для фиксированных векторов
        """
        self.recipes = recipes_data
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform
        self.include_components = include_components
        self.component_mapping = component_mapping or {}
        
        # Маппинг категорий в индексы
        self.categories = sorted(set([r.category for r in recipes_data]))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        # Загрузка изображений
        self.images = []
        self._load_images()
        
        # Предварительная векторизация компонентов (если есть маппинг)
        if self.include_components and self.component_mapping:
            self.component_vectors = self._precompute_component_vectors()
        else:
            self.component_vectors = None
    
    def _load_images(self):
        """Загрузка путей к изображениям"""
        if not self.image_dir:
            return
        
        for recipe in self.recipes:
            recipe_name_clean = recipe.name.replace('/', '_').replace('\\', '_')
            image_patterns = [
                self.image_dir / f"{recipe_name_clean}*.jpg",
                self.image_dir / f"{recipe_name_clean}*.png",
                self.image_dir / f"{recipe.name}*.jpg",
                self.image_dir / f"{recipe.name}*.png",
            ]
            
            image_paths = []
            for pattern in image_patterns:
                image_paths.extend(list(self.image_dir.glob(pattern.name)))
            
            if image_paths:
                recipe.image_paths = [str(p) for p in image_paths[:3]]  # Берем до 3 изображений
                self.images.extend(recipe.image_paths)
            else:
                logger.warning(f"Изображения для рецепта {recipe.name} не найдены")
    
    def _precompute_component_vectors(self) -> List[torch.Tensor]:
        """Предварительное вычисление векторов компонентов фиксированной длины (без воды)"""
        vectors = []
        
        if 'component_to_idx' not in self.component_mapping:
            logger.warning("В component_mapping отсутствует 'component_to_idx'")
            return vectors
        
        component_to_idx = self.component_mapping['component_to_idx']
        num_components = len(component_to_idx)
        
        for recipe in self.recipes:
            vector = torch.zeros(num_components, dtype=torch.float32)
            
            for component, value in recipe.components.items():
                # Пропускаем компоненты с водой
                if 'вода' in component.lower():
                    continue
                    
                if component in component_to_idx:
                    idx = component_to_idx[component]
                    vector[idx] = value / 1000.0  # Нормализация к тоннам
            
            vectors.append(vector)
        
        logger.info(f"Предварительно вычислено {len(vectors)} векторов компонентов длиной {num_components} (без воды)")
        return vectors
    
    def _get_component_vector(self, recipe: RecipeData, recipe_idx: int) -> torch.Tensor:
        """Получение вектора компонентов фиксированной длины (без воды)"""
        if self.component_vectors is not None and recipe_idx < len(self.component_vectors):
            return self.component_vectors[recipe_idx]
        
        # Резервный вариант: создаем вектор нулевой длины
        if self.include_components:
            logger.warning(f"Вектор компонентов не предвычислен для рецепта {recipe_idx}. Возвращаю нулевой вектор.")
        
        return torch.zeros(1, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Количество элементов в датасета"""
        return len(self.recipes) * max(1, len(self.images) // max(len(self.recipes), 1))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Получение элемента по индексу"""
        recipe_idx = idx % len(self.recipes)
        recipe = self.recipes[recipe_idx]
        
        # Загрузка изображения
        image = None
        if recipe.image_paths:
            img_idx = (idx // len(self.recipes)) % len(recipe.image_paths)
            image_path = recipe.image_paths[img_idx]
            image = DataLoader.load_image(image_path)
            
            if self.transform:
                image = self.transform(image)
        
        # Вектор компонентов фиксированной длины (без воды)
        components = self._get_component_vector(recipe, recipe_idx)
        
        # Категория
        category_idx = self.category_to_idx.get(recipe.category, 0)
        
        item = {
            'name': recipe.name,
            'category': torch.tensor(category_idx, dtype=torch.long),
            'category_name': recipe.category,
            'image': torch.FloatTensor(image).permute(2, 0, 1) if image is not None else torch.zeros((3, 224, 224)),
            'components': components,
            'components_dict': recipe.components
        }
        
        return item
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Получение распределения по категориям"""
        distribution = {}
        for recipe in self.recipes:
            distribution[recipe.category] = distribution.get(recipe.category, 0) + 1
        return distribution


def main():
    """Основная функция для тестирования загрузчика"""
    loader = RecipeLoader()
    
    try:
        # Попытка загрузить Excel файл
        excel_path = "data/raw/recipes.xlsx"
        if not Path(excel_path).exists():
            excel_path = "Рецептуры терразит.xlsx"
        
        if Path(excel_path).exists():
            logger.info(f"Найден Excel файл: {excel_path}")
            result = loader.process_pipeline(excel_path)
            
            print("\n" + "="*80)
            print("РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
            print("="*80)
            print(f"Обработано рецептов: {result['total_recipes']}")
            print(f"Категории: {result['categories']}")
            print(f"JSON сохранен: {result['json_path']}")
            print(f"Отчеты сохранены: {result['report_path']}")
            
            # Пример поиска похожих рецептов (без воды)
            test_components = {
                'Цемент белый ПЦ500': 100,
                'Цемент серый ПЦ500, кг': 150,
                'Песок лужский фр.0-0,63мм, кг': 342
            }
            
            similar = loader.find_similar_recipes(test_components, top_k=3)
            print("\nПОХОЖИЕ РЕЦЕПТЫ:")
            for recipe, similarity in similar:
                print(f"  - {recipe.name} (сходство: {similarity:.3f})")
            
        else:
            logger.warning("Excel файл не найден. Запустите create_test_excel.py для создания тестовых данных.")
            
    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
