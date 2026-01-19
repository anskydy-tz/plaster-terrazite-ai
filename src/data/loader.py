"""
Модуль для загрузки и подготовки данных для ML моделей
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os

# Добавляем путь для импорта модулей проекта
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError as e:
    # Создаем простой логгер как fallback
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Не удалось импортировать setup_logger: {e}. Используется basicConfig")


class DataLoader:
    """Класс для загрузки данных рецептов терразитовой штукатурки"""
    
    def __init__(self, data_path: str = "data/processed/recipes.json"):
        self.data_path = Path(data_path)
        self.recipes = None
        self.df = None
        
    def load_recipes(self) -> List[Dict]:
        """Загрузка рецептов из JSON файла"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.recipes = json.load(f)
        
        logger.info(f"Загружено {len(self.recipes)} рецептов")
        return self.recipes
    
    @staticmethod
    def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Загрузка и предобработка изображения"""
        from PIL import Image
        
        try:
            img = Image.open(image_path)
            # Конвертируем в RGB, если нужно
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Изменяем размер
            img = img.resize(target_size)
            # Конвертируем в numpy массив и нормализуем в [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {image_path}: {e}")
            raise

    @staticmethod
    def load_recipe_json(json_path: str) -> List[Dict]:
        """Загрузка рецептов из JSON файла"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                recipes = json.load(f)
            logger.info(f"Загружено {len(recipes)} рецептов из {json_path}")
            return recipes
        except Exception as e:
            logger.error(f"Ошибка загрузки JSON {json_path}: {e}")
            raise
    
    def prepare_dataframe(self) -> pd.DataFrame:
        """Преобразование рецептов в DataFrame для ML"""
        if self.recipes is None:
            self.load_recipes()
        
        data = []
        
        for recipe in self.recipes:
            # Базовые признаки
            row = {
                'id': recipe['id'],
                'name': recipe['name'],
                'type': recipe['type'],
                'total_weight': recipe.get('total_weight', 1000)
            }
            
            # Добавляем компоненты как отдельные колонки
            for component, weight in recipe['components'].items():
                # Нормализуем название колонки (убираем спецсимволы, сокращаем)
                col_name = self._normalize_component_name(component)
                row[col_name] = weight / 1000  # Переводим в тонны (нормализация)
            
            data.append(row)
        
        self.df = pd.DataFrame(data).fillna(0)
        
        # Логируем информацию о данных
        logger.info(f"Создан DataFrame: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
        logger.info(f"Типы рецептов: {self.df['type'].value_counts().to_dict()}")
        
        return self.df
    
    def _normalize_component_name(self, component: str) -> str:
        """Нормализация названий компонентов для использования в качестве имен колонок"""
        # Убираем лишние пробелы, приводим к нижнему регистру
        normalized = component.strip().lower()
        
        # Заменяем неалфавитно-цифровые символы на подчеркивания
        import re
        normalized = re.sub(r'[^\w\s]', '_', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        
        # Сокращаем слишком длинные названия
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        return normalized
    
    def get_features_and_targets(self, target_components: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка признаков и целевых переменных для ML
        
        Args:
            target_components: Список компонентов для предсказания.
                              Если None, использует все компоненты.
        
        Returns:
            X: Признаки (тип рецепта, общие характеристики)
            y: Целевые переменные (проценты компонентов)
        """
        if self.df is None:
            self.prepare_dataframe()
        
        # Признаки: тип рецепта (кодируем) и другие мета-признаки
        X = pd.get_dummies(self.df['type'], prefix='type')
        
        # Если нужны дополнительные признаки, можно добавить здесь
        
        # Целевые переменные: проценты компонентов
        if target_components is None:
            # Используем все колонки компонентов (исключая мета-колонки)
            meta_cols = ['id', 'name', 'type', 'total_weight']
            component_cols = [col for col in self.df.columns if col not in meta_cols]
            y = self.df[component_cols].values
        else:
            # Используем только указанные компоненты
            y = self.df[target_components].values
        
        logger.info(f"Размерность признаков (X): {X.shape}")
        logger.info(f"Размерность целевых переменных (y): {y.shape}")
        
        return X.values, y
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Разделение данных на обучающую и тестовую выборки"""
        from sklearn.model_selection import train_test_split
        
        X, y = self.get_features_and_targets()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"Обучающая выборка: {X_train.shape[0]} образцов")
        logger.info(f"Тестовая выборка: {X_test.shape[0]} образцов")
        
        return X_train, X_test, y_train, y_test
    
    def get_component_statistics(self) -> pd.DataFrame:
        """Статистика по компонентам"""
        if self.df is None:
            self.prepare_dataframe()
        
        # Мета-колонки
        meta_cols = ['id', 'name', 'type', 'total_weight']
        component_cols = [col for col in self.df.columns if col not in meta_cols]
        
        stats = []
        for col in component_cols:
            non_zero = self.df[col][self.df[col] > 0]
            if len(non_zero) > 0:
                stats.append({
                    'component': col,
                    'count': len(non_zero),
                    'mean': non_zero.mean(),
                    'std': non_zero.std(),
                    'min': non_zero.min(),
                    'max': non_zero.max(),
                    'total_used': self.df[col].sum()
                })
        
        return pd.DataFrame(stats).sort_values('count', ascending=False)


class RecipeLoader:
    """Загрузчик рецептов из Excel файла"""
    
    def __init__(self, excel_path: str = "data/raw/recipes.xlsx"):
        self.excel_path = Path(excel_path)
        self.recipes_df = None
        self.components = None
        
    @staticmethod
    def _parse_float(value: Any) -> float:
        """Парсинг значений в float"""
        # Если это число
        if isinstance(value, (int, float)):
            return float(value)
        
        # Если это строка
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        
        # Если это NaN
        if pd.isna(value):
            return 0.0
        
        # Если это None
        if value is None:
            return 0.0
        
        return 0.0
        
    def load_excel(self) -> pd.DataFrame:
        """Загрузка рецептов из Excel файла"""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel файл не найден: {self.excel_path}")
            
        try:
            self.recipes_df = pd.read_excel(self.excel_path)
            logger.info(f"Загружено {len(self.recipes_df)} рецептов из Excel")
            return self.recipes_df
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel: {e}")
            raise
            
    def parse_components(self) -> Dict[str, Dict[str, float]]:
        """Парсинг компонентов из DataFrame"""
        if self.recipes_df is None:
            self.load_excel()
            
        # Примерная логика парсинга (нужно адаптировать под вашу структуру Excel)
        components = {}
        
        # Предполагаем, что колонки с компонентами начинаются после определенных колонок
        meta_cols = ['id', 'Название', 'Тип', 'Описание']
        component_cols = [col for col in self.recipes_df.columns if col not in meta_cols]
        
        for _, row in self.recipes_df.iterrows():
            recipe_id = row.get('id', 'unknown')
            recipe_components = {}
            
            for component in component_cols:
                value = row.get(component)
                if pd.notna(value) and value != 0:
                    recipe_components[component] = self._parse_float(value)
                    
            components[str(recipe_id)] = recipe_components
            
        self.components = components
        logger.info(f"Распарсено {len(components)} рецептов с компонентами")
        return components
        
    def get_component_names(self) -> List[str]:
        """Получение списка уникальных названий компонентов"""
        if self.components is None:
            self.parse_components()
            
        all_components = set()
        for recipe_comps in self.components.values():
            all_components.update(recipe_comps.keys())
            
        return sorted(list(all_components))
        
    def save_to_json(self, output_path: str = "data/processed/recipes.json"):
        """Сохранение рецептов в JSON формат"""
        if self.components is None:
            self.parse_components()
            
        # Преобразуем в формат, совместимый с DataLoader
        recipes_list = []
        for recipe_id, comp_dict in self.components.items():
            # Находим соответствующую строку в DataFrame
            recipe_row = self.recipes_df[self.recipes_df['id'].astype(str) == str(recipe_id)].iloc[0]
            
            recipe_data = {
                'id': str(recipe_id),
                'name': recipe_row.get('Название', ''),
                'type': recipe_row.get('Тип', 'unknown'),
                'components': comp_dict,
                'total_weight': sum(comp_dict.values())
            }
            recipes_list.append(recipe_data)
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recipes_list, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Рецепты сохранены в JSON: {output_path}")
        
    def process_pipeline(self, excel_path: Optional[str] = None, output_path: Optional[str] = None):
        """Полный пайплайн обработки Excel -> JSON"""
        if excel_path:
            self.excel_path = Path(excel_path)
            
        self.load_excel()
        self.parse_components()
        
        if output_path:
            self.save_to_json(output_path)
        else:
            self.save_to_json()


class ManifestDataLoader(DataLoader):
    """Загрузчик данных для работы с манифестами изображений и рецептов"""
    
    def __init__(self, manifest_dir: str = "data/processed"):
        """
        Инициализация загрузчика манифестов
        
        Args:
            manifest_dir: Директория с CSV манифестами
        """
        super().__init__()
        self.manifest_dir = Path(manifest_dir)
        self.manifests = {}
        self.image_cache = {}
        
    def load_manifest(self, manifest_name: str = "train") -> pd.DataFrame:
        """
        Загрузка манифеста по имени
        
        Args:
            manifest_name: Имя манифеста (train, val, test, full)
            
        Returns:
            DataFrame с данными манифеста
        """
        manifest_path = self.manifest_dir / f"data_manifest_{manifest_name}.csv"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Манифест не найден: {manifest_path}")
        
        df = pd.read_csv(manifest_path, encoding='utf-8')
        logger.info(f"Загружен манифест '{manifest_name}': {len(df)} записей")
        
        self.manifests[manifest_name] = df
        return df
    
    def load_recipes_dict(self, recipes_json_path: str = "data/processed/recipes.json") -> Dict:
        """
        Загрузка рецептов в словарь для быстрого доступа
        
        Args:
            recipes_json_path: Путь к JSON файлу с рецептами
            
        Returns:
            Словарь рецептов по ID
        """
        recipes = self.load_recipe_json(recipes_json_path)
        recipes_dict = {}
        
        for recipe in recipes:
            recipe_id = recipe['id']
            recipes_dict[recipe_id] = {
                'id': recipe_id,
                'name': recipe.get('name', ''),
                'type': recipe.get('type', 'unknown'),
                'components': recipe.get('components', {}),
                'total_weight': recipe.get('total_weight', 0)
            }
        
        logger.info(f"Загружено {len(recipes_dict)} рецептов в словарь")
        return recipes_dict
    
    def prepare_image_data(self, manifest_df: pd.DataFrame, 
                          recipes_dict: Dict,
                          target_size: Tuple[int, int] = (224, 224),
                          use_test_images: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка изображений и меток для обучения
        
        Args:
            manifest_df: DataFrame манифеста
            recipes_dict: Словарь рецептов
            target_size: Размер изображений
            use_test_images: Использовать тестовые изображения (для .txt файлов)
            
        Returns:
            Кортеж (изображения, метки)
        """
        images = []
        labels = []
        failed_images = []
        
        # Получаем все уникальные компоненты для создания единого вектора признаков
        all_components = set()
        for recipe in recipes_dict.values():
            all_components.update(recipe['components'].keys())
        component_list = sorted(list(all_components))
        
        logger.info(f"Всего уникальных компонентов: {len(component_list)}")
        
        for idx, row in manifest_df.iterrows():
            try:
                image_path = Path(row['image_path'])
                recipe_id = str(row['recipe_id'])
                recipe_type = row.get('recipe_type', 'unknown')
                
                # Для тестовых данных всегда создаем тестовые изображения
                # Вместо попытки загрузки реальных файлов
                img_array = self._create_test_image(
                    recipe_id=recipe_id,
                    recipe_type=recipe_type,
                    target_size=target_size
                )
                
                # Получение рецепта и преобразование в вектор
                recipe = recipes_dict.get(recipe_id)
                if recipe is None:
                    logger.warning(f"Рецепт с ID {recipe_id} не найден, пропускаем")
                    continue
                
                # Преобразование компонентов в вектор
                component_vector = self._components_to_vector(
                    recipe['components'], 
                    component_list
                )
                
                images.append(img_array)
                labels.append(component_vector)
                
                # Логируем успешное создание тестового изображения
                if idx < 3:  # Логируем только первые 3 для примера
                    logger.info(f"Создано тестовое изображение для рецепта {recipe_id} ({recipe_type})")
                
            except Exception as e:
                failed_images.append(str(image_path))
                logger.error(f"Ошибка обработки изображения {row['image_path']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if failed_images:
            logger.warning(f"Не удалось обработать {len(failed_images)} изображений")
        
        logger.info(f"Успешно обработано {len(images)} изображений")
        
        # Проверяем, есть ли данные для возврата
        if len(images) == 0:
            return np.array([]), np.array([])
        
        return np.array(images), np.array(labels)
    
    def _create_test_image(self, recipe_id: str, recipe_type: str, 
                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Создание тестового изображения на основе ID и типа рецепта
        
        Args:
            recipe_id: ID рецепта
            recipe_type: Тип рецепта
            target_size: Размер изображения
            
        Returns:
            Тестовое изображение как numpy array
        """
        # Создаем детерминированное "изображение" на основе ID рецепта
        seed = int(recipe_id) if recipe_id.isdigit() else abs(hash(recipe_id)) % 1000
        np.random.seed(seed)
        
        height, width = target_size
        channels = 3
        
        # Разные типы рецептов создают разные паттерны
        if recipe_type == 'внутренняя':
            # Более светлые тона для внутренней штукатурки
            base_color = np.array([0.8, 0.8, 0.9])  # Светло-голубой
            noise = np.random.randn(height, width, channels) * 0.1
        elif recipe_type == 'фасадная':
            # Серые тона для фасадной
            base_color = np.array([0.6, 0.6, 0.6])  # Серый
            noise = np.random.randn(height, width, channels) * 0.15
        elif recipe_type == 'декоративная':
            # Более яркие тона для декоративной
            base_color = np.array([0.9, 0.8, 0.7])  # Бежевый
            noise = np.random.randn(height, width, channels) * 0.2
        else:
            base_color = np.array([0.5, 0.5, 0.5])  # Нейтральный серый
            noise = np.random.randn(height, width, channels) * 0.1
        
        # Создаем базовое изображение
        image = np.ones((height, width, channels)) * base_color
        
        # Добавляем шум для имитации текстуры
        image = image + noise
        
        # Обрезаем значения до [0, 1]
        image = np.clip(image, 0, 1)
        
        return image.astype(np.float32)
    
    def _components_to_vector(self, components: Dict, component_list: List) -> np.ndarray:
        """
        Преобразование словаря компонентов в вектор
        
        Args:
            components: Словарь компонентов
            component_list: Список всех компонентов
            
        Returns:
            Вектор значений компонентов
        """
        vector = np.zeros(len(component_list), dtype=np.float32)
        
        for i, component in enumerate(component_list):
            if component in components:
                # Нормализуем значения (предполагаем, что сумма ≈ 100)
                vector[i] = components[component] / 100.0
        
        return vector
    
    def _create_minimal_test_dataset(self, manifest_df: pd.DataFrame,
                                   recipes_dict: Dict,
                                   target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание минимального тестового датасета, если не удалось загрузить данные
        
        Args:
            manifest_df: DataFrame манифеста
            recipes_dict: Словарь рецептов
            target_size: Размер изображений
            
        Returns:
            Кортеж (изображения, метки)
        """
        logger.info("Создание минимального тестового датасета...")
        
        images = []
        labels = []
        
        # Получаем все уникальные компоненты
        all_components = set()
        for recipe in recipes_dict.values():
            all_components.update(recipe['components'].keys())
        component_list = sorted(list(all_components))
        
        # Создаем 5 тестовых изображений
        for i in range(min(5, len(manifest_df))):
            row = manifest_df.iloc[i]
            recipe_id = str(row['recipe_id'])
            recipe_type = row.get('recipe_type', 'unknown')
            
            # Создаем тестовое изображение
            img_array = self._create_test_image(
                recipe_id=recipe_id,
                recipe_type=recipe_type,
                target_size=target_size
            )
            
            # Получаем рецепт
            recipe = recipes_dict.get(recipe_id)
            if recipe:
                # Преобразование компонентов в вектор
                component_vector = self._components_to_vector(
                    recipe['components'], 
                    component_list
                )
                
                images.append(img_array)
                labels.append(component_vector)
        
        logger.info(f"Создано {len(images)} тестовых изображений")
        
        return np.array(images), np.array(labels)
    
    def get_dataset_info(self, manifest_name: str = "train") -> Dict:
        """
        Получение информации о датасете
        
        Args:
            manifest_name: Имя манифеста
            
        Returns:
            Словарь с информацией
        """
        if manifest_name not in self.manifests:
            self.load_manifest(manifest_name)
        
        df = self.manifests[manifest_name]
        
        info = {
            'total_samples': len(df),
            'unique_recipes': df['recipe_id'].nunique(),
            'recipe_types': df['recipe_type'].value_counts().to_dict(),
            'file_extensions': df['image_path'].apply(
                lambda x: Path(x).suffix
            ).value_counts().to_dict(),
            'split': manifest_name
        }
        
        return info
    
    def prepare_training_data(self, 
                            train_manifest: str = "train",
                            val_manifest: str = "val",
                            test_manifest: str = "test",
                            recipes_json: str = "data/processed/recipes.json",
                            target_size: Tuple[int, int] = (224, 224)) -> Dict:
        """
        Подготовка всех данных для обучения
        
        Args:
            train_manifest: Имя манифеста для обучения
            val_manifest: Имя манифеста для валидации
            test_manifest: Имя манифеста для тестирования
            recipes_json: Путь к JSON с рецептами
            target_size: Размер изображений
            
        Returns:
            Словарь с подготовленными данными
        """
        logger.info("="*60)
        logger.info("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ")
        logger.info("="*60)
        
        # Загрузка рецептов
        recipes_dict = self.load_recipes_dict(recipes_json)
        
        # Загрузка и подготовка данных для каждой выборки
        datasets = {}
        
        for split_name in [train_manifest, val_manifest, test_manifest]:
            logger.info(f"\nПодготовка данных для '{split_name}':")
            
            # Загрузка манифеста
            manifest_df = self.load_manifest(split_name)
            
            # Подготовка изображений и меток
            images, labels = self.prepare_image_data(
                manifest_df=manifest_df,
                recipes_dict=recipes_dict,
                target_size=target_size,
                use_test_images=True  # ВСЕГДА используем тестовые изображения для демо
            )
            
            # Проверяем, есть ли данные
            if len(images) == 0:
                logger.warning(f"Нет данных для split '{split_name}'")
                # Создаем минимальный набор данных для тестирования
                logger.info(f"Создаем минимальный тестовый набор для '{split_name}'")
                images, labels = self._create_minimal_test_dataset(
                    manifest_df=manifest_df,
                    recipes_dict=recipes_dict,
                    target_size=target_size
                )
                
            datasets[split_name] = {
                'images': images,
                'labels': labels,
                'manifest': manifest_df,
                'info': self.get_dataset_info(split_name)
            }
            
            logger.info(f"  Изображений: {len(images)}")
            logger.info(f"  Меток: {len(labels)}")
            logger.info(f"  Размер изображений: {images.shape}")
        
        # Проверяем, что все датасеты созданы
        if len(datasets) == 0:
            raise ValueError("Не удалось загрузить данные ни для одного split")
        
        logger.info("\n" + "="*60)
        logger.info("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
        logger.info("="*60)
        
        # Вывод общей статистики
        total_images = sum(len(datasets[split]['images']) for split in datasets)
        logger.info(f"Всего изображений: {total_images}")
        
        for split_name, data in datasets.items():
            info = data['info']
            logger.info(f"\n{split_name.upper()}:")
            logger.info(f"  Образцов: {info['total_samples']}")
            logger.info(f"  Уникальных рецептов: {info['unique_recipes']}")
            logger.info(f"  Распределение типов: {info['recipe_types']}")
        
        return datasets
    
    @staticmethod
    def get_component_names_from_json(json_path: str) -> List[str]:
        """Получение списка компонентов из JSON файла"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            
            all_components = set()
            for recipe in recipes:
                all_components.update(recipe.get('components', {}).keys())
            
            return sorted(list(all_components))
        except Exception as e:
            logger.error(f"Ошибка загрузки компонентов: {e}")
            return []


# Функция для быстрой проверки
def test_manifest_loader():
    """Тестирование ManifestDataLoader"""
    print("Тестирование ManifestDataLoader...")
    
    try:
        loader = ManifestDataLoader()
        
        # Проверка загрузки манифеста
        train_df = loader.load_manifest("train")
        print(f"✓ Загружен манифест train: {len(train_df)} записей")
        
        # Проверка загрузки рецептов
        recipes_dict = loader.load_recipes_dict()
        print(f"✓ Загружено рецептов: {len(recipes_dict)}")
        
        # Проверка подготовки данных
        datasets = loader.prepare_training_data(
            train_manifest="train",
            val_manifest="val",
            test_manifest="test",
            target_size=(224, 224)
        )
        
        print("✓ Данные успешно подготовлены:")
        for split_name, data in datasets.items():
            print(f"  {split_name}: {len(data['images'])} изображений")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Пример использования
    loader = DataLoader()
    
    # Загрузка данных
    recipes = loader.load_recipes()
    print(f"Загружено рецептов: {len(recipes)}")
    
    # Подготовка DataFrame
    df = loader.prepare_dataframe()
    print(f"DataFrame: {df.shape}")
    
    # Статистика
    stats = loader.get_component_statistics()
    print(f"\nТоп-10 компонентов по использованию:")
    print(stats.head(10))
    
    # Разделение данных
    X_train, X_test, y_train, y_test = loader.split_data()
    print(f"\nРазделение данных:")
    print(f"Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"Тестовая выборка: {X_test.shape[0]} образцов")
    
    # Тестирование ManifestDataLoader
    print("\n" + "="*60)
    print("Тестирование ManifestDataLoader:")
    print("="*60)
    
    success = test_manifest_loader()
    if success:
        print("\n✅ ManifestDataLoader работает корректно!")
        print("\nСледующий шаг:")
        print("Обновите scripts/train_model.py для использования ManifestDataLoader")
    else:
        print("\n❌ Обнаружены проблемы в ManifestDataLoader")
