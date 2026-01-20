"""
Упрощенный классификатор для быстрого прототипирования и baseline моделей.
Интегрирован с проектом Terrazite AI для работы с категориями рецептов и компонентами.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleAggregateClassifier:
    """
    Простой классификатор на основе случайного леса для определения
    категории терразитового состава по изображению и компонентам.
    Может использоваться для быстрого прототипирования и baseline оценки.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 random_state: int = 42,
                 typical_components: Optional[Dict[str, List[str]]] = None):
        """
        Инициализация классификатора
        
        Args:
            n_estimators: Количество деревьев в RandomForest
            random_state: Семя для воспроизводимости
            typical_components: Словарь типичных компонентов по категориям
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Модель
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Вспомогательные объекты
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Информация о проекте
        self.feature_names = []
        self.classes_ = []
        self.category_info = {}
        
        # Типичные компоненты по категориям
        self.typical_components = typical_components or self._load_default_components()
        
        logger.info(f"Инициализирован SimpleAggregateClassifier с {n_estimators} деревьями")
    
    def _load_default_components(self) -> Dict[str, List[str]]:
        """
        Загрузка типичных компонентов из конфигурации проекта
        
        Returns:
            Словарь категория -> список типичных компонентов
        """
        try:
            # Пробуем загрузить из конфигурации
            components_by_category = {}
            
            for category in config.data.recipe_categories:
                # Получаем информацию о категории
                category_info = config.get_category_info(category)
                
                # Берем типичные компоненты или используем первые из групп
                typical_comps = []
                
                # Для каждой категории выбираем характерные компоненты
                if category == 'Терразит':
                    typical_comps = ['Цемент белый ПЦ500', 'Песок лужский фр.0-0,63мм, кг']
                elif category == 'Шовный':
                    typical_comps = ['Цемент серый ПЦ500, кг', 'Микрокальцит МК100 фр.0,1 мм, кг']
                elif category == 'Мастика':
                    typical_comps = ['Цемент белый ПЦ500', 'Доломитовая мука, кг']
                elif category == 'Терраццо':
                    typical_comps = ['Мрамор белый фр.0,5-1,0 мм, кг', 'Пигмент желтый S313, кг']
                elif category == 'Ретушь':
                    typical_comps = ['Цемент белый ПЦ500', 'РПП Полипласт (Dairen 1400, Vinnapas 4023, Vinavil 5603, WWJF - 8020, ОРП 7085, Elotex) кг']
                
                components_by_category[category] = typical_comps
            
            logger.info("Типичные компоненты загружены из конфигурации")
            return components_by_category
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить компоненты из конфигурации: {e}")
            
            # Заглушка по умолчанию
            return {
                'Терразит': ['Цемент белый ПЦ500', 'Песок лужский'],
                'Шовный': ['Цемент серый ПЦ500, кг', 'Микрокальцит'],
                'Мастика': ['Цемент белый ПЦ500', 'Доломитовая мука, кг'],
                'Терраццо': ['Мрамор белый', 'Пигменты'],
                'Ретушь': ['Цемент белый ПЦ500', 'РПП Полипласт']
            }
    
    def extract_color_histogram(self, 
                               image: np.ndarray, 
                               bins: int = 32) -> np.ndarray:
        """
        Извлечение гистограммы цветов из изображения
        
        Args:
            image: Изображение как numpy array [H, W, 3]
            bins: Количество бинов в гистограмме
            
        Returns:
            Вектор признаков гистограммы
        """
        try:
            # Проверяем размерность изображения
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.warning(f"Неправильная размерность изображения: {image.shape}")
                # Пытаемся исправить
                if len(image.shape) == 3 and image.shape[2] > 3:
                    image = image[:, :, :3]
                elif len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
            
            # Разделяем на каналы RGB
            channels = []
            for i in range(3):
                channel = image[:, :, i]
                hist, _ = np.histogram(channel, bins=bins, range=(0, 256))
                channels.append(hist)
            
            # Объединяем и нормализуем
            features = np.concatenate(channels)
            features = features / (image.shape[0] * image.shape[1])  # Нормализация по пикселям
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка извлечения гистограммы: {e}")
            # Возвращаем нулевой вектор соответствующей размерности
            return np.zeros(3 * bins)
    
    def extract_component_features(self, 
                                 components_dict: Dict[str, float]) -> np.ndarray:
        """
        Извлечение признаков из словаря компонентов
        
        Args:
            components_dict: Словарь компонент -> значение в кг
            
        Returns:
            Вектор признаков компонентов
        """
        try:
            # Используем все компоненты из конфигурации
            all_components = []
            for group_components in config.data.component_groups.values():
                all_components.extend(group_components)
            
            # Создаем вектор фиксированной длины
            features = np.zeros(len(all_components))
            
            for i, component in enumerate(all_components):
                if component in components_dict:
                    features[i] = components_dict[component]
            
            # Нормализация
            total = features.sum()
            if total > 0:
                features = features / total * 1000  # Нормализуем и масштабируем
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков компонентов: {e}")
            return np.zeros(100)  # Вектор по умолчанию
    
    def combine_features(self, 
                        image_features: np.ndarray,
                        component_features: np.ndarray) -> np.ndarray:
        """
        Комбинирование признаков изображения и компонентов
        
        Args:
            image_features: Признаки изображения
            component_features: Признаки компонентов
            
        Returns:
            Объединенный вектор признаков
        """
        # Взвешенное объединение
        image_weight = 0.6  # Больший вес для изображения
        component_weight = 0.4
        
        # Нормализация
        if image_features.max() > 0:
            image_features = image_features / image_features.max()
        
        if component_features.max() > 0:
            component_features = component_features / component_features.max()
        
        # Объединение
        combined = np.concatenate([
            image_features * image_weight,
            component_features * component_weight
        ])
        
        return combined
    
    def fit(self, 
           X_images: List[np.ndarray], 
           X_components: List[Dict[str, float]],
           y: List[str]) -> 'SimpleAggregateClassifier':
        """
        Обучение модели на изображениях и компонентах
        
        Args:
            X_images: Список изображений
            X_components: Список словарей компонентов
            y: Список меток категорий
            
        Returns:
            self
        """
        logger.info("Извлечение и комбинирование признаков...")
        
        # Извлечение признаков
        X_features = []
        for img, comp in zip(X_images, X_components):
            img_features = self.extract_color_histogram(img)
            comp_features = self.extract_component_features(comp)
            combined = self.combine_features(img_features, comp_features)
            X_features.append(combined)
        
        X_features = np.array(X_features)
        self.feature_names = [f'feature_{i}' for i in range(X_features.shape[1])]
        
        # Кодирование меток
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Нормализация признаков
        X_features = self.scaler.fit_transform(X_features)
        
        # Обучение модели
        logger.info(f"Обучение RandomForest на {len(X_features)} образцах...")
        self.model.fit(X_features, y_encoded)
        
        # Кросс-валидация для оценки
        cv_scores = cross_val_score(self.model, X_features, y_encoded, cv=5)
        logger.info(f"Кросс-валидация (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Информация о классах
        self.category_info = {}
        for i, class_name in enumerate(self.classes_):
            class_indices = np.where(y_encoded == i)[0]
            self.category_info[class_name] = {
                'count': len(class_indices),
                'typical_components': self.typical_components.get(class_name, [])
            }
        
        logger.info(f"Обучено. Классы: {list(self.classes_)}")
        
        return self
    
    def fit_from_dataframe(self, 
                          df: pd.DataFrame,
                          image_column: str = 'image',
                          component_columns: List[str] = None) -> 'SimpleAggregateClassifier':
        """
        Обучение модели из DataFrame
        
        Args:
            df: DataFrame с данными
            image_column: Колонка с изображениями
            component_columns: Колонки с компонентами
            
        Returns:
            self
        """
        # Подготовка данных
        X_images = []
        X_components = []
        y = []
        
        for _, row in df.iterrows():
            # Изображение
            if image_column in row:
                X_images.append(row[image_column])
            
            # Компоненты
            components_dict = {}
            if component_columns:
                for col in component_columns:
                    if col in row and pd.notna(row[col]):
                        components_dict[col] = float(row[col])
            
            X_components.append(components_dict)
            
            # Метка (предполагаем, что есть колонка 'category')
            if 'category' in row:
                y.append(str(row['category']))
        
        return self.fit(X_images, X_components, y)
    
    def predict(self, 
               X_images: Union[np.ndarray, List[np.ndarray]],
               X_components: Optional[List[Dict[str, float]]] = None) -> np.ndarray:
        """
        Предсказание категорий для изображений
        
        Args:
            X_images: Одно изображение или список изображений
            X_components: Список словарей компонентов (опционально)
            
        Returns:
            Предсказанные категории
        """
        # Подготовка входных данных
        if isinstance(X_images, np.ndarray) and len(X_images.shape) == 3:
            X_images = [X_images]
        
        if X_components is None:
            X_components = [{} for _ in range(len(X_images))]
        elif isinstance(X_components, dict):
            X_components = [X_components]
        
        # Извлечение признаков
        X_features = []
        for img, comp in zip(X_images, X_components):
            img_features = self.extract_color_histogram(img)
            comp_features = self.extract_component_features(comp)
            combined = self.combine_features(img_features, comp_features)
            X_features.append(combined)
        
        X_features = np.array(X_features)
        X_features = self.scaler.transform(X_features)
        
        # Предсказание
        y_encoded = self.model.predict(X_features)
        y = self.label_encoder.inverse_transform(y_encoded)
        
        return y
    
    def predict_with_components(self, 
                               image: np.ndarray,
                               components: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Предсказание с дополнительной информацией о компонентах
        
        Args:
            image: Входное изображение
            components: Словарь компонентов (опционально)
            
        Returns:
            Словарь с предсказанием и компонентами
        """
        prediction = self.predict([image], [components] if components else None)[0]
        
        # Получаем вероятности
        if hasattr(self.model, 'predict_proba'):
            # Подготовка признаков
            img_features = self.extract_color_histogram(image)
            comp_features = self.extract_component_features(components or {})
            combined = self.combine_features(img_features, comp_features)
            combined = self.scaler.transform([combined])
            
            probs = self.model.predict_proba(combined)[0]
            confidence = max(probs)
            probabilities = {
                cls: float(prob) 
                for cls, prob in zip(self.classes_, probs)
            }
        else:
            confidence = 1.0
            probabilities = {}
        
        # Типичные компоненты для категории
        typical_components = self._get_typical_components(prediction)
        
        # Информация о категории
        category_info = self.category_info.get(prediction, {})
        
        return {
            'category': prediction,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'typical_components': typical_components,
            'category_info': category_info,
            'model_type': 'RandomForest',
            'components_used': bool(components)
        }
    
    def predict_proba(self, 
                     X_images: Union[np.ndarray, List[np.ndarray]],
                     X_components: Optional[List[Dict[str, float]]] = None) -> np.ndarray:
        """
        Вероятности для каждого класса
        
        Args:
            X_images: Изображения
            X_components: Компоненты
            
        Returns:
            Матрица вероятностей [n_samples, n_classes]
        """
        if isinstance(X_images, np.ndarray) and len(X_images.shape) == 3:
            X_images = [X_images]
        
        if X_components is None:
            X_components = [{} for _ in range(len(X_images))]
        elif isinstance(X_components, dict):
            X_components = [X_components]
        
        # Извлечение признаков
        X_features = []
        for img, comp in zip(X_images, X_components):
            img_features = self.extract_color_histogram(img)
            comp_features = self.extract_component_features(comp)
            combined = self.combine_features(img_features, comp_features)
            X_features.append(combined)
        
        X_features = np.array(X_features)
        X_features = self.scaler.transform(X_features)
        
        return self.model.predict_proba(X_features)
    
    def evaluate(self, 
                X_images: List[np.ndarray],
                X_components: List[Dict[str, float]],
                y: List[str],
                plot_results: bool = True) -> Dict[str, Any]:
        """
        Оценка точности модели
        
        Args:
            X_images: Тестовые изображения
            X_components: Тестовые компоненты
            y: Истинные метки
            plot_results: Визуализировать результаты
            
        Returns:
            Словарь с метриками
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        y_pred = self.predict(X_images, X_components)
        accuracy = accuracy_score(y, y_pred)
        
        # Подробный отчет
        report = classification_report(y, y_pred, output_dict=True)
        
        logger.info(f"Точность: {accuracy:.2%}")
        logger.info("\nОтчет по классификации:")
        for cls, metrics in report.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                logger.info(f"  {cls}: precision={metrics['precision']:.3f}, "
                          f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y, y_pred, labels=self.classes_)
        
        # Визуализация
        if plot_results:
            self.plot_confusion_matrix(cm, self.classes_)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y
        }
    
    def plot_confusion_matrix(self, 
                             cm: np.ndarray,
                             class_names: List[str],
                             save_path: Optional[str] = None):
        """
        Визуализация матрицы ошибок
        
        Args:
            cm: Матрица ошибок
            class_names: Названия классов
            save_path: Путь для сохранения
        """
        plt.figure(figsize=(10, 8))
        
        # Тепловая карта
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Матрица ошибок SimpleAggregateClassifier')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Матрица ошибок сохранена: {save_path}")
        
        plt.show()
        plt.close()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Получение важности признаков
        
        Returns:
            DataFrame с важностью признаков
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Модель не поддерживает важность признаков")
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        # Создаем DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Сортируем по убыванию важности
        df = df.sort_values('importance', ascending=False)
        
        # Группируем по типу признаков (изображение/компоненты)
        df['feature_type'] = df['feature'].apply(
            lambda x: 'image' if x.startswith('feature_') and int(x.split('_')[1]) < 96 
            else 'component'
        )
        
        logger.info("Важность признаков:")
        logger.info(f"  Всего признаков: {len(df)}")
        logger.info(f"  Изображение: {df[df['feature_type'] == 'image']['importance'].sum():.3f}")
        logger.info(f"  Компоненты: {df[df['feature_type'] == 'component']['importance'].sum():.3f}")
        
        return df
    
    def _get_typical_components(self, category: str) -> List[str]:
        """
        Получение типичных компонентов для категории
        
        Args:
            category: Категория рецепта
            
        Returns:
            Список типичных компонентов
        """
        # Сначала пробуем получить из конфигурации проекта
        try:
            if hasattr(self, 'typical_components') and category in self.typical_components:
                return self.typical_components[category]
        except:
            pass
        
        # Заглушка по умолчанию
        components_by_category = {
            'Терразит': ['Цемент белый ПЦ500', 'Песок лужский фр.0-0,63мм, кг'],
            'Шовный': ['Цемент серый ПЦ500, кг', 'Микрокальцит МК100 фр.0,1 мм, кг'],
            'Мастика': ['Цемент белый ПЦ500', 'Доломитовая мука, кг'],
            'Терраццо': ['Мрамор белый фр.0,5-1,0 мм, кг', 'Пигмент желтый S313, кг'],
            'Ретушь': ['Цемент белый ПЦ500', 'РПП Полипласт (Dairen 1400, Vinnapas 4023, Vinavil 5603, WWJF - 8020, ОРП 7085, Elotex) кг']
        }
        
        return components_by_category.get(category, [])
    
    def save(self, path: str):
        """
        Сохранение модели и связанных объектов
        
        Args:
            path: Путь для сохранения
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.classes_,
            'category_info': self.category_info,
            'typical_components': self.typical_components,
            'config': {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Модель сохранена в {path}")
    
    def load(self, path: str) -> 'SimpleAggregateClassifier':
        """
        Загрузка модели и связанных объектов
        
        Args:
            path: Путь к файлу модели
            
        Returns:
            self
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        
        data = joblib.load(path)
        
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.classes_ = data['classes']
        self.category_info = data.get('category_info', {})
        self.typical_components = data.get('typical_components', {})
        
        if 'config' in data:
            self.n_estimators = data['config'].get('n_estimators', 100)
            self.random_state = data['config'].get('random_state', 42)
        
        logger.info(f"Модель загружена из {path}")
        logger.info(f"Классы: {list(self.classes_)}")
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Словарь с информацией о модели
        """
        info = {
            'model_type': 'SimpleAggregateClassifier (RandomForest)',
            'n_estimators': self.n_estimators,
            'classes': list(self.classes_),
            'num_classes': len(self.classes_),
            'num_features': len(self.feature_names),
            'category_info': self.category_info,
            'typical_components': self.typical_components,
            'random_state': self.random_state
        }
        
        return info


def create_simple_classifier(n_estimators: int = 100, 
                           random_state: int = 42) -> SimpleAggregateClassifier:
    """
    Фабричная функция для создания простого классификатора
    
    Args:
        n_estimators: Количество деревьев
        random_state: Семя для воспроизводимости
        
    Returns:
        SimpleAggregateClassifier
    """
    return SimpleAggregateClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )


def test_classifier():
    """Тестирование классификатора"""
    logger.info("🧪 Тестирование SimpleAggregateClassifier")
    logger.info("=" * 50)
    
    # Создаем тестовые данные
    num_samples = 100
    
    # Изображения
    images = [
        np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) 
        for _ in range(num_samples)
    ]
    
    # Компоненты
    components = []
    for _ in range(num_samples):
        comp_dict = {
            'Цемент белый ПЦ500': np.random.uniform(100, 300),
            'Песок лужский фр.0-0,63мм, кг': np.random.uniform(200, 500)
        }
        components.append(comp_dict)
    
    # Метки (5 категорий)
    categories = ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь']
    labels = np.random.choice(categories, size=num_samples)
    
    # Создаем и обучаем модель
    clf = SimpleAggregateClassifier(n_estimators=50, random_state=42)
    
    try:
        clf.fit(images, components, labels)
        
        # Тестирование на подмножестве
        test_images = images[:10]
        test_components = components[:10]
        test_labels = labels[:10]
        
        # Предсказание
        predictions = clf.predict(test_images, test_components)
        logger.info(f"Предсказания: {predictions}")
        
        # Оценка
        metrics = clf.evaluate(test_images, test_components, test_labels, plot_results=False)
        logger.info(f"Точность: {metrics['accuracy']:.2%}")
        
        # Предсказание с компонентами
        single_prediction = clf.predict_with_components(test_images[0], test_components[0])
        logger.info(f"Предсказание с компонентами: {single_prediction}")
        
        # Важность признаков
        feature_importance = clf.get_feature_importance()
        if not feature_importance.empty:
            logger.info(f"Топ-5 важных признаков:")
            for _, row in feature_importance.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Информация о модели
        model_info = clf.get_model_info()
        logger.info(f"Информация о модели:")
        logger.info(f"  Тип: {model_info['model_type']}")
        logger.info(f"  Классы: {model_info['num_classes']}")
        logger.info(f"  Признаков: {model_info['num_features']}")
        
        logger.info("\n✅ SimpleAggregateClassifier готов к работе!")
        
        return clf
        
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Запуск теста
    clf = test_classifier()
    
    if clf:
        print("\nПример использования:")
        print("1. clf = SimpleAggregateClassifier(n_estimators=100)")
        print("2. clf.fit(images, components, labels)")
        print("3. prediction = clf.predict(test_images, test_components)")
        print("4. detailed = clf.predict_with_components(image, components)")
        print("5. clf.save('models/simple_classifier.joblib')")
