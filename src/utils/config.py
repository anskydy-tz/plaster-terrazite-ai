"""
Конфигурация проекта Terrazite AI
Включает настройки для работы с категориями компонентов
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import yaml

from .logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DataConfig:
    """Конфигурация данных"""
    # Директории
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    images_dir: str = "data/processed/images"
    
    # Файлы
    excel_file: str = "recipes.xlsx"
    processed_json: str = "recipes_processed.json"
    ml_data_file: str = "ml_ready_data.json"
    
    # Категории рецептов (на основе анализа Excel)
    recipe_categories: List[str] = field(default_factory=lambda: [
        'Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь'
    ])
    
    # Группы компонентов (на основе анализа Excel)
    component_groups: Dict[str, List[str]] = field(default_factory=lambda: {
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
    })
    
    # Настройки загрузки
    image_size: tuple = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Настройки аугментации
    augmentation_enabled: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    # Основные параметры
    model_name: str = "TerraziteResNet50"
    input_size: tuple = (224, 224, 3)
    num_categories: int = 5  # Терразит, Шовный, Мастика, Терраццо, Ретушь
    num_components: int = 100  # Будет обновлено после анализа данных
    
    # Архитектура
    backbone: str = "resnet50"
    use_pretrained: bool = True
    hidden_size: int = 512
    dropout_rate: float = 0.3
    
    # Обучение
    batch_size: int = 32  # ДОБАВЛЕНО: размер батча
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Loss weights
    category_weight: float = 1.0
    component_weight: float = 0.5
    regression_weight: float = 0.3


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    # Пути
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    tensorboard_dir: str = "runs"
    
    # Сохранение
    save_frequency: int = 5  # Сохранять каждые N эпох
    best_model_metric: str = "val_loss"
    
    # Мониторинг
    monitor_metrics: List[str] = field(default_factory=lambda: [
        'loss', 'val_loss', 'accuracy', 'val_accuracy'
    ])


@dataclass
class APIConfig:
    """Конфигурация API"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    workers: int = 4
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*"
    ])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds


@dataclass
class StreamlitConfig:
    """Конфигурация Streamlit интерфейса"""
    title: str = "Terrazite AI - Подбор рецепта терразитовой штукатурки"
    page_icon: str = "🏗️"
    layout: str = "wide"
    
    # Вкладки
    tabs: List[str] = field(default_factory=lambda: [
        "🔍 Анализ изображения",
        "📊 База рецептов",
        "⚙️ Настройки модели"
    ])


@dataclass 
class ProjectConfig:
    """Основная конфигурация проекта"""
    # Версия
    version: str = "1.1.0"
    project_name: str = "Terrazite AI"
    
    # Режимы
    mode: str = "development"  # development, production, testing
    debug: bool = True
    
    # Пути
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    config_file: str = "config.yaml"
    
    # Подконфигурации
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    
    def __post_init__(self):
        """Пост-инициализация: создание директорий"""
        self._create_directories()
    
    def _create_directories(self):
        """Создание необходимых директорий"""
        dirs_to_create = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.images_dir,
            self.training.checkpoint_dir,
            self.training.logs_dir,
            self.training.tensorboard_dir,
            "reports",
            "reports/visualizations",
            "exports"
        ]
        
        for dir_path in dirs_to_create:
            full_path = Path(self.project_root) / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория: {full_path}")
    
    def update_from_excel(self, excel_path: str = None):
        """
        Обновление конфигурации на основе анализа Excel файла
        
        Args:
            excel_path: Путь к Excel файлу (если None, используется из конфигурации)
        """
        try:
            from src.data.component_analyzer import ComponentAnalyzer
            
            excel_path = excel_path or str(Path(self.project_root) / self.data.excel_file)
            
            if not Path(excel_path).exists():
                logger.warning(f"Excel файл не найден для обновления конфигурации: {excel_path}")
                return
            
            # Анализ Excel файла
            analyzer = ComponentAnalyzer(excel_path)
            analyzer.load_excel()
            features = analyzer.get_component_features()
            
            # Обновляем количество компонентов
            self.model.num_components = features.get('total_components', 100)
            
            # Обновляем группы компонентов (если отличаются)
            current_groups_set = set(str(g) for g in self.data.component_groups.keys())
            analyzer_groups_set = set(str(g) for g in analyzer.COMPONENT_GROUPS.keys())
            
            if current_groups_set != analyzer_groups_set:
                logger.info(f"Обновление групп компонентов: обнаружено {len(analyzer_groups_set)} групп")
                self.data.component_groups = analyzer.COMPONENT_GROUPS
            
            # Сохраняем обновленную конфигурацию
            self.save()
            
            logger.info(f"Конфигурация обновлена на основе анализа Excel файла")
            logger.info(f"Количество компонентов: {self.model.num_components}")
            logger.info(f"Группы компонентов: {len(self.data.component_groups)}")
            
        except Exception as e:
            logger.error(f"Ошибка при обновлении конфигурации из Excel: {e}")
    
    def save(self, config_path: str = None):
        """
        Сохранение конфигурации в файл
        
        Args:
            config_path: Путь для сохранения (если None, используется project_root/config.yaml)
        """
        if config_path is None:
            config_path = Path(self.project_root) / self.config_file
        
        # Конвертируем dataclass в словарь
        config_dict = self._to_dict()
        
        # Сохраняем как YAML
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Конфигурация сохранена: {config_path}")
    
    def load(self, config_path: str = None):
        """
        Загрузка конфигурации из файла
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        if config_path is None:
            config_path = Path(self.project_root) / self.config_file
        
        if not Path(config_path).exists():
            logger.warning(f"Файл конфигурации не найден: {config_path}")
            return self
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # Обновляем объект конфигурации
            self._update_from_dict(config_dict)
            logger.info(f"Конфигурация загружена: {config_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
        
        return self
    
    def _to_dict(self) -> Dict[str, Any]:
        """Конвертация конфигурации в словарь"""
        config_dict = {}
        
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            
            if hasattr(field_value, '_to_dict'):
                config_dict[field_name] = field_value._to_dict()
            elif hasattr(field_value, '__dataclass_fields__'):
                # Рекурсивно конвертируем dataclass
                config_dict[field_name] = {
                    f.name: getattr(field_value, f.name)
                    for f in field_value.__dataclass_fields__.values()
                }
            else:
                config_dict[field_name] = field_value
        
        return config_dict
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Обновление конфигурации из словаря"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                
                if hasattr(current_value, '_update_from_dict') and isinstance(value, dict):
                    current_value._update_from_dict(value)
                elif hasattr(current_value, '__dataclass_fields__') and isinstance(value, dict):
                    # Обновляем dataclass
                    for sub_key, sub_value in value.items():
                        if hasattr(current_value, sub_key):
                            setattr(current_value, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def get_component_group(self, component_name: str) -> Optional[str]:
        """
        Получение группы для компонента
        
        Args:
            component_name: Название компонента
            
        Returns:
            Название группы или None
        """
        for group_name, components in self.data.component_groups.items():
            if component_name in components:
                return group_name
        return None
    
    def get_category_info(self, category_name: str) -> Dict[str, Any]:
        """
        Получение информации о категории рецептов
        
        Args:
            category_name: Название категории
            
        Returns:
            Словарь с информацией о категории
        """
        if category_name not in self.data.recipe_categories:
            return {"error": f"Категория {category_name} не найдена"}
        
        # Получаем компоненты, характерные для этой категории
        typical_components = []
        
        # Здесь можно добавить логику для определения характерных компонентов
        # на основе статистики из обработанных данных
        
        return {
            "name": category_name,
            "typical_components": typical_components,
            "description": self._get_category_description(category_name)
        }
    
    def _get_category_description(self, category_name: str) -> str:
        """Получение описания категории"""
        descriptions = {
            'Терразит': 'Основные строительные смеси для отделки фасадов и интерьеров',
            'Шовный': 'Затирочные составы для заполнения швов и трещин',
            'Мастика': 'Клеевые и герметизирующие составы',
            'Терраццо': 'Декоративные покрытия с мраморной крошкой',
            'Ретушь': 'Ремонтные составы для восстановления поверхностей'
        }
        
        return descriptions.get(category_name, "Описание не доступно")


# Глобальный экземпляр конфигурации
config = ProjectConfig()


def setup_config(config_path: str = None) -> ProjectConfig:
    """
    Настройка конфигурации проекта
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Объект конфигурации
    """
    global config
    
    if config_path:
        config = config.load(config_path)
    elif Path(config.config_file).exists():
        config = config.load()
    
    # Обновляем из Excel, если есть файл
    excel_path = Path(config.project_root) / config.data.excel_file
    if excel_path.exists():
        try:
            config.update_from_excel(str(excel_path))
        except Exception as e:
            logger.warning(f"Не удалось обновить конфигурацию из Excel: {e}")
    
    logger.info(f"Конфигурация загружена: {config.project_name} v{config.version}")
    logger.info(f"Режим: {config.mode}, Отладка: {config.debug}")
    logger.info(f"Категории рецептов: {len(config.data.recipe_categories)}")
    logger.info(f"Группы компонентов: {len(config.data.component_groups)}")
    
    return config


def save_current_config(config_path: str = None):
    """Сохранение текущей конфигурации"""
    config.save(config_path)


def create_default_config(config_path: str = "config.yaml"):
    """Создание конфигурации по умолчанию"""
    default_config = ProjectConfig()
    default_config.save(config_path)
    logger.info(f"Конфигурация по умолчанию создана: {config_path}")
    return default_config


def get_component_mapping() -> Dict[str, int]:
    """
    Получение маппинга компонентов для ML модели
    
    Returns:
        Словарь компонент -> индекс
    """
    try:
        # Пытаемся загрузить из обработанных данных
        ml_data_path = Path(config.project_root) / config.data.processed_data_dir / config.data.ml_data_file
        
        if ml_data_path.exists():
            with open(ml_data_path, 'r', encoding='utf-8') as f:
                ml_data = json.load(f)
            
            component_mapping = ml_data.get('component_mapping', {}).get('component_to_idx', {})
            
            if component_mapping:
                logger.info(f"Загружен маппинг для {len(component_mapping)} компонентов")
                return component_mapping
    except Exception as e:
        logger.warning(f"Не удалось загрузить маппинг компонентов: {e}")
    
    # Возвращаем пустой маппинг, если не удалось загрузить
    logger.warning("Используется пустой маппинг компонентов")
    return {}


if __name__ == "__main__":
    # Тестирование конфигурации
    cfg = setup_config()
    
    print("\n" + "="*80)
    print("ТЕСТ КОНФИГУРАЦИИ ПРОЕКТА")
    print("="*80)
    print(f"Проект: {cfg.project_name} v{cfg.version}")
    print(f"Режим: {cfg.mode}")
    print(f"Корневая директория: {cfg.project_root}")
    
    print(f"\nДАННЫЕ:")
    print(f"  Категории рецептов: {cfg.data.recipe_categories}")
    print(f"  Группы компонентов: {list(cfg.data.component_groups.keys())}")
    
    print(f"\nМОДЕЛЬ:")
    print(f"  Название: {cfg.model.model_name}")
    print(f"  Категорий: {cfg.model.num_categories}")
    print(f"  Компонентов: {cfg.model.num_components}")
    
    print(f"\nAPI:")
    print(f"  Хост: {cfg.api.host}")
    print(f"  Порт: {cfg.api.port}")
    
    # Пример получения группы для компонента
    test_component = "Цемент белый ПЦ500"
    group = cfg.get_component_group(test_component)
    print(f"\nПример: компонент '{test_component}' относится к группе: {group}")
    
    # Пример получения информации о категории
    test_category = "Терразит"
    category_info = cfg.get_category_info(test_category)
    print(f"Категория '{test_category}': {category_info.get('description', 'Нет описания')}")
    
    print("\nКонфигурация успешно загружена!")
