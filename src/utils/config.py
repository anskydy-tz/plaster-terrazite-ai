"""
Конфигурационный модуль проекта Terrazite AI
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json
import logging

logger = logging.getLogger(__name__)


class Config:
    """Класс для управления конфигурацией проекта"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent.parent
        
        # Пути по умолчанию
        self.paths = {
            'data_raw': self.base_dir / 'data' / 'raw',
            'data_processed': self.base_dir / 'data' / 'processed',
            'models': self.base_dir / 'models',
            'logs': self.base_dir / 'logs',
            'reports': self.base_dir / 'reports',
            'checkpoints': self.base_dir / 'checkpoints'
        }
        
        # Параметры данных
        self.data = {
            'image_size': (224, 224),
            'batch_size': 32,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_seed': 42,
            'num_workers': 4,
            'prefetch_factor': 2
        }
        
        # Параметры модели
        self.model = {
            'num_components': 5,
            'backbone': 'efficientnet-b0',
            'dropout_rate': 0.3,
            'hidden_size': 512,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        }
        
        # Параметры обучения
        self.training = {
            'epochs': 100,
            'patience': 10,
            'regression_weight': 1.0,
            'classification_weight': 0.5,
            'early_stopping': True,
            'save_best_only': True,
            'monitor': 'val_loss',
            'monitor_mode': 'min'
        }
        
        # Параметры аугментации
        self.augmentation = {
            'enabled': True,
            'flip': True,
            'rotate': True,
            'color': True,
            'noise': True,
            'dropout': True,
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'brightness_range': [0.8, 1.2],
            'horizontal_flip': True
        }
        
        # API параметры
        self.api = {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': True,
            'reload': True,
            'workers': 1
        }
        
        # Streamlit параметры
        self.streamlit = {
            'port': 8501,
            'theme': 'light',
            'browser_gather_usage_stats': False
        }
        
        # Загружаем кастомную конфигурацию если указана
        if config_path:
            self.load_config(config_path)
        
        # Создаем директории
        self.create_directories()
        
        logger.info(f"Конфигурация загружена. Базовая директория: {self.base_dir}")
    
    def load_config(self, config_path: str):
        """Загрузка конфигурации из файла"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Конфигурационный файл не найден: {config_path}")
            return
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.error(f"Неподдерживаемый формат конфигурации: {config_path.suffix}")
                return
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации {config_path}: {e}")
            return
        
        # Рекурсивное обновление конфигурации
        self._update_dict(self.paths, config_data.get('paths', {}))
        self._update_dict(self.data, config_data.get('data', {}))
        self._update_dict(self.model, config_data.get('model', {}))
        self._update_dict(self.training, config_data.get('training', {}))
        self._update_dict(self.augmentation, config_data.get('augmentation', {}))
        self._update_dict(self.api, config_data.get('api', {}))
        self._update_dict(self.streamlit, config_data.get('streamlit', {}))
        
        logger.info(f"Конфигурация загружена из {config_path}")
    
    def _update_dict(self, target: Dict, source: Dict):
        """Рекурсивное обновление словаря"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def create_directories(self):
        """Создание необходимых директорий"""
        for key, path in self.paths.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория {key}: {path}")
    
    def save(self, output_path: str):
        """Сохранение конфигурации в файл"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'paths': {k: str(v) for k, v in self.paths.items()},
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'augmentation': self.augmentation,
            'api': self.api,
            'streamlit': self.streamlit
        }
        
        if output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Неподдерживаемый формат: {output_path.suffix}")
        
        logger.info(f"Конфигурация сохранена в {output_path}")
    
    def get_path(self, key: str) -> Path:
        """Получение пути по ключу"""
        if key in self.paths:
            return self.paths[key]
        raise KeyError(f"Ключ пути не найден: {key}. Доступные ключи: {list(self.paths.keys())}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Получение значения конфигурации"""
        if hasattr(self, section):
            section_dict = getattr(self, section)
            if key in section_dict:
                return section_dict[key]
        return default
    
    def update(self, section: str, updates: Dict[str, Any]):
        """Обновление секции конфигурации"""
        if hasattr(self, section):
            section_dict = getattr(self, section)
            section_dict.update(updates)
            logger.info(f"Обновлена секция {section}: {list(updates.keys())}")
        else:
            logger.warning(f"Секция {section} не найдена в конфигурации")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        return {
            'paths': {k: str(v) for k, v in self.paths.items()},
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'augmentation': self.augmentation,
            'api': self.api,
            'streamlit': self.streamlit
        }
    
    def __repr__(self) -> str:
        return f"Config(base_dir={self.base_dir})"


# Создаем глобальный экземпляр конфигурации
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Получение глобального экземпляра конфигурации (синглтон)
    
    Args:
        config_path: Путь к конфигурационному файлу
    
    Returns:
        Экземпляр Config
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


if __name__ == "__main__":
    # Тестирование конфигурации
    config = Config()
    print("✅ Конфигурация создана успешно")
    print(f"📁 Базовая директория: {config.base_dir}")
    print(f"🖼️  Размер изображений: {config.data['image_size']}")
    print(f"📊 Параметры обучения: {config.training['epochs']} эпох")
    print(f"🔧 Пути: {config.paths}")
