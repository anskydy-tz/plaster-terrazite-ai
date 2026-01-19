"""
Модуль для настройки логирования в проекте Terrazite AI
"""
import logging
import sys
from pathlib import Path
import colorlog


def setup_logger(name: str = "terrazite_ai", level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера с цветным выводом
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        
    Returns:
        Настроенный логгер
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Проверяем, есть ли уже обработчики у логгера
    if logger.handlers:
        return logger
    
    # Форматтер с цветами для консоли
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Форматтер для файла (без цветов)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)
    
    # Обработчик для файла
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(
        log_dir / "terrazite_ai.log",
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_format)
    
    # Добавляем обработчики к логгеру
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def setup_file_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера для записи в файл
    
    Args:
        name: Имя логгера
        log_file: Имя файла для логирования
        level: Уровень логирования
        
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Проверяем, есть ли уже обработчики у логгера
    if logger.handlers:
        return logger
    
    # Форматтер для файла
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для файла
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(
        log_dir / log_file,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "terrazite_ai") -> logging.Logger:
    """
    Получение существующего логгера или создание нового
    
    Args:
        name: Имя логгера
        
    Returns:
        Логгер
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger


# Пример использования
if __name__ == "__main__":
    logger = setup_logger("test_logger")
    logger.info("Тест логгера: INFO сообщение")
    logger.warning("Тест логгера: WARNING сообщение")
    logger.error("Тест логгера: ERROR сообщение")
    logger.debug("Тест логгера: DEBUG сообщение")
