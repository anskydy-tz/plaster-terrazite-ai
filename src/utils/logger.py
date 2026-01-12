"""
Настройка логирования для проекта
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "terrazite_ai", log_level: str = "INFO"):
    """
    Настройка логгера
    
    Args:
        name: Имя логгера
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Настроенный логгер
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Если уже есть хендлеры, не добавляем новые
    if logger.handlers:
        return logger
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый хендлер
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "terrazite_ai.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
