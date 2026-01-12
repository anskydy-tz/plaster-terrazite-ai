#!/usr/bin/env python
"""
Скрипт для запуска всех тестов проекта
"""
import sys
import pytest

if __name__ == "__main__":
    # Параметры для pytest
    args = [
        "tests/",
        "-v",  # Подробный вывод
        "--tb=short",  # Короткие tracebacks
        "--cov=src",  # Включить покрытие кода для src/
        "--cov-report=term-missing",  # Показать непокрытые строки
        "--cov-report=html:coverage_html",  # HTML отчет
        "-W", "ignore::DeprecationWarning"  # Игнорировать предупреждения
    ]
    
    # Добавляем аргументы командной строки
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Запускаем тесты
    exit_code = pytest.main(args)
    
    sys.exit(exit_code)
