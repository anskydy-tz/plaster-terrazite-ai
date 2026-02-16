# run_tests.py (обновленная версия)
#! /usr/bin/env python3
"""
Централизованный запуск всех тестов проекта Terrazite AI.
"""
import subprocess
import sys
from pathlib import Path

# Добавляем путь к проекту для импорта конфига, если нужно
sys.path.append(str(Path(__file__).parent))

TESTS = [
    ("Базовая проверка модели", "python test_model_basic.py"),
    ("Проверка полного пайплайна", "python test_full_pipeline.py"),
    # Сюда же можно добавить запуск существующих тестов из tests/
    # ("Модульные тесты", "pytest tests/"),
]

def run_tests():
    print("="*60)
    print("ЗАПУСК ВСЕХ ТЕСТОВ TERRAZITE AI")
    print("="*60)
    all_passed = True
    for test_name, command in TESTS:
        print(f"\n▶️  Запуск: {test_name}")
        print("-"*40)
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"❌ Тест '{test_name}' провален.")
            all_passed = False
        else:
            print(f"✅ Тест '{test_name}' пройден.")

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ. Проверьте вывод выше.")
    print("="*60)
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
