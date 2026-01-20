#!/usr/bin/env python3
"""
Единый скрипт для запуска всего пайплайна Terrazite AI.
Запускает все этапы: подготовка данных → создание манифеста → обучение → оценка.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TerrazitePipeline:
    """Класс для запуска полного пайплайна Terrazite AI"""
    
    def __init__(self, config_path: str = None):
        """
        Инициализация пайплайна.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.steps_completed = []
        self.errors = []
        self.start_time = datetime.now()
        
        logger.info(f"🚀 Инициализация пайплайна Terrazite AI")
        logger.info(f"Время начала: {self.start_time}")
    
    def run_step(self, step_name: str, command: list, check_output: bool = True):
        """
        Запуск шага пайплайна.
        
        Args:
            step_name: Название шага
            command: Команда для выполнения
            check_output: Проверять ли вывод на ошибки
            
        Returns:
            True если успешно, False если ошибка
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ШАГ: {step_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Запускаем команду
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Логируем вывод
            if result.stdout:
                logger.info(f"Вывод {step_name}:\n{result.stdout}")
            
            if result.stderr:
                logger.warning(f"Ошибки {step_name}:\n{result.stderr}")
            
            # Проверяем код возврата
            if result.returncode != 0:
                error_msg = f"Ошибка выполнения {step_name}: код {result.returncode}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                return False
            
            # Проверяем вывод на наличие ошибок
            if check_output and "error" in result.stdout.lower():
                error_msg = f"Ошибка в выводе {step_name}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                return False
            
            # Шаг выполнен успешно
            self.steps_completed.append(step_name)
            logger.info(f"✅ {step_name} выполнен успешно")
            return True
            
        except Exception as e:
            error_msg = f"Исключение при выполнении {step_name}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False
    
    def check_prerequisites(self):
        """Проверка необходимых условий"""
        logger.info("\n🔍 Проверка необходимых условий...")
        
        requirements = [
            ("data/raw/", "Директория с сырыми данными"),
            ("src/", "Исходный код проекта"),
            ("requirements.txt", "Зависимости проекта"),
            ("create_test_excel.py", "Скрипт создания тестовых данных")
        ]
        
        all_ok = True
        for path, description in requirements:
            if Path(path).exists():
                logger.info(f"✅ {description}: {path}")
            else:
                logger.warning(f"⚠️  {description}: {path} не найден")
                all_ok = False
        
        return all_ok
    
    def create_test_data(self):
        """Шаг 1: Создание тестовых данных"""
        return self.run_step(
            "Создание тестовых данных",
            ["python", "create_test_excel.py"]
        )
    
    def process_excel(self):
        """Шаг 2: Обработка Excel файла"""
        return self.run_step(
            "Обработка Excel файла",
            ["python", "scripts/process_excel.py", "--no-analyze"]
        )
    
    def create_manifest(self):
        """Шаг 3: Создание манифеста данных"""
        return self.run_step(
            "Создание манифеста данных",
            ["python", "create_data_manifest.py"]
        )
    
    def prepare_dataset(self):
        """Шаг 4: Подготовка датасета"""
        return self.run_step(
            "Подготовка датасета",
            ["python", "scripts/prepare_image_dataset.py", "--create-mapping", "--no-augmentation"]
        )
    
    def train_model(self, epochs: int = 10, batch_size: int = 4):
        """Шаг 5: Обучение модели"""
        return self.run_step(
            "Обучение модели",
            [
                "python", "scripts/train_model.py",
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--plot"
            ]
        )
    
    def run_tests(self):
        """Шаг 6: Запуск тестов"""
        return self.run_step(
            "Запуск тестов",
            ["python", "run_tests.py"]
        )
    
    def generate_report(self):
        """Генерация отчета о выполнении пайплайна"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "pipeline": "Terrazite AI",
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "steps_completed": self.steps_completed,
            "errors": self.errors,
            "success": len(self.errors) == 0,
            "summary": {
                "total_steps": len(self.steps_completed),
                "successful_steps": len(self.steps_completed) - len(self.errors),
                "failed_steps": len(self.errors)
            }
        }
        
        # Сохраняем отчет
        report_path = Path("reports/pipeline_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Вывод отчета
        print("\n" + "="*80)
        print("ОТЧЕТ О ВЫПОЛНЕНИИ ПАЙПЛАЙНА")
        print("="*80)
        
        print(f"\n📊 Статистика:")
        print(f"  Начало: {report['start_time']}")
        print(f"  Конец: {report['end_time']}")
        print(f"  Длительность: {duration}")
        print(f"  Шагов выполнено: {report['summary']['total_steps']}")
        print(f"  Успешных шагов: {report['summary']['successful_steps']}")
        print(f"  Неудачных шагов: {report['summary']['failed_steps']}")
        
        print(f"\n✅ Выполненные шаги:")
        for step in report['steps_completed']:
            print(f"  • {step}")
        
        if report['errors']:
            print(f"\n❌ Ошибки:")
            for error in report['errors']:
                print(f"  • {error}")
        else:
            print(f"\n🎉 Все шаги выполнены успешно!")
        
        print(f"\n📄 Полный отчет сохранен: {report_path}")
        
        return report
    
    def run_full_pipeline(self, quick_mode: bool = False):
        """
        Запуск полного пайплайна.
        
        Args:
            quick_mode: Быстрый режим (меньше эпох, без аугментации)
        """
        logger.info("\n" + "="*80)
        logger.info("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА TERRAZITE AI")
        logger.info("="*80)
        
        # Проверка условий
        if not self.check_prerequisites():
            logger.warning("⚠️  Некоторые условия не выполнены, но продолжаем...")
        
        # Определяем параметры
        epochs = 5 if quick_mode else 50
        batch_size = 2 if quick_mode else 32
        
        # Шаги пайплайна
        steps = [
            ("Создание тестовых данных", self.create_test_data),
            ("Обработка Excel файла", self.process_excel),
            ("Создание манифеста данных", self.create_manifest),
            ("Подготовка датасета", self.prepare_dataset),
            ("Обучение модели", lambda: self.train_model(epochs, batch_size)),
        ]
        
        # Запускаем шаги
        for step_name, step_func in steps:
            if not step_func():
                logger.error(f"❌ Пайплайн остановлен на шаге: {step_name}")
                break
        
        # Генерация отчета
        report = self.generate_report()
        
        return report


def main():
    """Основная функция скрипта"""
    parser = argparse.ArgumentParser(description='Запуск полного пайплайна Terrazite AI')
    parser.add_argument('--quick', action='store_true',
                       help='Быстрый режим (тестовый прогон)')
    parser.add_argument('--steps', type=str, default='all',
                       help='Шаги для выполнения (all, data, train, test)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Размер батча')
    parser.add_argument('--no-report', action='store_true',
                       help='Не генерировать отчет')
    
    args = parser.parse_args()
    
    # Создаем и запускаем пайплайн
    pipeline = TerrazitePipeline()
    
    if args.steps == 'all':
        report = pipeline.run_full_pipeline(quick_mode=args.quick)
    else:
        # Выполняем только указанные шаги
        if 'data' in args.steps:
            pipeline.create_test_data()
            pipeline.process_excel()
            pipeline.create_manifest()
            pipeline.prepare_dataset()
        
        if 'train' in args.steps:
            epochs = args.epochs or (5 if args.quick else 50)
            batch_size = args.batch_size or (2 if args.quick else 32)
            pipeline.train_model(epochs, batch_size)
        
        if 'test' in args.steps:
            pipeline.run_tests()
        
        report = pipeline.generate_report()
    
    # Возвращаем код завершения
    if report.get('success'):
        print("\n🎉 ПАЙПЛАЙН ВЫПОЛНЕН УСПЕШНО!")
        return 0
    else:
        print("\n⚠️  ПАЙПЛАЙН ВЫПОЛНЕН С ОШИБКАМИ")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
