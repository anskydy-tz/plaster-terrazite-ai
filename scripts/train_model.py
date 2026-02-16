#!/usr/bin/env python3
"""
Скрипт для обучения модели Terrazite AI.
Использует стандартные манифесты из data/processed/ и конфигурацию проекта.
"""
import sys
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

import torch
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional

from src.models.trainer import ModelTrainer
from src.utils.config import config, setup_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def validate_data_manifests() -> bool:
    """
    Проверка наличия необходимых манифестов данных.
    
    Returns:
        True если все манифесты существуют
    """
    required_files = [
        "data/processed/data_manifest_train.csv",
        "data/processed/data_manifest_val.csv", 
        "data/processed/data_manifest_test.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Манифест не найден: {file_path}")
            all_exist = False
    
    if not all_exist:
        logger.info("\n💡 Сначала запустите подготовку данных:")
        logger.info("   python scripts/create_data_manifest.py")
        logger.info("   python scripts/prepare_image_dataset.py")
    
    return all_exist


def train_model(args: argparse.Namespace) -> tuple:
    """
    Основная функция обучения модели.
    
    Args:
        args: Аргументы командной строки
        
    Returns:
        Кортеж (trainer, history, metrics)
    """
    logger.info("=" * 60)
    logger.info("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ TERRAZITE AI")
    logger.info("=" * 60)
    
    # Настройка конфигурации
    if args.config:
        setup_config(args.config)
        logger.info(f"Конфигурация загружена: {args.config}")
    
    # Приоритет: аргументы > конфигурация > значения по умолчанию
    trainer_config = {
        'batch_size': args.batch_size or config.model.batch_size,
        'learning_rate': args.learning_rate or config.model.learning_rate,
        'epochs': args.epochs or config.model.epochs,
        'weight_decay': args.weight_decay or config.model.weight_decay,
        'device': args.device
    }
    
    logger.info("\n📋 ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    logger.info(f"  Устройство: {trainer_config['device']}")
    logger.info(f"  Batch size: {trainer_config['batch_size']}")
    logger.info(f"  Learning rate: {trainer_config['learning_rate']}")
    logger.info(f"  Эпохи: {trainer_config['epochs']}")
    logger.info(f"  Weight decay: {trainer_config['weight_decay']}")
    
    # Проверка манифестов
    if not validate_data_manifests():
        raise FileNotFoundError("Не найдены необходимые манифесты данных")
    
    # Создание тренера
    logger.info("\n🔧 Инициализация тренера...")
    trainer = ModelTrainer(trainer_config)
    
    # Подготовка данных с использованием стандартных манифестов
    logger.info("\n📊 Загрузка данных...")
    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        batch_size=trainer_config['batch_size'],
        train_manifest='data/processed/data_manifest_train.csv',
        val_manifest='data/processed/data_manifest_val.csv',
        test_manifest='data/processed/data_manifest_test.csv'
    )
    
    logger.info(f"  Обучающая выборка: {len(train_loader.dataset)} изображений")
    logger.info(f"  Валидационная выборка: {len(val_loader.dataset)} изображений")
    logger.info(f"  Тестовая выборка: {len(test_loader.dataset)} изображений")
    
    # Создание модели
    logger.info("\n🏗️ Создание модели...")
    model = trainer.create_model()
    
    # Информация о модели
    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Всего параметров: {total_params:,}")
    logger.info(f"  Обучаемых параметров: {trainable_params:,}")
    if 'num_categories' in model_info:
        logger.info(f"  Категорий: {model_info['num_categories']}")
    if 'num_components' in model_info:
        logger.info(f"  Компонентов: {model_info['num_components']}")
    
    # Обучение
    logger.info("\n🚀 ЗАПУСК ОБУЧЕНИЯ...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=trainer_config['epochs'],
        save_path=args.save_path
    )
    
    # Оценка на тестовых данных
    logger.info("\n📈 ОЦЕНКА МОДЕЛИ...")
    metrics = trainer.evaluate(test_loader)
    
    # Вывод метрик
    logger.info("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Сохранение результатов
    save_training_results(trainer, history, metrics, args)
    
    # Визуализация
    if args.plot:
        plot_training_results(history, args.output_dir, metrics)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    logger.info("=" * 60)
    
    return trainer, history, metrics


def save_training_results(trainer: ModelTrainer,
                         history: Dict[str, list],
                         metrics: Dict[str, float],
                         args: argparse.Namespace) -> None:
    """
    Сохранение результатов обучения.
    
    Args:
        trainer: Обученный тренер
        history: История обучения
        metrics: Итоговые метрики
        args: Аргументы командной строки
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Формируем имя модели
    if args.save_path:
        model_path = Path(args.save_path)
    else:
        # Создаем информативное имя с метриками
        accuracy = metrics.get('test_accuracy', 0)
        loss = metrics.get('test_loss', 0)
        model_name = f"terrazite_model_acc{accuracy:.3f}_loss{loss:.3f}_{timestamp}.pth"
        model_path = output_dir / model_name
    
    # Сохраняем модель
    trainer.save_model(str(model_path))
    
    # Подготавливаем полные результаты
    results = {
        'timestamp': timestamp,
        'command_args': vars(args),
        'model_config': trainer.model.get_model_info() if hasattr(trainer.model, 'get_model_info') else {},
        'training_history': {
            'loss': history.get('loss', []),
            'val_loss': history.get('val_loss', []),
            'category_accuracy': history.get('category_accuracy', []),
            'val_category_accuracy': history.get('val_category_accuracy', []),
            'lr': history.get('lr', []),
            'epochs_completed': len(history.get('loss', []))
        },
        'test_metrics': metrics,
        'model_path': str(model_path)
    }
    
    # Сохраняем результаты в JSON
    results_path = output_dir / f"training_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Создаем краткий README с результатами
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"# Результаты обучения {timestamp}\n\n")
        f.write(f"## Метрики на тестовой выборке\n")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"- **{metric}**: {value:.4f}\n")
        f.write(f"\n## Параметры обучения\n")
        for key, value in vars(args).items():
            f.write(f"- {key}: {value}\n")
        f.write(f"\n## Файлы\n")
        f.write(f"- Модель: `{model_path.name}`\n")
        f.write(f"- Результаты: `{results_path.name}`\n")
    
    logger.info(f"\n💾 РЕЗУЛЬТАТЫ СОХРАНЕНЫ:")
    logger.info(f"  • Модель: {model_path}")
    logger.info(f"  • Результаты: {results_path}")
    logger.info(f"  • README: {readme_path}")


def plot_training_results(history: Dict[str, list],
                         output_dir: str,
                         metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Визуализация результатов обучения.
    
    Args:
        history: История обучения
        output_dir: Директория для сохранения
        metrics: Итоговые метрики для отображения
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем фигуру с тремя графиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Результаты обучения Terrazite AI - {timestamp}', fontsize=16, fontweight='bold')
    
    # 1. График потерь (Loss)
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    if 'loss' in history and history['loss']:
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    axes[0, 0].set_title('Динамика потерь (Loss)', fontsize=14)
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Добавляем итоговое значение на график
    if metrics and 'test_loss' in metrics:
        axes[0, 0].axhline(y=metrics['test_loss'], color='g', linestyle='--', alpha=0.5, 
                           label=f"Test Loss: {metrics['test_loss']:.4f}")
        axes[0, 0].legend()
    
    # 2. График точности (Accuracy)
    if 'category_accuracy' in history and history['category_accuracy']:
        axes[0, 1].plot(epochs, history['category_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    if 'val_category_accuracy' in history and history['val_category_accuracy']:
        axes[0, 1].plot(epochs, history['val_category_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    
    axes[0, 1].set_title('Точность классификации', fontsize=14)
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Добавляем итоговое значение на график
    if metrics and 'test_accuracy' in metrics:
        axes[0, 1].axhline(y=metrics['test_accuracy'], color='g', linestyle='--', alpha=0.5,
                           label=f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        axes[0, 1].legend()
    
    # 3. График скорости обучения (Learning Rate)
    if 'lr' in history and history['lr']:
        axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_title('Скорость обучения (Learning Rate)', fontsize=14)
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate Schedule\nNot Available',
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Скорость обучения', fontsize=14)
    
    # 4. Сводная информация
    axes[1, 1].axis('off')
    info_text = "📊 ИТОГОВАЯ СТАТИСТИКА\n\n"
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'accuracy' in key:
                    info_text += f"✅ {key}: {value:.2%}\n"
                elif 'loss' in key:
                    info_text += f"📉 {key}: {value:.4f}\n"
                else:
                    info_text += f"• {key}: {value:.4f}\n"
    
    info_text += f"\n⏱️ Эпох: {len(history.get('loss', []))}"
    
    axes[1, 1].text(0.1, 0.9, info_text,
                   transform=axes[1, 1].transAxes,
                   fontsize=12,
                   verticalalignment='top',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Сохраняем график
    plot_path = output_dir / f"training_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"📊 Графики сохранены: {plot_path}")


def test_predictions(trainer: ModelTrainer,
                    test_loader: torch.utils.data.DataLoader,
                    num_samples: int = 5) -> list:
    """
    Тестирование предсказаний модели на нескольких примерах.
    
    Args:
        trainer: Обученный тренер
        test_loader: Загрузчик тестовых данных
        num_samples: Количество примеров для показа
        
    Returns:
        Список примеров с предсказаниями
    """
    logger.info(f"\n🔍 ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЙ ({num_samples} примеров)...")
    
    trainer.model.eval()
    examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            images = batch['image'].to(trainer.device)
            categories = batch['category'].to(trainer.device)
            recipe_names = batch.get('name', [f"Пример_{i}"] * len(images))
            
            # Предсказание
            outputs = trainer.model(images)
            category_probs = torch.softmax(outputs['category_logits'], dim=1)
            predicted = torch.argmax(category_probs, dim=1)
            
            # Категории
            idx_to_category = {v: k for k, v in trainer.dataset.category_to_idx.items()}
            
            for j in range(len(images)):
                true_cat = categories[j].item()
                pred_cat = predicted[j].item()
                confidence = category_probs[j, pred_cat].item()
                
                example = {
                    'recipe_name': recipe_names[j] if isinstance(recipe_names, list) else recipe_names,
                    'true_category': idx_to_category.get(true_cat, f"Class_{true_cat}"),
                    'true_idx': true_cat,
                    'predicted_category': idx_to_category.get(pred_cat, f"Class_{pred_cat}"),
                    'predicted_idx': pred_cat,
                    'confidence': confidence,
                    'correct': true_cat == pred_cat
                }
                examples.append(example)
    
    # Вывод результатов
    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("=" * 80)
    
    correct_count = 0
    for i, ex in enumerate(examples, 1):
        status = "✅" if ex['correct'] else "❌"
        correct_count += 1 if ex['correct'] else 0
        
        print(f"\n{status} Пример {i}:")
        print(f"  Рецепт: {ex['recipe_name']}")
        print(f"  Истинная категория: {ex['true_category']}")
        print(f"  Предсказание: {ex['predicted_category']} (уверенность: {ex['confidence']:.2%})")
    
    accuracy = correct_count / len(examples)
    print(f"\n📊 Точность на примерах: {accuracy:.2%} ({correct_count}/{len(examples)})")
    
    return examples


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description='Обучение модели Terrazite AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/train_model.py                     # Обучение с параметрами по умолчанию
  python scripts/train_model.py --epochs 100        # Увеличить количество эпох
  python scripts/train_model.py --batch-size 16     # Уменьшить размер батча
  python scripts/train_model.py --plot              # Создать графики
  python scripts/train_model.py --device cuda       # Использовать GPU
        """
    )
    
    # Параметры данных
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к файлу конфигурации (опционально)')
    
    # Параметры модели
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Размер батча (по умолчанию: {config.model.batch_size})')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help=f'Скорость обучения (по умолчанию: {config.model.learning_rate})')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Количество эпох (по умолчанию: {config.model.epochs})')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help=f'Weight decay (по умолчанию: {config.model.weight_decay})')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Устройство для обучения (auto/cuda/cpu)')
    
    # Сохранение
    parser.add_argument('--save-path', type=str, default=None,
                       help='Путь для сохранения модели (опционально)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Директория для сохранения результатов')
    
    # Дополнительные опции
    parser.add_argument('--plot', action='store_true',
                       help='Создавать графики обучения')
    parser.add_argument('--test-samples', type=int, default=5,
                       help='Количество тестовых примеров для показа')
    parser.add_argument('--quick-test', action='store_true',
                       help='Быстрый тест (2 эпохи, малый батч)')
    
    args = parser.parse_args()
    
    # Быстрый тест
    if args.quick_test:
        args.epochs = 2
        args.batch_size = 4
        args.plot = True
        logger.info("⚡ Режим быстрого тестирования: epochs=2, batch_size=4")
    
    # Автоматическое определение устройства
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"ℹ️ Автоматически выбрано устройство: {args.device}")
    
    # Проверка GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA не доступна. Переключаюсь на CPU.")
        args.device = 'cpu'
    
    try:
        # Запуск обучения
        trainer, history, metrics = train_model(args)
        
        # Тестирование предсказаний
        if args.test_samples > 0:
            # Пересоздаем test_loader для примеров
            _, _, test_loader = trainer.prepare_dataloaders(
                batch_size=args.batch_size or config.model.batch_size,
                test_manifest='data/processed/data_manifest_test.csv'
            )
            test_predictions(trainer, test_loader, args.test_samples)
        
        # Вывод итоговой информации
        print("\n" + "=" * 80)
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 80)
        
        print("\n📊 ИТОГОВЫЕ МЕТРИКИ:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'accuracy' in key:
                    print(f"  {key}: {value:.2%}")
                elif 'loss' in key:
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:.4f}")
        
        print(f"\n📁 Результаты сохранены в: {args.output_dir}")
        
        print("\n🔍 ЧТО ДАЛЬШЕ:")
        print("  1. Запустите тестирование: python test_model_basic.py")
        print("  2. Запустите API: uvicorn src.api.main:app --reload")
        print("  3. Откройте интерфейс: streamlit run streamlit_app.py")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Ошибка: {e}")
        print("\n💡 Сначала подготовьте данные:")
        print("   python scripts/create_data_manifest.py")
        print("   python scripts/prepare_image_dataset.py")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
