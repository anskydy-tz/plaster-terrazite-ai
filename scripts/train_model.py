#!/usr/bin/env python3
"""
Скрипт для обучения модели Terrazite AI.
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

from src.models.trainer import ModelTrainer
from src.utils.config import config, setup_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_model(args):
    """Основная функция обучения модели."""
    logger.info(f"Начало обучения модели Terrazite AI")
    logger.info(f"Параметры: {args}")
    
    # Настройка конфигурации
    setup_config(args.config)
    
    # Создаем тренер
    trainer_config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'device': args.device
    }
    
    trainer = ModelTrainer(trainer_config)
    
    # Подготавливаем данные
    logger.info("Подготовка данных...")
    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        batch_size=args.batch_size
    )
    
    # Создаем модель
    logger.info("Создание модели...")
    model = trainer.create_model()
    
    # Обучение
    logger.info("Обучение модели...")
    history = trainer.train(
        train_loader, 
        val_loader,
        epochs=args.epochs,
        save_path=args.save_path
    )
    
    # Оценка
    logger.info("Оценка модели на тестовых данных...")
    metrics = trainer.evaluate(test_loader)
    
    # Сохранение результатов
    save_training_results(trainer, history, metrics, args)
    
    # Визуализация
    if args.plot:
        plot_training_results(history, args.output_dir)
    
    logger.info(f"Обучение завершено!")
    
    return trainer, history, metrics


def save_training_results(trainer, history, metrics, args):
    """Сохранение результатов обучения."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Сохраняем историю обучения
    history_path = output_dir / f"training_history_{timestamp}.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'history': history,
            'args': vars(args),
            'timestamp': timestamp,
            'metrics': metrics
        }, f, indent=2)
    
    # 2. Сохраняем метрики
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # 3. Сохраняем конфигурацию модели
    model_info = trainer.model.get_model_info() if hasattr(trainer.model, 'get_model_info') else {}
    config_path = output_dir / f"model_config_{timestamp}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2)
    
    # 4. Сохраняем модель (если не сохранена автоматически)
    if args.save_path:
        model_path = Path(args.save_path)
    else:
        model_path = output_dir / f"terrazite_model_{timestamp}.pth"
    
    trainer.save_model(str(model_path))
    
    logger.info(f"Результаты сохранены в {output_dir}")
    logger.info(f"  История: {history_path}")
    logger.info(f"  Метрики: {metrics_path}")
    logger.info(f"  Модель: {model_path}")


def plot_training_results(history, output_dir):
    """Визуализация результатов обучения."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['category_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_category_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Category Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule (если есть)
    if 'lr' in history:
        axes[1, 0].plot(history['lr'], label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Пустой график для будущих метрик
    axes[1, 1].text(0.5, 0.5, 'Additional Metrics\nWill Appear Here', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Additional Metrics')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Training Results - {timestamp}', fontsize=16)
    plt.tight_layout()
    
    # Сохраняем график
    plot_path = output_dir / f"training_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Графики сохранены: {plot_path}")


def test_predictions(trainer, test_loader, num_samples=3):
    """Тестирование предсказаний модели."""
    logger.info(f"Тестирование предсказаний на {num_samples} примерах...")
    
    trainer.model.eval()
    
    examples = []
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
        
        # Получаем данные
        images = batch['image'].to(trainer.device)
        categories = batch['category'].to(trainer.device)
        recipe_names = batch['name']
        
        # Предсказание
        with torch.no_grad():
            outputs = trainer.model(images)
            category_probs = torch.softmax(outputs['category_logits'], dim=1)
            predicted_categories = torch.argmax(category_probs, dim=1)
        
        # Собираем информацию
        for j in range(len(images)):
            example = {
                'recipe_name': recipe_names[j],
                'true_category': categories[j].item(),
                'predicted_category': predicted_categories[j].item(),
                'confidence': category_probs[j, predicted_categories[j]].item()
            }
            examples.append(example)
    
    # Выводим результаты
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЙ")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nПример {i}:")
        print(f"  Рецепт: {example['recipe_name']}")
        print(f"  Истинная категория: {example['true_category']}")
        print(f"  Предсказанная категория: {example['predicted_category']}")
        print(f"  Уверенность: {example['confidence']:.4f}")
        
        if example['true_category'] == example['predicted_category']:
            print(f"  ✅ ПРЕДСКАЗАНИЕ ВЕРНОЕ")
        else:
            print(f"  ❌ ПРЕДСКАЗАНИЕ НЕВЕРНОЕ")
    
    return examples


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description='Обучение модели Terrazite AI')
    
    # Параметры данных
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к файлу конфигурации')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Директория с изображениями')
    parser.add_argument('--recipes-json', type=str, default=None,
                       help='JSON файл с рецептами')
    
    # Параметры модели
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер батча')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Скорость обучения')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Вес decay для optimizer')
    parser.add_argument('--device', type=str, default='auto',
                       help='Устройство для обучения (cuda/cpu/auto)')
    
    # Сохранение
    parser.add_argument('--save-path', type=str, default=None,
                       help='Путь для сохранения модели')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Директория для сохранения результатов')
    
    # Дополнительные опции
    parser.add_argument('--plot', action='store_true',
                       help='Создавать графики обучения')
    parser.add_argument('--test-only', action='store_true',
                       help='Только тестирование без обучения')
    parser.add_argument('--test-samples', type=int, default=3,
                       help='Количество тестовых примеров для показа')
    
    args = parser.parse_args()
    
    # Автоматическое определение устройства
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Создание пути для сохранения
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_path = f"checkpoints/terrazite_model_{timestamp}.pth"
    
    # Проверка наличия GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA не доступна. Переключаюсь на CPU.")
        args.device = 'cpu'
    
    print(f"\n{'='*80}")
    print("НАСТРОЙКИ ОБУЧЕНИЯ")
    print("="*80)
    print(f"Устройство: {args.device}")
    print(f"Батч: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Эпохи: {args.epochs}")
    print(f"Вес decay: {args.weight_decay}")
    print(f"Сохранение: {args.save_path}")
    
    try:
        if args.test_only:
            # Только тестирование
            logger.info("Режим тестирования...")
            trainer = ModelTrainer()
            
            # Загрузка модели
            if Path(args.save_path).exists():
                trainer.load_model(args.save_path)
                logger.info(f"Модель загружена: {args.save_path}")
            else:
                logger.error(f"Файл модели не найден: {args.save_path}")
                return
            
            # Подготовка данных
            _, _, test_loader = trainer.prepare_dataloaders(batch_size=args.batch_size)
            
            # Тестирование
            metrics = trainer.evaluate(test_loader)
            
            # Показать несколько примеров
            test_predictions(trainer, test_loader, args.test_samples)
            
        else:
            # Полное обучение
            trainer, history, metrics = train_model(args)
            
            # Показать несколько примеров
            _, _, test_loader = trainer.prepare_dataloaders(batch_size=args.batch_size)
            test_predictions(trainer, test_loader, args.test_samples)
        
        print("\n" + "="*80)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*80)
        
        if 'test_accuracy' in metrics:
            print(f"Точность на тесте: {metrics['test_accuracy']:.4f}")
        if 'test_loss' in metrics:
            print(f"Loss на тесте: {metrics['test_loss']:.4f}")
        
        print(f"\nМодель сохранена: {args.save_path}")
        print(f"Результаты сохранены в: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
