"""
PyTorch тренер для обучения моделей терразитовой штукатурки.
Поддерживает многозадачное обучение (категории + компоненты)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from ..utils.config import config, setup_config
from ..utils.logger import setup_logger
from ..data.loader import TerraziteDataset
from .terrazite_model import TerraziteModel, MultiTaskLoss, create_model

logger = setup_logger(__name__)


class ModelTrainer:
    """
    PyTorch тренер для обучения моделей терразитовой штукатурки
    с поддержкой категорий и компонентов
    """
    
    def __init__(self, trainer_config: Optional[Dict[str, Any]] = None):
        """
        Инициализация тренера
        
        Args:
            trainer_config: Конфигурация обучения
        """
        # Загрузка конфигурации
        self.config = config
        self.trainer_config = self._load_trainer_config(trainer_config)
        
        # Устройство
        self.device = self._setup_device(self.trainer_config.get('device', 'auto'))
        
        # Модель и компоненты
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Трекеры
        self.writer = None
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_category_acc': [], 'val_category_acc': [],
            'train_component_f1': [], 'val_component_f1': [],
            'learning_rates': []
        }
    
    def _load_trainer_config(self, trainer_config: Optional[Dict]) -> Dict[str, Any]:
        """Загрузка конфигурации обучения"""
        default_config = {
            'device': 'auto',
            'batch_size': config.model.batch_size,
            'learning_rate': config.model.learning_rate,
            'weight_decay': config.model.weight_decay,
            'epochs': config.model.epochs,
            'early_stopping_patience': config.model.early_stopping_patience,
            'gradient_clip': 1.0,
            'warmup_epochs': 3,
            'save_frequency': config.training.save_frequency,
            'log_frequency': 10,
            'num_workers': 4,
            'pin_memory': True,
            'mixed_precision': True,
            
            # Веса потерь
            'category_weight': config.model.category_weight,
            'component_weight': config.model.component_weight,
            'regression_weight': config.model.regression_weight,
            
            # Настройки оптимизатора
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'scheduler_params': {
                'T_max': 100,
                'eta_min': 1e-6
            },
            
            # Аугментация
            'augmentation': config.data.augmentation_enabled,
            'augmentation_params': {
                'rotation_range': config.data.rotation_range,
                'width_shift_range': config.data.width_shift_range,
                'height_shift_range': config.data.height_shift_range,
                'horizontal_flip': config.data.horizontal_flip
            }
        }
        
        if trainer_config:
            default_config.update(trainer_config)
        
        return default_config
    
    def _setup_device(self, device_str: str) -> torch.device:
        """Настройка устройства для обучения"""
        if device_str == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_str)
        
        logger.info(f"Используется устройство: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA версия: {torch.version.cuda}")
            logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        return device
    
    def create_model(self, 
                    model_type: str = 'terrazite',
                    **model_kwargs) -> nn.Module:
        """
        Создание модели
        
        Args:
            model_type: Тип модели ('terrazite', 'ensemble')
            **model_kwargs: Дополнительные параметры модели
            
        Returns:
            Созданная модель
        """
        logger.info(f"Создание модели типа: {model_type}")
        
        # Параметры модели из конфигурации
        model_params = {
            'num_categories': self.config.model.num_categories,
            'num_components': self.config.model.num_components,
            'hidden_size': self.config.model.hidden_size,
            'dropout_rate': self.config.model.dropout_rate,
            'use_pretrained': self.config.model.use_pretrained
        }
        
        # Обновление пользовательскими параметрами
        model_params.update(model_kwargs)
        
        # Создание модели
        self.model = create_model(model_type, **model_params)
        self.model.to(self.device)
        
        # Загрузка маппинга компонентов
        self._load_component_mapping()
        
        # Создание функции потерь
        self.criterion = MultiTaskLoss(
            category_weight=self.trainer_config['category_weight'],
            component_weight=self.trainer_config['component_weight'],
            regression_weight=self.trainer_config['regression_weight']
        ).to(self.device)
        
        # Создание оптимизатора
        self._create_optimizer()
        
        # Информация о модели
        self._log_model_info()
        
        return self.model
    
    def _create_optimizer(self):
        """Создание оптимизатора"""
        optimizer_name = self.trainer_config['optimizer'].lower()
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.trainer_config['learning_rate'],
                weight_decay=self.trainer_config['weight_decay']
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.trainer_config['learning_rate'],
                weight_decay=self.trainer_config['weight_decay']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.trainer_config['learning_rate'],
                momentum=0.9,
                weight_decay=self.trainer_config['weight_decay']
            )
        else:
            logger.warning(f"Неизвестный оптимизатор {optimizer_name}, использую AdamW")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.trainer_config['learning_rate'],
                weight_decay=self.trainer_config['weight_decay']
            )
        
        logger.info(f"Оптимизатор: {optimizer_name}, lr={self.trainer_config['learning_rate']}")
    
    def _create_scheduler(self):
        """Создание планировщика скорости обучения"""
        scheduler_name = self.trainer_config['scheduler'].lower()
        scheduler_params = self.trainer_config.get('scheduler_params', {})
        
        if scheduler_name == 'cosine':
            scheduler_params.setdefault('T_max', self.trainer_config['epochs'])
            scheduler_params.setdefault('eta_min', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **scheduler_params
            )
        elif scheduler_name == 'plateau':
            scheduler_params.setdefault('mode', 'min')
            scheduler_params.setdefault('patience', 5)
            scheduler_params.setdefault('factor', 0.5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **scheduler_params
            )
        elif scheduler_name == 'step':
            scheduler_params.setdefault('step_size', 30)
            scheduler_params.setdefault('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                **scheduler_params
            )
        else:
            logger.warning(f"Неизвестный scheduler {scheduler_name}, не использую scheduler")
            self.scheduler = None
    
    def _load_component_mapping(self):
        """Загрузка маппинга компонентов"""
        try:
            ml_data_path = Path(self.config.project_root) / self.config.data.processed_data_dir / self.config.data.ml_data_file
            if ml_data_path.exists():
                with open(ml_data_path, 'r', encoding='utf-8') as f:
                    ml_data = json.load(f)
                
                if 'component_mapping' in ml_data:
                    mapping_data = ml_data['component_mapping']
                    self.model.load_component_mapping_from_dict(mapping_data)
                    logger.info(f"Маппинг компонентов загружен: {len(mapping_data.get('component_to_idx', {}))} компонентов")
        except Exception as e:
            logger.warning(f"Не удалось загрузить маппинг компонентов: {e}")
    
    def _log_model_info(self):
        """Логирование информации о модели"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Модель создана: {self.model.__class__.__name__}")
        logger.info(f"Всего параметров: {total_params:,}")
        logger.info(f"Обучаемых параметров: {trainable_params:,}")
        logger.info(f"Категорий: {self.config.model.num_categories}")
        logger.info(f"Компонентов: {self.config.model.num_components}")
        
        # Логирование архитектуры
        logger.debug("Архитектура модели:")
        for name, module in self.model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            logger.debug(f"  {name}: {module.__class__.__name__}, параметров: {num_params:,}")
    
    def prepare_dataloaders(self, 
                          dataset_path: Optional[str] = None,
                          batch_size: Optional[int] = None,
                          **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Подготовка DataLoader для обучения, валидации и тестирования
        
        Args:
            dataset_path: Путь к датасету
            batch_size: Размер батча
            
        Returns:
            Кортеж (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or self.trainer_config['batch_size']
        
        # Загрузка рецептов
        from ..data.loader import RecipeLoader
        recipe_loader = RecipeLoader()
        
        # Попытка загрузить Excel файл
        excel_path = Path(self.config.project_root) / self.config.data.excel_file
        if not excel_path.exists():
            # Поиск альтернативных путей
            possible_paths = [
                excel_path,
                Path("data/raw/recipes.xlsx"),
                Path("Рецептуры терразит.xlsx")
            ]
            for path in possible_paths:
                if path.exists():
                    excel_path = path
                    break
        
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel файл не найден: {self.config.data.excel_file}")
        
        logger.info(f"Загрузка данных из: {excel_path}")
        recipe_loader.load_excel(str(excel_path))
        
        # Получение всех рецептов
        recipes = recipe_loader.get_all_recipes()
        
        # Создание датасета
        dataset = TerraziteDataset(
            recipes_data=recipes,
            image_dir=self.config.data.images_dir,
            transform=self._create_transforms('train'),
            include_components=True
        )
        
        # Разделение данных
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Создание DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.trainer_config['num_workers'],
            pin_memory=self.trainer_config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.trainer_config['num_workers'],
            pin_memory=self.trainer_config['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.trainer_config['num_workers'],
            pin_memory=self.trainer_config['pin_memory']
        )
        
        logger.info(f"DataLoader созданы:")
        logger.info(f"  Train: {len(train_loader.dataset)} образцов")
        logger.info(f"  Val: {len(val_loader.dataset)} образцов")
        logger.info(f"  Test: {len(test_loader.dataset)} образцов")
        
        return train_loader, val_loader, test_loader
    
    def _create_transforms(self, mode: str = 'train'):
        """
        Создание трансформаций для данных
        
        Args:
            mode: Режим ('train', 'val', 'test')
            
        Returns:
            Композиция трансформаций или None
        """
        from torchvision import transforms
        
        if mode == 'train' and self.trainer_config['augmentation']:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(self.trainer_config['augmentation_params']['rotation_range']),
                transforms.RandomHorizontalFlip(
                    self.trainer_config['augmentation_params'].get('horizontal_flip', 0.5)
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Обучение на одной эпохе
        
        Args:
            train_loader: DataLoader для обучения
            
        Returns:
            Словарь с метриками эпохи
        """
        self.model.train()
        total_loss = 0
        category_correct = 0
        category_total = 0
        component_correct = 0
        component_total = 0
        
        # Прогресс-бар
        pbar = tqdm(train_loader, desc=f"Эпоха {self.current_epoch + 1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Перемещение данных на устройство
            images = batch['image'].to(self.device)
            categories = batch['category'].to(self.device)
            components = batch['components'].to(self.device)
            
            # Подготовка целей для многозадачного обучения
            targets = {
                'category': categories,
                'components_binary': (components > 0).float(),
                'components_values': components
            }
            
            # Прямой проход
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Вычисление потерь
            losses = self.criterion(outputs, targets)
            loss = losses['total']
            
            # Обратный проход
            loss.backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.trainer_config['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Метрики
            total_loss += loss.item()
            
            # Точность категорий
            category_preds = torch.argmax(outputs['category_logits'], dim=1)
            category_correct += (category_preds == categories).sum().item()
            category_total += categories.size(0)
            
            # F1 для компонентов
            component_probs = torch.sigmoid(outputs['component_logits'])
            component_preds = (component_probs > 0.5).float()
            component_correct += (component_preds == targets['components_binary']).sum().item()
            component_total += targets['components_binary'].numel()
            
            # Обновление прогресс-бара
            if batch_idx % self.trainer_config['log_frequency'] == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'cat_acc': category_correct / max(category_total, 1),
                    'comp_acc': component_correct / max(component_total, 1)
                })
        
        # Вычисление средних метрик
        metrics = {
            'loss': total_loss / len(train_loader),
            'category_accuracy': category_correct / max(category_total, 1),
            'component_accuracy': component_correct / max(component_total, 1),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Валидация модели
        
        Args:
            val_loader: DataLoader для валидации
            
        Returns:
            Словарь с метриками валидации
        """
        self.model.eval()
        total_loss = 0
        category_correct = 0
        category_total = 0
        component_correct = 0
        component_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация", leave=False):
                # Перемещение данных на устройство
                images = batch['image'].to(self.device)
                categories = batch['category'].to(self.device)
                components = batch['components'].to(self.device)
                
                # Подготовка целей
                targets = {
                    'category': categories,
                    'components_binary': (components > 0).float(),
                    'components_values': components
                }
                
                # Прямой проход
                outputs = self.model(images)
                
                # Вычисление потерь
                losses = self.criterion(outputs, targets)
                loss = losses['total']
                
                # Метрики
                total_loss += loss.item()
                
                # Точность категорий
                category_preds = torch.argmax(outputs['category_logits'], dim=1)
                category_correct += (category_preds == categories).sum().item()
                category_total += categories.size(0)
                
                # F1 для компонентов
                component_probs = torch.sigmoid(outputs['component_logits'])
                component_preds = (component_probs > 0.5).float()
                component_correct += (component_preds == targets['components_binary']).sum().item()
                component_total += targets['components_binary'].numel()
        
        # Вычисление средних метрик
        metrics = {
            'loss': total_loss / len(val_loader),
            'category_accuracy': category_correct / max(category_total, 1),
            'component_accuracy': component_correct / max(component_total, 1)
        }
        
        return metrics
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: Optional[int] = None,
             save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Полный цикл обучения модели
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            epochs: Количество эпох
            save_path: Путь для сохранения модели
            
        Returns:
            История обучения
        """
        epochs = epochs or self.trainer_config['epochs']
        save_path = save_path or Path(self.config.training.checkpoint_dir) / "best_model.pth"
        
        # Создание планировщика
        self._create_scheduler()
        
        # Инициализация TensorBoard
        self._init_tensorboard()
        
        # Ранняя остановка
        early_stopping_counter = 0
        best_val_loss = float('inf')
        
        logger.info(f"Начало обучения на {epochs} эпох")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_metrics = self.validate(val_loader)
            
            # Обновление планировщика
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Логирование метрик
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Сохранение истории
            self._update_history(train_metrics, val_metrics)
            
            # Сохранение лучшей модели
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(save_path)
                early_stopping_counter = 0
                logger.info(f"Новая лучшая модель! Val Loss: {best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
            
            # Сохранение чекпоинта
            if (epoch + 1) % self.trainer_config['save_frequency'] == 0:
                checkpoint_path = Path(self.config.training.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics['loss'])
            
            # Ранняя остановка
            if early_stopping_counter >= self.trainer_config['early_stopping_patience']:
                logger.info(f"Ранняя остановка на эпохе {epoch+1}")
                break
        
        # Закрытие TensorBoard
        if self.writer:
            self.writer.close()
        
        logger.info(f"Обучение завершено. Лучший Val Loss: {best_val_loss:.4f}")
        
        return self.history
    
    def _init_tensorboard(self):
        """Инициализация TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(self.config.training.tensorboard_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard логи в: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard не установлен. Пропускаю инициализацию.")
            self.writer = None
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Логирование метрик в TensorBoard и консоль"""
        # Консоль
        logger.info(
            f"Эпоха {epoch + 1:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Cat Acc: {train_metrics['category_accuracy']:.3f} | "
            f"Val Cat Acc: {val_metrics['category_accuracy']:.3f} | "
            f"LR: {train_metrics.get('learning_rate', 0):.6f}"
        )
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/category_train', train_metrics['category_accuracy'], epoch)
            self.writer.add_scalar('Accuracy/category_val', val_metrics['category_accuracy'], epoch)
            self.writer.add_scalar('Accuracy/component_train', train_metrics['component_accuracy'], epoch)
            self.writer.add_scalar('Accuracy/component_val', val_metrics['component_accuracy'], epoch)
            
            if 'learning_rate' in train_metrics:
                self.writer.add_scalar('Learning_rate', train_metrics['learning_rate'], epoch)
    
    def _update_history(self, train_metrics: Dict, val_metrics: Dict):
        """Обновление истории обучения"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_category_acc'].append(train_metrics['category_accuracy'])
        self.history['val_category_acc'].append(val_metrics['category_accuracy'])
        self.history['train_component_f1'].append(train_metrics['component_accuracy'])
        self.history['val_component_f1'].append(val_metrics['component_accuracy'])
        
        if 'learning_rate' in train_metrics:
            self.history['learning_rates'].append(train_metrics['learning_rate'])
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Оценка модели на тестовых данных
        
        Args:
            test_loader: DataLoader для тестирования
            
        Returns:
            Словарь с метриками
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Тестирование"):
                # Перемещение данных на устройство
                images = batch['image'].to(self.device)
                categories = batch['category'].to(self.device)
                components = batch['components'].to(self.device)
                
                # Подготовка целей
                targets = {
                    'category': categories,
                    'components_binary': (components > 0).float(),
                    'components_values': components
                }
                
                # Прямой проход
                outputs = self.model(images)
                
                # Вычисление потерь
                losses = self.criterion(outputs, targets)
                total_loss += losses['total'].item()
                
                # Сбор предсказаний
                category_probs = torch.softmax(outputs['category_logits'], dim=1)
                category_preds = torch.argmax(category_probs, dim=1)
                
                component_probs = torch.sigmoid(outputs['component_logits'])
                component_preds = (component_probs > 0.5).float()
                
                all_predictions.append({
                    'categories': category_preds.cpu().numpy(),
                    'components': component_preds.cpu().numpy()
                })
                
                all_targets.append({
                    'categories': categories.cpu().numpy(),
                    'components': targets['components_binary'].cpu().numpy()
                })
        
        # Агрегация метрик
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        all_cat_preds = np.concatenate([p['categories'] for p in all_predictions])
        all_cat_targets = np.concatenate([t['categories'] for t in all_targets])
        
        all_comp_preds = np.concatenate([p['components'].reshape(-1) for p in all_predictions])
        all_comp_targets = np.concatenate([t['components'].reshape(-1) for t in all_targets])
        
        # Вычисление метрик
        category_accuracy = accuracy_score(all_cat_targets, all_cat_preds)
        component_accuracy = accuracy_score(all_comp_targets, all_comp_preds)
        
        metrics = {
            'test_loss': total_loss / len(test_loader),
            'test_category_accuracy': category_accuracy,
            'test_component_accuracy': component_accuracy,
            'test_category_f1': f1_score(all_cat_targets, all_cat_preds, average='weighted'),
            'test_component_f1': f1_score(all_comp_targets, all_comp_preds, average='binary'),
            'category_report': classification_report(all_cat_targets, all_cat_preds, output_dict=True),
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        logger.info(f"Оценка завершена:")
        logger.info(f"  Test Loss: {metrics['test_loss']:.4f}")
        logger.info(f"  Category Accuracy: {metrics['test_category_accuracy']:.4f}")
        logger.info(f"  Component Accuracy: {metrics['test_component_accuracy']:.4f}")
        
        return metrics
    
    def save_model(self, path: str):
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict() if hasattr(self.criterion, 'state_dict') else None,
            'history': self.history,
            'config': {
                'model': self.config.model.__dict__,
                'trainer': self.trainer_config
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Модель сохранена: {path}")
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """
        Сохранение чекпоинта
        
        Args:
            path: Путь для сохранения
            epoch: Номер эпохи
            val_loss: Loss на валидации
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Чекпоинт сохранен: {path}")
    
    def load_model(self, path: str):
        """
        Загрузка модели
        
        Args:
            path: Путь к файлу модели
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Загрузка состояния модели
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Модель загружена из: {path}")
        logger.info(f"Эпоха: {self.current_epoch}")
    
    def predict(self, 
               images: torch.Tensor,
               threshold: float = 0.5) -> Dict[str, Any]:
        """
        Предсказание для одного или нескольких изображений
        
        Args:
            images: Тензор изображений [batch_size, 3, H, W]
            threshold: Порог для компонентов
            
        Returns:
            Словарь с предсказаниями
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            
            # Обработка предсказаний
            category_probs = torch.softmax(outputs['category_logits'], dim=1)
            category_preds = torch.argmax(category_probs, dim=1)
            category_confidences = torch.max(category_probs, dim=1).values
            
            component_probs = torch.sigmoid(outputs['component_logits'])
            component_preds = (component_probs > threshold).float()
            component_confidences = component_probs
            
            # Регрессия компонентов
            component_values = torch.sigmoid(outputs['component_regression']) * 1000
            
            predictions = {
                'categories': category_preds.cpu().numpy(),
                'category_probabilities': category_probs.cpu().numpy(),
                'category_confidences': category_confidences.cpu().numpy(),
                'components_binary': component_preds.cpu().numpy(),
                'components_probabilities': component_probs.cpu().numpy(),
                'components_values': component_values.cpu().numpy()
            }
        
        return predictions


def create_trainer(trainer_config: Optional[Dict[str, Any]] = None) -> ModelTrainer:
    """
    Фабричная функция для создания тренера
    
    Args:
        trainer_config: Конфигурация обучения
        
    Returns:
        Объект ModelTrainer
    """
    return ModelTrainer(trainer_config)


def test_trainer():
    """Тестирование тренера"""
    logger.info("Тестирование ModelTrainer...")
    
    # Создаем тренер
    trainer = ModelTrainer({
        'batch_size': 4,
        'epochs': 2,
        'learning_rate': 0.001
    })
    
    # Создаем модель
    model = trainer.create_model()
    
    # Создаем тестовые данные
    from torch.utils.data import TensorDataset, DataLoader
    
    # Тестовые данные (2 образца)
    images = torch.randn(2, 3, 224, 224)
    categories = torch.randint(0, 5, (2,))
    components = torch.randn(2, trainer.config.model.num_components)
    
    dataset = TensorDataset(images, categories, components)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Тест обучения на одной эпохе
    trainer.model.train()
    for batch in dataloader:
        img, cat, comp = batch
        img, cat, comp = img.to(trainer.device), cat.to(trainer.device), comp.to(trainer.device)
        
        targets = {
            'category': cat,
            'components_binary': (comp > 0).float(),
            'components_values': comp
        }
        
        outputs = trainer.model(img)
        losses = trainer.criterion(outputs, targets)
        
        logger.info(f"Test passed! Loss: {losses['total'].item():.4f}")
        break
    
    logger.info("Тест тренера завершен успешно!")
    
    return trainer


if __name__ == "__main__":
    # Тестирование тренера
    trainer = test_trainer()
    
    # Пример использования
    print("\nПример использования тренера:")
    print("1. Создать тренер: trainer = ModelTrainer(config)")
    print("2. Создать модель: trainer.create_model()")
    print("3. Подготовить данные: train_loader, val_loader, test_loader = trainer.prepare_dataloaders()")
    print("4. Обучить: history = trainer.train(train_loader, val_loader, epochs=50)")
    print("5. Оценить: metrics = trainer.evaluate(test_loader)")
    print("6. Сохранить: trainer.save_model('models/best_model.pth')")
