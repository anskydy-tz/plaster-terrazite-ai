"""
Модель машинного обучения для определения рецепта терразитовой штукатурки по изображению
с поддержкой категорий компонентов (без воды)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import json
from pathlib import Path
import logging

from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TerraziteModel(nn.Module):
    """
    Основная модель для определения рецепта терразитовой штукатурки
    с мультимодальной архитектурой (изображение + компоненты без воды)
    """
    
    def __init__(self, 
                 num_categories: int = 5,  # 5 категорий: Терразит, Шовный, Мастика, Терраццо, Ретушь
                 num_components: int = 58,  # ИСПРАВЛЕНО: 58 компонентов без воды
                 hidden_size: int = 512,
                 dropout_rate: float = 0.3,
                 use_pretrained: bool = True):
        """
        Инициализация модели
        
        Args:
            num_categories: Количество категорий рецептов
            num_components: Количество уникальных компонентов (без воды)
            hidden_size: Размер скрытого слоя
            dropout_rate: Rate для dropout
            use_pretrained: Использовать предобученные веса для ResNet
        """
        super(TerraziteModel, self).__init__()
        
        # Загрузка конфигурации
        self.config = config
        self.num_categories = num_categories
        self.num_components = num_components
        
        # Энкодер изображений (ResNet50)
        self.image_encoder = models.resnet50(pretrained=use_pretrained)
        
        # Заменяем последний слой ResNet
        num_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()  # Удаляем классификатор
        
        # Энкодер для компонентов (без воды)
        self.component_encoder = nn.Sequential(
            nn.Linear(num_components, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Мультимодальный слой
        self.multimodal_encoder = nn.Sequential(
            nn.Linear(num_features + hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Классификаторы
        self.category_classifier = nn.Linear(hidden_size, num_categories)  # Категория рецепта
        self.component_predictor = nn.Linear(hidden_size, num_components)  # Компоненты (без воды)
        self.recipe_classifier = nn.Linear(hidden_size, 128)  # Конкретный рецепт
        
        # Слои для регрессии компонентов (без воды)
        self.component_regressors = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_components)
        ])
        
        # Инициализация атрибутов для маппинга компонентов (без воды)
        self.component_to_idx = {}
        self.idx_to_component = {}
        self.idx_to_group = {}
        
        # Загрузка информации о компонентах (без воды)
        self._load_component_info()
        
        # Инициализация весов
        self._initialize_weights()
        
        logger.info(f"Инициализирована модель TerraziteModel с {num_categories} категориями")
        logger.info(f"Количество компонентов (без воды): {num_components}")
    
    def _initialize_weights(self):
        """Инициализация весов слоев"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_component_info(self):
        """Загрузка информации о компонентах из конфигурации (без воды)"""
        try:
            # Загружаем группы компонентов из конфигурации
            self.component_groups = self.config.data.component_groups
            
            # Инвертируем маппинг: компонент -> группа (исключая воду)
            self.component_to_group = {}
            for group_name, components in self.component_groups.items():
                for component in components:
                    # Пропускаем компоненты с водой
                    if 'вода' not in component.lower():
                        self.component_to_group[component] = group_name
            
            # Загружаем категории рецептов
            self.recipe_categories = self.config.data.recipe_categories
            
            logger.info(f"Загружено групп компонентов (без воды): {len(self.component_groups)}")
            logger.info(f"Загружено категорий рецептов: {len(self.recipe_categories)}")
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить информацию о компонентах: {e}")
            self.component_groups = {}
            self.component_to_group = {}
            self.recipe_categories = ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь']
    
    def forward(self, 
                images: torch.Tensor, 
                components: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели
        
        Args:
            images: Тензор изображений [batch_size, 3, H, W]
            components: Тензор компонентов [batch_size, num_components] или None
            
        Returns:
            Словарь с предсказаниями
        """
        # Кодирование изображения
        image_features = self.image_encoder(images)  # [batch_size, num_features]
        
        # Кодирование компонентов (если предоставлены)
        if components is not None:
            component_features = self.component_encoder(components)  # [batch_size, hidden_size]
        else:
            # Если компоненты не предоставлены, используем нулевой вектор
            batch_size = images.size(0)
            component_features = torch.zeros(batch_size, 512, device=images.device)
        
        # Объединение признаков
        combined_features = torch.cat([image_features, component_features], dim=1)  # [batch_size, num_features + hidden_size]
        
        # Мультимодальное кодирование
        multimodal_features = self.multimodal_encoder(combined_features)  # [batch_size, hidden_size]
        
        # Предсказания
        outputs = {
            'category_logits': self.category_classifier(multimodal_features),  # Категория рецепта
            'component_logits': self.component_predictor(multimodal_features),  # Все компоненты (без воды)
            'recipe_features': self.recipe_classifier(multimodal_features),  # Признаки для рецепта
            'multimodal_features': multimodal_features  # Общие признаки
        }
        
        # Регрессия для каждого компонента (без воды)
        component_predictions = []
        for i, regressor in enumerate(self.component_regressors):
            pred = regressor(multimodal_features)
            component_predictions.append(pred)
        
        outputs['component_regression'] = torch.cat(component_predictions, dim=1)
        
        return outputs
    
    def predict_category(self, 
                        images: torch.Tensor, 
                        components: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Предсказание категории рецепта
        
        Args:
            images: Тензор изображений
            components: Тензор компонентов или None
            
        Returns:
            Кортеж (предсказанные категории, вероятности)
        """
        outputs = self.forward(images, components)
        probs = F.softmax(outputs['category_logits'], dim=1)
        predicted = torch.argmax(probs, dim=1)
        
        return predicted, probs
    
    def predict_components(self, 
                          images: torch.Tensor, 
                          threshold: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Предсказание компонентов рецепта (без воды)
        
        Args:
            images: Тензор изображений
            threshold: Порог для бинаризации
            
        Returns:
            Словарь с предсказаниями компонентов
        """
        outputs = self.forward(images)
        component_probs = torch.sigmoid(outputs['component_logits'])
        component_preds = (component_probs > threshold).float()
        
        # Регрессионные предсказания (количества)
        component_values = torch.sigmoid(outputs['component_regression']) * 1000  # Масштабирование до кг
        
        return {
            'binary_predictions': component_preds,
            'probabilities': component_probs,
            'values': component_values
        }
    
    def get_component_groups_predictions(self, 
                                        component_predictions: torch.Tensor,
                                        component_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Агрегация предсказаний компонентов по группам (без воды)
        
        Args:
            component_predictions: Бинарные предсказания компонентов
            component_probs: Вероятности компонентов
            
        Returns:
            Словарь с предсказаниями по группам
        """
        if not self.component_groups:
            return {}
        
        batch_size = component_predictions.size(0)
        device = component_predictions.device
        
        group_predictions = {}
        
        # Создаем маппинг индексов компонентов
        # Для этого нужна информация о компонентах из загрузчика данных
        component_to_idx = getattr(self, 'component_to_idx', {})
        
        if not component_to_idx:
            logger.warning("Маппинг компонентов не загружен. Использую псевдогруппы.")
            # Создаем псевдогруппы на основе конфигурации
            for group_name, components in self.component_groups.items():
                if components:
                    # Просто создаем нулевые тензоры для демонстрации
                    group_predictions[group_name] = {
                        'presence': torch.zeros(batch_size, device=device),
                        'confidence': torch.zeros(batch_size, device=device),
                        'count': torch.zeros(batch_size, device=device, dtype=torch.long)
                    }
            return group_predictions
        
        # Агрегируем по группам
        for group_name, components in self.component_groups.items():
            group_indices = []
            for component in components:
                # Пропускаем компоненты с водой
                if 'вода' in component.lower():
                    continue
                    
                if component in component_to_idx:
                    group_indices.append(component_to_idx[component])
            
            if group_indices:
                # Бинарные предсказания для группы
                group_binary = component_predictions[:, group_indices]
                group_probs = component_probs[:, group_indices]
                
                # Присутствие хотя бы одного компонента из группы
                group_presence = (group_binary.sum(dim=1) > 0).float()
                
                # Средняя уверенность
                group_confidence = group_probs.mean(dim=1)
                
                # Количество компонентов в группе
                group_count = group_binary.sum(dim=1).long()
                
                group_predictions[group_name] = {
                    'presence': group_presence,
                    'confidence': group_confidence,
                    'count': group_count
                }
        
        return group_predictions
    
    def load_component_mapping(self, mapping_path: str):
        """
        Загрузка маппинга компонентов из файла (без воды)
        
        Args:
            mapping_path: Путь к JSON файлу с маппингом
        """
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            # Фильтруем компоненты с водой
            filtered_mapping = {}
            for idx_str, component in mapping_data.items():
                if 'вода' not in component.lower():
                    filtered_mapping[idx_str] = component
            
            self.component_to_idx = {v: int(k) for k, v in filtered_mapping.items()}
            self.idx_to_component = {int(k): v for k, v in filtered_mapping.items()}
            
            # Создаем обратный маппинг для групп
            self.idx_to_group = {}
            for idx, component in self.idx_to_component.items():
                self.idx_to_group[idx] = self.component_to_group.get(component, 'other')
            
            logger.info(f"Загружен маппинг для {len(self.component_to_idx)} компонентов (без воды)")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки маппинга компонентов: {e}")
    
    def load_component_mapping_from_dict(self, mapping_data: Dict):
        """
        Загрузка маппинга компонентов из словаря (без воды)
        
        Args:
            mapping_data: Словарь с маппингом компонентов
        """
        # Проверяем структуру mapping_data
        if 'component_to_idx' in mapping_data:
            # Фильтруем компоненты с водой
            component_to_idx = {}
            for component, idx in mapping_data['component_to_idx'].items():
                if 'вода' not in component.lower():
                    component_to_idx[component] = idx
            
            self.component_to_idx = component_to_idx
            self.idx_to_component = {int(v): k for k, v in component_to_idx.items()}
        else:
            # Предполагаем, что это простой маппинг idx->component
            filtered_mapping = {}
            for idx_str, component in mapping_data.items():
                if 'вода' not in component.lower():
                    filtered_mapping[idx_str] = component
            
            self.component_to_idx = {v: int(k) for k, v in filtered_mapping.items()}
            self.idx_to_component = {int(k): v for k, v in filtered_mapping.items()}
        
        # Создаем обратный маппинг для групп
        self.idx_to_group = {}
        for component, idx in self.component_to_idx.items():
            self.idx_to_group[idx] = self.component_to_group.get(component, 'other')
        
        logger.info(f"Маппинг компонентов загружен: {len(self.component_to_idx)} компонентов (без воды)")
    
    def decode_components(self, 
                         component_indices: torch.Tensor, 
                         component_values: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """
        Декодирование предсказанных компонентов в читаемый формат (без воды)
        
        Args:
            component_indices: Индексы предсказанных компонентов
            component_values: Значения компонентов (если есть)
            
        Returns:
            Список словарей с информацией о компонентах
        """
        if not hasattr(self, 'idx_to_component'):
            raise ValueError("Маппинг компонентов не загружен. Сначала вызовите load_component_mapping()")
        
        batch_size = component_indices.size(0)
        decoded = []
        
        for i in range(batch_size):
            recipe_components = []
            indices = torch.nonzero(component_indices[i]).flatten()
            
            for idx in indices:
                idx_int = idx.item()
                component_name = self.idx_to_component.get(idx_int, f"component_{idx_int}")
                component_group = self.idx_to_group.get(idx_int, 'unknown')
                
                component_info = {
                    'name': component_name,
                    'group': component_group,
                    'present': True
                }
                
                if component_values is not None:
                    component_info['value'] = component_values[i, idx].item()
                    component_info['value_kg'] = component_info['value'] * 1000  # Переводим в кг
                
                recipe_components.append(component_info)
            
            decoded.append(recipe_components)
        
        return decoded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Словарь с информацией
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'name': 'TerraziteModel',
            'num_categories': self.num_categories,
            'num_components': self.num_components,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_groups': list(self.component_groups.keys()) if self.component_groups else [],
            'recipe_categories': self.recipe_categories,
            'component_mapping_loaded': len(self.component_to_idx) > 0,
            'mapped_components': len(self.component_to_idx),
            'note': 'Модель работает только с сухими компонентами (вода исключена)'
        }
        
        return info
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Заморозка/разморозка backbone сети
        
        Args:
            freeze: True - заморозить, False - разморозить
        """
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            logger.info("Backbone сети заморожен")
        else:
            logger.info("Backbone сети разморожен")
    
    def unfreeze_backbone_layers(self, num_layers: int = 10):
        """
        Постепенная разморозка слоев backbone
        
        Args:
            num_layers: Количество слоев для разморозки
        """
        # Замораживаем все слои
        self.freeze_backbone(True)
        
        # Размораживаем последние num_layers слоев
        children = list(self.image_encoder.children())
        layers_to_unfreeze = children[-num_layers:] if num_layers < len(children) else children
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Разморожено последних {num_layers} слоев backbone")


class MultiTaskLoss(nn.Module):
    """
    Мультизадачная функция потерь для обучения модели
    """
    
    def __init__(self, 
                 category_weight: float = 1.0,
                 component_weight: float = 0.8,  # ИСПРАВЛЕНО: увеличено, так как компоненты важны
                 regression_weight: float = 0.5):  # ИСПРАВЛЕНО: увеличено для лучшей регрессии
        """
        Инициализация функции потерь
        
        Args:
            category_weight: Вес потерь для классификации категорий
            component_weight: Вес потерь для классификации компонентов
            regression_weight: Вес потерь для регрессии компонентов
        """
        super(MultiTaskLoss, self).__init__()
        self.category_weight = category_weight
        self.component_weight = component_weight
        self.regression_weight = regression_weight
        
        # Функции потерь
        self.category_loss = nn.CrossEntropyLoss()
        self.component_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = nn.MSELoss()
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Вычисление потерь
        
        Args:
            outputs: Выходы модели
            targets: Целевые значения
            
        Returns:
            Словарь с потерями
        """
        losses = {}
        
        # Потери для категорий
        if 'category' in targets:
            category_loss = self.category_loss(outputs['category_logits'], targets['category'])
            losses['category'] = category_loss * self.category_weight
        
        # Потери для компонентов (бинарная классификация)
        if 'components_binary' in targets:
            component_loss = self.component_loss(outputs['component_logits'], targets['components_binary'])
            losses['component'] = component_loss * self.component_weight
        
        # Потери для регрессии компонентов
        if 'components_values' in targets:
            regression_loss = self.regression_loss(outputs['component_regression'], targets['components_values'])
            losses['regression'] = regression_loss * self.regression_weight
        
        # Общие потери
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


class TerraziteEnsemble(nn.Module):
    """
    Ансамбль моделей для улучшения точности предсказаний (без воды)
    """
    
    def __init__(self, 
                 num_models: int = 3,
                 num_categories: int = 5,
                 num_components: int = 58):  # ИСПРАВЛЕНО: 58 компонентов без воды
        """
        Инициализация ансамбля
        
        Args:
            num_models: Количество моделей в ансамбле
            num_categories: Количество категорий
            num_components: Количество компонентов (без воды)
        """
        super(TerraziteEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            TerraziteModel(num_categories=num_categories, num_components=num_components)
            for _ in range(num_models)
        ])
        
        # Мета-классификатор для объединения предсказаний
        self.meta_classifier = nn.Linear(num_models * num_categories, num_categories)
        self.meta_component_predictor = nn.Linear(num_models * num_components, num_components)
        
    def forward(self, 
                images: torch.Tensor, 
                components: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Прямой проход ансамбля
        
        Args:
            images: Тензор изображений
            components: Тензор компонентов
            
        Returns:
            Словарь с предсказаниями
        """
        all_outputs = []
        
        # Получаем предсказания от всех моделей
        for model in self.models:
            outputs = model(images, components)
            all_outputs.append(outputs)
        
        # Объединяем предсказания категорий
        category_logits = [outputs['category_logits'] for outputs in all_outputs]
        combined_category = torch.cat(category_logits, dim=1)
        ensemble_category = self.meta_classifier(combined_category)
        
        # Объединяем предсказания компонентов
        component_logits = [outputs['component_logits'] for outputs in all_outputs]
        combined_components = torch.cat(component_logits, dim=1)
        ensemble_components = self.meta_component_predictor(combined_components)
        
        # Берем среднее по остальным выходам
        avg_multimodal = torch.mean(torch.stack([outputs['multimodal_features'] for outputs in all_outputs]), dim=0)
        avg_recipe_features = torch.mean(torch.stack([outputs['recipe_features'] for outputs in all_outputs]), dim=0)
        avg_regression = torch.mean(torch.stack([outputs['component_regression'] for outputs in all_outputs]), dim=0)
        
        return {
            'category_logits': ensemble_category,
            'component_logits': ensemble_components,
            'recipe_features': avg_recipe_features,
            'multimodal_features': avg_multimodal,
            'component_regression': avg_regression
        }
    
    def predict_with_confidence(self, 
                               images: torch.Tensor, 
                               components: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Предсказание с оценкой уверенности
        
        Args:
            images: Тензор изображений
            components: Тензор компонентов
            
        Returns:
            Словарь с предсказаниями и уверенностью
        """
        # Получаем предсказания от всех моделей
        all_category_preds = []
        all_component_preds = []
        
        for model in self.models:
            outputs = model(images, components)
            
            # Категории
            category_probs = F.softmax(outputs['category_logits'], dim=1)
            all_category_preds.append(category_probs)
            
            # Компоненты
            component_probs = torch.sigmoid(outputs['component_logits'])
            all_component_preds.append(component_probs)
        
        # Вычисляем среднее и стандартное отклонение
        category_stack = torch.stack(all_category_preds, dim=0)  # [num_models, batch_size, num_categories]
        category_mean = torch.mean(category_stack, dim=0)
        category_std = torch.std(category_stack, dim=0)
        
        component_stack = torch.stack(all_component_preds, dim=0)  # [num_models, batch_size, num_components]
        component_mean = torch.mean(component_stack, dim=0)
        component_std = torch.std(component_stack, dim=0)
        
        # Предсказанные категории
        predicted_categories = torch.argmax(category_mean, dim=1)
        
        # Уверенность (1 - стандартное отклонение)
        category_confidence = 1.0 - category_std.mean(dim=1)
        component_confidence = 1.0 - component_std.mean(dim=1)
        
        return {
            'categories': predicted_categories,
            'category_probabilities': category_mean,
            'category_confidence': category_confidence,
            'component_probabilities': component_mean,
            'component_confidence': component_confidence,
            'category_std': category_std,
            'component_std': component_std
        }
    
    def load_component_mapping_from_dict(self, mapping_data: Dict):
        """
        Загрузка маппинга компонентов из словаря для всех моделей ансамбля
        
        Args:
            mapping_data: Словарь с маппингами компонентов
        """
        for model in self.models:
            model.load_component_mapping_from_dict(mapping_data)
        
        logger.info(f"Маппинг компонентов загружен для всех {len(self.models)} моделей ансамбля")


def create_model(model_type: str = 'terrazite', **kwargs) -> nn.Module:
    """
    Фабричная функция для создания модели
    
    Args:
        model_type: Тип модели ('terrazite', 'ensemble')
        **kwargs: Дополнительные параметры
        
    Returns:
        Созданная модель
    """
    if model_type == 'terrazite':
        return TerraziteModel(**kwargs)
    elif model_type == 'ensemble':
        return TerraziteEnsemble(**kwargs)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def test_model():
    """Тестирование модели"""
    # Создаем тестовую модель
    model = TerraziteModel(num_categories=5, num_components=58)
    
    # Тестовые данные
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    components = torch.randn(batch_size, 58)
    
    # Прямой проход
    outputs = model(images, components)
    
    print("Тестирование модели (без воды):")
    print(f"  Входные изображения: {images.shape}")
    print(f"  Входные компоненты: {components.shape}")
    print(f"  Выходные категории: {outputs['category_logits'].shape}")
    print(f"  Выходные компоненты: {outputs['component_logits'].shape}")
    print(f"  Мультимодальные признаки: {outputs['multimodal_features'].shape}")
    print(f"  Регрессия компонентов: {outputs['component_regression'].shape}")
    
    # Предсказание категории
    predicted, probs = model.predict_category(images)
    print(f"  Предсказанные категории: {predicted}")
    print(f"  Вероятности: {probs.shape}")
    
    # Тестирование функции потерь
    targets = {
        'category': torch.randint(0, 5, (batch_size,)),
        'components_binary': (components > 0).float(),
        'components_values': components
    }
    
    criterion = MultiTaskLoss()
    losses = criterion(outputs, targets)
    print(f"  Потери: {losses}")
    
    # Информация о модели
    info = model.get_model_info()
    print(f"  Всего параметров: {info['total_parameters']:,}")
    print(f"  Обучаемых параметров: {info['trainable_parameters']:,}")
    print(f"  Группы компонентов: {len(info['component_groups'])}")
    print(f"  Примечание: {info['note']}")
    
    # Тестирование ансамбля
    ensemble = TerraziteEnsemble(num_models=2, num_categories=5, num_components=58)
    ensemble_outputs = ensemble(images, components)
    print(f"\nАнсамбль (без воды):")
    print(f"  Категории: {ensemble_outputs['category_logits'].shape}")
    print(f"  Компоненты: {ensemble_outputs['component_logits'].shape}")
    
    return model


if __name__ == "__main__":
    # Запуск теста
    model = test_model()
    print("\nМодель успешно протестирована!")
