"""
Модуль для аугментации изображений
"""
import numpy as np
from typing import List, Tuple, Optional
import logging
import albumentations as A

logger = logging.getLogger(__name__)

class DataAugmentor:
    """Класс для аугментации изображений штукатурки"""
    
    def __init__(self, augment: bool = True):
        self.augment = augment
        
        # Базовые аугментации (всегда применяются)
        self.base_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225)),
        ])
        
        # Аугментации для тренировочных данных
        if augment:
            self.train_transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                                 rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                         contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, 
                                   sat_shift_limit=30, 
                                   val_shift_limit=20, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, 
                              max_width=32, fill_value=0, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), 
                           std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.train_transform = self.base_transform
    
    def augment_image(self, image: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Применение аугментаций к изображению"""
        try:
            if is_training and self.augment:
                augmented = self.train_transform(image=image)
            else:
                augmented = self.base_transform(image=image)
            return augmented['image']
        except Exception as e:
            logger.error(f"Ошибка аугментации изображения: {e}")
            # Возвращаем изображение с базовой трансформацией
            try:
                augmented = self.base_transform(image=image)
                return augmented['image']
            except:
                return image
    
    def augment_batch(self, images: List[np.ndarray], 
                      is_training: bool = True) -> List[np.ndarray]:
        """Аугментация батча изображений"""
        augmented_images = []
        for image in images:
            augmented_images.append(self.augment_image(image, is_training))
        return augmented_images
    
    def create_augmentation_pipeline(self, config: Optional[dict] = None) -> A.Compose:
        """Создание кастомного пайплайна аугментаций"""
        if config is None:
            config = {
                'flip': True,
                'rotate': True,
                'color': True,
                'noise': True,
                'dropout': True
            }
        
        transforms = []
        
        # Всегда ресайз и нормализация
        transforms.append(A.Resize(224, 224))
        
        # Добавляем выбранные аугментации
        if config.get('flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
            transforms.append(A.VerticalFlip(p=0.5))
        
        if config.get('rotate', False):
            transforms.append(A.RandomRotate90(p=0.5))
            transforms.append(A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ))
        
        if config.get('color', False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ))
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ))
        
        if config.get('noise', False):
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
            transforms.append(A.Blur(blur_limit=3, p=0.3))
        
        if config.get('dropout', False):
            transforms.append(A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3
            ))
        
        # Всегда нормализация в конце
        transforms.append(A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ))
        
        return A.Compose(transforms)
