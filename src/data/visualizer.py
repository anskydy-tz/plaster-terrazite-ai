"""
Модуль для визуализации данных
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Класс для визуализации данных рецептов"""
    
    def __init__(self):
        self.figsize = (12, 8)
        
    def plot_recipe_components(self, recipes: List[Dict], top_n: int = 10):
        """Визуализация компонентов рецептов"""
        # Сбор статистики по компонентам
        component_stats = {}
        for recipe in recipes:
            for component, value in recipe.get('components', {}).items():
                if component not in component_stats:
                    component_stats[component] = []
                component_stats[component].append(value)
        
        # Сортировка по среднему значению
        sorted_components = sorted(
            component_stats.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )[:top_n]
        
        # Построение графика
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Средние значения компонентов
        components = [c[0] for c in sorted_components]
        means = [np.mean(c[1]) for c in sorted_components]
        axes[0, 0].barh(components, means)
        axes[0, 0].set_title(f'Топ-{top_n} компонентов (среднее значение)')
        axes[0, 0].set_xlabel('Среднее количество')
        
        # 2. Распределение весов
        for component, values in sorted_components[:5]:
            axes[0, 1].hist(values, alpha=0.5, label=component, bins=20)
        axes[0, 1].set_title('Распределение весов компонентов')
        axes[0, 1].set_xlabel('Вес')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].legend()
        
        # 3. Корреляционная матрица (для топ компонентов)
        if len(sorted_components) > 1:
            corr_matrix = np.corrcoef([c[1] for c in sorted_components])
            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Корреляция между компонентами')
            axes[1, 0].set_xticks(range(len(components)))
            axes[1, 0].set_yticks(range(len(components)))
            axes[1, 0].set_xticklabels(components, rotation=45, ha='right')
            axes[1, 0].set_yticklabels(components)
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Количество рецептов по типу
        if 'type' in recipes[0]:
            types = [r.get('type', 'unknown') for r in recipes]
            type_counts = {t: types.count(t) for t in set(types)}
            axes[1, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Распределение по типам рецептов')
        
        plt.tight_layout()
        return fig
    
    def plot_model_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               feature_names: Optional[List[str]] = None):
        """Визуализация предсказаний модели"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Сравнение предсказанных и истинных значений
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('Истинные значения')
        axes[0, 0].set_ylabel('Предсказанные значения')
        axes[0, 0].set_title('Предсказания vs Истинные значения')
        
        # 2. Ошибки предсказания
        errors = y_pred - y_true
        axes[0, 1].hist(errors, bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Ошибка предсказания')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].set_title('Распределение ошибок')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        
        # 3. Квантиль-квантиль график
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q график ошибок')
        
        # 4. Абсолютные ошибки по компонентам
        if feature_names and len(feature_names) == y_true.shape[1]:
            mae = np.mean(np.abs(y_pred - y_true), axis=0)
            axes[1, 1].barh(range(len(feature_names)), mae)
            axes[1, 1].set_yticks(range(len(feature_names)))
            axes[1, 1].set_yticklabels(feature_names)
            axes[1, 1].set_xlabel('Средняя абсолютная ошибка (MAE)')
            axes[1, 1].set_title('Ошибки по компонентам')
        
        plt.tight_layout()
        return fig
