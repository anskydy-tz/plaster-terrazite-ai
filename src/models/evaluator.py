"""
Модуль для оценки PyTorch моделей Terrazite AI
с визуализацией результатов и анализом ошибок
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    mean_squared_error, mean_absolute_error, 
    f1_score, accuracy_score, precision_recall_curve,
    roc_curve, auc
)
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PyTorchEvaluator:
    """
    Оценщик для PyTorch моделей Terrazite AI
    с поддержкой многозадачной оценки (категории + компоненты)
    """
    
    def __init__(self, model=None, device='auto'):
        """
        Инициализация оценщика
        
        Args:
            model: PyTorch модель
            device: Устройство для вычислений
        """
        self.model = model
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        logger.info(f"Инициализирован PyTorchEvaluator на устройстве: {self.device}")
    
    def evaluate_model(self, 
                      test_loader,
                      category_names: Optional[List[str]] = None,
                      component_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Полная оценка модели на тестовых данных
        
        Args:
            test_loader: DataLoader с тестовыми данными
            category_names: Названия категорий
            component_names: Названия компонентов
            
        Returns:
            Словарь с метриками оценки
        """
        if self.model is None:
            raise ValueError("Модель не установлена")
        
        self.model.eval()
        
        # Получение предсказаний
        all_predictions, all_targets = self._get_predictions(test_loader)
        
        # Оценка категорий
        category_metrics = self.evaluate_categories(
            all_predictions['categories'],
            all_targets['categories'],
            category_names
        )
        
        # Оценка компонентов
        component_metrics = self.evaluate_components(
            all_predictions['components'],
            all_targets['components'],
            component_names
        )
        
        # Композитные метрики
        composite_metrics = self._calculate_composite_metrics(
            category_metrics, component_metrics
        )
        
        # Сбор всех метрик
        metrics = {
            'category_metrics': category_metrics,
            'component_metrics': component_metrics,
            'composite_metrics': composite_metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        logger.info(f"Оценка завершена:")
        logger.info(f"  Точность категорий: {category_metrics['accuracy']:.4f}")
        logger.info(f"  F1 категорий: {category_metrics['f1_weighted']:.4f}")
        logger.info(f"  Точность компонентов: {component_metrics['accuracy']:.4f}")
        logger.info(f"  Композитный score: {composite_metrics['composite_score']:.4f}")
        
        return metrics
    
    def _get_predictions(self, data_loader):
        """
        Получение предсказаний модели
        
        Args:
            data_loader: DataLoader с данными
            
        Returns:
            Кортеж (predictions, targets)
        """
        all_category_preds = []
        all_category_targets = []
        all_component_preds = []
        all_component_targets = []
        all_recipe_names = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                # Перемещение данных на устройство
                images = batch['image'].to(self.device)
                categories = batch['category'].to(self.device)
                components = batch['components'].to(self.device)
                recipe_names = batch['name']
                
                # Прямой проход
                outputs = self.model(images)
                
                # Категории
                category_probs = torch.softmax(outputs['category_logits'], dim=1)
                category_preds = torch.argmax(category_probs, dim=1)
                
                # Компоненты (бинарные)
                component_probs = torch.sigmoid(outputs['component_logits'])
                component_preds = (component_probs > 0.5).float()
                
                # Сохранение
                all_category_preds.append(category_preds.cpu().numpy())
                all_category_targets.append(categories.cpu().numpy())
                all_component_preds.append(component_preds.cpu().numpy())
                all_component_targets.append((components > 0).float().cpu().numpy())
                all_recipe_names.extend(recipe_names)
        
        # Объединение батчей
        predictions = {
            'categories': np.concatenate(all_category_preds),
            'components': np.concatenate(all_component_preds),
            'recipe_names': all_recipe_names
        }
        
        targets = {
            'categories': np.concatenate(all_category_targets),
            'components': np.concatenate(all_component_targets)
        }
        
        return predictions, targets
    
    def evaluate_categories(self, 
                          predictions: np.ndarray,
                          targets: np.ndarray,
                          category_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Оценка классификации категорий
        
        Args:
            predictions: Предсказанные категории
            targets: Истинные категории
            category_names: Названия категорий
            
        Returns:
            Словарь с метриками категорий
        """
        if category_names is None:
            category_names = [f'Категория_{i}' for i in range(config.model.num_categories)]
        
        # Вычисление метрик
        accuracy = accuracy_score(targets, predictions)
        f1_weighted = f1_score(targets, predictions, average='weighted')
        f1_macro = f1_score(targets, predictions, average='macro')
        
        # Матрица ошибок
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(
            targets, predictions, 
            target_names=category_names, 
            output_dict=True
        )
        
        # Подробные метрики по классам
        class_metrics = {}
        for i, class_name in enumerate(category_names):
            # Бинарные метрики для каждого класса
            binary_preds = (predictions == i).astype(int)
            binary_targets = (targets == i).astype(int)
            
            if np.sum(binary_targets) > 0:  # Если класс присутствует в данных
                class_f1 = f1_score(binary_targets, binary_preds)
                class_precision = report.get(str(i), {}).get('precision', 0)
                class_recall = report.get(str(i), {}).get('recall', 0)
            else:
                class_f1 = class_precision = class_recall = 0.0
            
            class_metrics[class_name] = {
                'f1': float(class_f1),
                'precision': float(class_precision),
                'recall': float(class_recall),
                'support': int(np.sum(binary_targets))
            }
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_metrics': class_metrics,
            'num_classes': len(category_names)
        }
        
        return metrics
    
    def evaluate_components(self,
                          predictions: np.ndarray,
                          targets: np.ndarray,
                          component_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Оценка предсказания компонентов
        
        Args:
            predictions: Предсказанные компоненты
            targets: Истинные компоненты
            component_names: Названия компонентов
            
        Returns:
            Словарь с метриками компонентов
        """
        if predictions.ndim == 2 and targets.ndim == 2:
            # Вычисление метрик для каждого компонента
            num_components = predictions.shape[1]
            
            if component_names is None:
                component_names = [f'Компонент_{i}' for i in range(num_components)]
            
            # Общие метрики
            accuracy = accuracy_score(targets.flatten(), predictions.flatten())
            f1_micro = f1_score(targets.flatten(), predictions.flatten(), average='micro')
            f1_macro = f1_score(targets.flatten(), predictions.flatten(), average='macro')
            
            # Метрики по компонентам
            component_metrics = {}
            for i in range(num_components):
                comp_preds = predictions[:, i]
                comp_targets = targets[:, i]
                
                if np.sum(comp_targets) > 0:  # Если компонент встречается
                    comp_f1 = f1_score(comp_targets, comp_preds)
                    comp_precision = precision_score(comp_targets, comp_preds, zero_division=0)
                    comp_recall = recall_score(comp_targets, comp_preds, zero_division=0)
                else:
                    comp_f1 = comp_precision = comp_recall = 0.0
                
                component_name = component_names[i] if i < len(component_names) else f'Компонент_{i}'
                component_metrics[component_name] = {
                    'f1': float(comp_f1),
                    'precision': float(comp_precision),
                    'recall': float(comp_recall),
                    'presence_rate': float(np.mean(comp_targets)),
                    'prediction_rate': float(np.mean(comp_preds))
                }
            
            metrics = {
                'accuracy': float(accuracy),
                'f1_micro': float(f1_micro),
                'f1_macro': float(f1_macro),
                'component_metrics': component_metrics,
                'num_components': num_components
            }
            
            return metrics
        
        else:
            logger.warning("Неверная размерность данных компонентов")
            return {
                'accuracy': 0.0,
                'f1_micro': 0.0,
                'f1_macro': 0.0,
                'component_metrics': {},
                'num_components': 0
            }
    
    def _calculate_composite_metrics(self, 
                                   category_metrics: Dict, 
                                   component_metrics: Dict) -> Dict[str, float]:
        """
        Вычисление композитных метрик
        
        Args:
            category_metrics: Метрики категорий
            component_metrics: Метрики компонентов
            
        Returns:
            Словарь с композитными метриками
        """
        # Веса задач (из конфигурации)
        category_weight = config.model.category_weight
        component_weight = config.model.component_weight
        total_weight = category_weight + component_weight
        
        # Нормализация весов
        category_weight_norm = category_weight / total_weight
        component_weight_norm = component_weight / total_weight
        
        # Композитный score
        composite_score = (
            category_metrics['accuracy'] * category_weight_norm +
            component_metrics['accuracy'] * component_weight_norm
        )
        
        # Дополнительные метрики
        balanced_accuracy = np.sqrt(
            category_metrics['accuracy'] * component_metrics['accuracy']
        )
        
        # Эффективность модели (чем выше, тем лучше)
        efficiency = (
            category_metrics['f1_weighted'] * 0.4 +
            component_metrics['f1_macro'] * 0.3 +
            balanced_accuracy * 0.3
        )
        
        return {
            'composite_score': float(composite_score),
            'balanced_accuracy': float(balanced_accuracy),
            'efficiency': float(efficiency),
            'category_weight': float(category_weight_norm),
            'component_weight': float(component_weight_norm)
        }
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray,
                            class_names: List[str],
                            save_path: Optional[str] = None):
        """
        Построение матрицы ошибок
        
        Args:
            cm: Матрица ошибок
            class_names: Названия классов
            save_path: Путь для сохранения
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Тепловая карта
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title('Матрица ошибок классификации категорий')
        ax.set_ylabel('Истинный класс')
        ax.set_xlabel('Предсказанный класс')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Матрица ошибок сохранена: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_component_analysis(self,
                              component_metrics: Dict[str, Any],
                              save_path: Optional[str] = None):
        """
        Визуализация анализа компонентов
        
        Args:
            component_metrics: Метрики компонентов
            save_path: Путь для сохранения
        """
        if not component_metrics.get('component_metrics'):
            logger.warning("Нет данных для анализа компонентов")
            return
        
        metrics_dict = component_metrics['component_metrics']
        
        # Создание DataFrame
        df_data = []
        for comp_name, metrics in metrics_dict.items():
            df_data.append({
                'Компонент': comp_name,
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Presence Rate': metrics['presence_rate'],
                'Prediction Rate': metrics['prediction_rate']
            })
        
        df = pd.DataFrame(df_data)
        
        # Сортировка по F1 score
        df = df.sort_values('F1', ascending=False)
        
        # Создание графиков
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score по компонентам', 'Precision-Recall Balance',
                          'Частота присутствия', 'Соотношение Precision/Recall'),
            vertical_spacing=0.15
        )
        
        # График 1: F1 Score
        fig.add_trace(
            go.Bar(
                x=df['Компонент'][:20],  # Топ-20 компонентов
                y=df['F1'][:20],
                name='F1 Score',
                marker_color='royalblue'
            ),
            row=1, col=1
        )
        
        # График 2: Precision vs Recall
        fig.add_trace(
            go.Scatter(
                x=df['Precision'],
                y=df['Recall'],
                mode='markers',
                text=df['Компонент'],
                marker=dict(
                    size=10,
                    color=df['F1'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="F1 Score")
                ),
                name='Precision-Recall'
            ),
            row=1, col=2
        )
        
        # Добавляем линию идеального баланса
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # График 3: Частота присутствия
        fig.add_trace(
            go.Bar(
                x=df['Компонент'][:20],
                y=df['Presence Rate'][:20],
                name='Presence Rate',
                marker_color='coral'
            ),
            row=2, col=1
        )
        
        # График 4: Соотношение Precision/Recall
        fig.add_trace(
            go.Scatter(
                x=df['Компонент'][:15],
                y=df['Precision'][:15],
                mode='lines+markers',
                name='Precision',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Компонент'][:15],
                y=df['Recall'][:15],
                mode='lines+markers',
                name='Recall',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        # Обновление layout
        fig.update_layout(
            title_text="Анализ предсказания компонентов",
            height=900,
            showlegend=True
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_xaxes(title_text="Компоненты", tickangle=45, row=2, col=2)
        
        fig.update_yaxes(title_text="F1 Score", row=1, col=1)
        fig.update_yaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Частота", row=2, col=1)
        fig.update_yaxes(title_text="Значение", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Анализ компонентов сохранен: {save_path}")
        
        fig.show()
    
    def generate_report(self, 
                       metrics: Dict[str, Any],
                       output_dir: str = 'reports/evaluation') -> str:
        """
        Генерация полного отчета об оценке
        
        Args:
            metrics: Метрики оценки
            output_dir: Директория для сохранения
            
        Returns:
            Путь к отчету
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. JSON отчет
        json_path = output_path / 'evaluation_metrics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        
        # 2. Текстовый отчет
        txt_path = output_path / 'evaluation_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ ОБ ОЦЕНКЕ МОДЕЛИ TERRAZITE AI\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая информация
            f.write("ОБЩИЕ МЕТРИКИ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Дата оценки: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Композитный score: {metrics['composite_metrics']['composite_score']:.4f}\n")
            f.write(f"Эффективность модели: {metrics['composite_metrics']['efficiency']:.4f}\n\n")
            
            # Категории
            f.write("КАТЕГОРИИ РЕЦЕПТОВ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Точность: {metrics['category_metrics']['accuracy']:.4f}\n")
            f.write(f"F1 (weighted): {metrics['category_metrics']['f1_weighted']:.4f}\n")
            f.write(f"F1 (macro): {metrics['category_metrics']['f1_macro']:.4f}\n\n")
            
            # Компоненты
            f.write("КОМПОНЕНТЫ РЕЦЕПТОВ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Точность: {metrics['component_metrics']['accuracy']:.4f}\n")
            f.write(f"F1 (micro): {metrics['component_metrics']['f1_micro']:.4f}\n")
            f.write(f"F1 (macro): {metrics['component_metrics']['f1_macro']:.4f}\n")
            f.write(f"Оценено компонентов: {metrics['component_metrics']['num_components']}\n\n")
            
            # Лучшие/худшие категории
            f.write("ТОП-3 КАТЕГОРИЙ ПО F1 SCORE:\n")
            f.write("-" * 40 + "\n")
            class_metrics = metrics['category_metrics']['class_metrics']
            sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
            
            for i, (class_name, class_metrics) in enumerate(sorted_classes[:3]):
                f.write(f"{i+1}. {class_name}: F1={class_metrics['f1']:.4f}, "
                       f"Precision={class_metrics['precision']:.4f}, "
                       f"Recall={class_metrics['recall']:.4f}\n")
            
            f.write("\nРЕКОМЕНДАЦИИ:\n")
            f.write("-" * 40 + "\n")
            
            # Анализ и рекомендации
            if metrics['category_metrics']['accuracy'] < 0.7:
                f.write("• Точность категорий низкая. Рекомендуется:\n")
                f.write("  - Увеличить количество обучающих данных\n")
                f.write("  - Настроить гиперпараметры модели\n")
                f.write("  - Проверить баланс классов\n")
            
            if metrics['component_metrics']['f1_macro'] < 0.5:
                f.write("• Качество предсказания компонентов низкое. Рекомендуется:\n")
                f.write("  - Увеличить вес компонентной задачи (component_weight)\n")
                f.write("  - Добавить больше данных с аннотированными компонентами\n")
                f.write("  - Использовать балансировку для редких компонентов\n")
            
            if metrics['composite_metrics']['composite_score'] >= 0.8:
                f.write("• Модель показывает отличные результаты! Готова к использованию.\n")
        
        # 3. Визуализации
        if 'confusion_matrix' in metrics['category_metrics']:
            cm = np.array(metrics['category_metrics']['confusion_matrix'])
            class_names = list(metrics['category_metrics']['class_metrics'].keys())
            
            self.plot_confusion_matrix(
                cm,
                class_names,
                str(output_path / 'confusion_matrix.png')
            )
        
        if 'component_metrics' in metrics:
            self.plot_component_analysis(
                metrics['component_metrics'],
                str(output_path / 'component_analysis.html')
            )
        
        logger.info(f"Отчет об оценке сохранен в: {output_path}")
        
        return str(output_path)


def create_evaluator(model=None, device='auto'):
    """
    Фабричная функция для создания оценщика
    
    Args:
        model: PyTorch модель
        device: Устройство
        
    Returns:
        Объект PyTorchEvaluator
    """
    return PyTorchEvaluator(model, device)


if __name__ == "__main__":
    """Тестирование оценщика"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.models.terrazite_model import TerraziteModel
    
    logger.info("Тестирование PyTorchEvaluator...")
    
    # Создаем тестовую модель
    model = TerraziteModel(num_categories=5, num_components=50)
    
    # Создаем оценщик
    evaluator = PyTorchEvaluator(model)
    
    # Создаем тестовые данные
    batch_size = 10
    test_predictions = {
        'categories': np.random.randint(0, 5, batch_size),
        'components': np.random.randint(0, 2, (batch_size, 50))
    }
    
    test_targets = {
        'categories': np.random.randint(0, 5, batch_size),
        'components': np.random.randint(0, 2, (batch_size, 50))
    }
    
    # Оценка категорий
    category_names = ['Терразит', 'Шовный', 'Мастика', 'Терраццо', 'Ретушь']
    category_metrics = evaluator.evaluate_categories(
        test_predictions['categories'],
        test_targets['categories'],
        category_names
    )
    
    logger.info(f"Точность категорий: {category_metrics['accuracy']:.4f}")
    
    logger.info("✅ PyTorchEvaluator готов к работе!")
