"""
Модуль для оценки и интерпретации моделей.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
import tensorflow as tf
import logging
from typing import Dict, List, Tuple, Optional
import json
import os

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Класс для оценки производительности моделей и визуализации результатов.
    """
    
    def __init__(self, model=None, model_path=None):
        self.model = model
        if model_path and not model:
            self.load_model(model_path)
        
        # Настройка стиля графиков
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_model(self, model_path: str):
        """Загрузка модели из файла."""
        if model_path.endswith('.h5'):
            self.model = tf.keras.models.load_model(model_path)
        else:
            # Предполагаем, что это simple_classifier
            import joblib
            data = joblib.load(model_path)
            self.model = data['model']
        
        logger.info(f"Модель загружена из {model_path}")
    
    def evaluate_classification(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Оценка классификационной части модели.
        
        Args:
            X_test: Тестовые изображения
            y_test: Истинные метки (one-hot encoded)
            class_names: Названия классов
            save_dir: Директория для сохранения графиков
        
        Returns:
            Словарь с метриками
        """
        logger.info("Оценка классификации...")
        
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        # Предсказания
        if hasattr(self.model, 'predict'):
            # Для Keras моделей
            predictions = self.model.predict(X_test)
            # Если модель многозадачная, берем только классификационный выход
            if isinstance(predictions, list):
                y_pred_proba = predictions[1]  # classification_output
            else:
                y_pred_proba = predictions
            
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Для sklearn моделей
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # Вычисление метрик
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Визуализация матрицы ошибок
        self.plot_confusion_matrix(cm, class_names, save_dir)
        
        # Визуализация примеров ошибок
        if save_dir:
            self.plot_misclassified_examples(X_test, y_true, y_pred, class_names, save_dir)
        
        logger.info(f"Точность: {report['accuracy']:.2%}")
        logger.info(f"Отчет:\n{json.dumps(report, indent=2)}")
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist()
        }
    
    def evaluate_regression(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        component_names: List[str],
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Оценка регрессионной части модели.
        
        Args:
            X_test: Тестовые изображения
            y_test: Истинные значения компонентов (нормированные)
            component_names: Названия компонентов
            save_dir: Директория для сохранения графиков
        
        Returns:
            Словарь с метриками
        """
        logger.info("Оценка регрессии...")
        
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        # Предсказания
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_test)
            # Если модель многозадачная, берем только регрессионный выход
            if isinstance(predictions, list):
                y_pred = predictions[0]  # regression_output
            else:
                y_pred = predictions
        else:
            raise ValueError("Модель не поддерживает регрессию")
        
        # Вычисление метрик для каждого компонента
        mse_per_component = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        mae_per_component = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        
        # Общие метрики
        total_mse = mean_squared_error(y_test, y_pred)
        total_mae = mean_absolute_error(y_test, y_pred)
        total_rmse = np.sqrt(total_mse)
        
        # Визуализация
        if save_dir:
            self.plot_regression_results(y_test, y_pred, component_names, save_dir)
        
        logger.info(f"Общая MSE: {total_mse:.4f}")
        logger.info(f"Общая MAE: {total_mae:.4f}")
        logger.info(f"Общая RMSE: {total_rmse:.4f}")
        
        return {
            'mse_per_component': dict(zip(component_names, mse_per_component.tolist())),
            'mae_per_component': dict(zip(component_names, mae_per_component.tolist())),
            'total_mse': float(total_mse),
            'total_mae': float(total_mae),
            'total_rmse': float(total_rmse),
            'predictions': y_pred.tolist(),
            'true_values': y_test.tolist()
        }
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_dir: Optional[str] = None
    ):
        """Визуализация матрицы ошибок."""
        plt.figure(figsize=(10, 8))
        
        # Тепловая карта
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Матрица ошибок классификации')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'))
        
        plt.show()
        plt.close()
    
    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        component_names: List[str],
        save_dir: Optional[str] = None
    ):
        """Визуализация результатов регрессии."""
        num_components = y_true.shape[1]
        
        # Ограничиваем количество компонентов для визуализации
        max_components = min(10, num_components)
        
        # Создаем сетку графиков
        fig, axes = plt.subplots(2, (max_components + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(max_components):
            ax = axes[i]
            
            # Scatter plot истинных vs предсказанных значений
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            
            # Линия идеального предсказания
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальное предсказание')
            
            ax.set_xlabel('Истинные значения')
            ax.set_ylabel('Предсказанные значения')
            ax.set_title(f'{component_names[i]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Убираем лишние оси
        for i in range(max_components, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Предсказание компонентов рецепта', fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'regression_results.png'), dpi=300)
            plt.savefig(os.path.join(save_dir, 'regression_results.pdf'))
        
        plt.show()
        plt.close()
    
    def plot_misclassified_examples(
        self,
        X_test: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_dir: str,
        num_examples: int = 5
    ):
        """Визуализация примеров ошибок классификации."""
        # Находим индексы ошибок
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            logger.info("Нет ошибок классификации для визуализации")
            return
        
        # Ограничиваем количество примеров
        num_examples = min(num_examples, len(misclassified_idx))
        selected_idx = misclassified_idx[:num_examples]
        
        # Создаем график
        fig, axes = plt.subplots(1, num_examples, figsize=(15, 4))
        if num_examples == 1:
            axes = [axes]
        
        for i, idx in enumerate(selected_idx):
            ax = axes[i]
            
            # Показываем изображение
            if len(X_test.shape) == 4:
                img = X_test[idx]
                if img.shape[-1] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap='gray')
            else:
                # Если изображения не в правильном формате, пропускаем
                ax.text(0.5, 0.5, f"Ошибка\n{class_names[y_true[idx]]} → {class_names[y_pred[idx]]}",
                       ha='center', va='center', fontsize=12)
            
            ax.set_title(f"Истинный: {class_names[y_true[idx]]}\nПредсказанный: {class_names[y_pred[idx]]}")
            ax.axis('off')
        
        plt.suptitle(f'Примеры ошибок классификации ({len(misclassified_idx)} всего)', fontsize=14)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'misclassified_examples.png'), dpi=300)
            plt.savefig(os.path.join(save_dir, 'misclassified_examples.pdf'))
        
        plt.show()
        plt.close()
    
    def generate_report(
        self,
        classification_results: Dict,
        regression_results: Dict,
        save_path: str = 'reports/evaluation_report.json'
    ):
        """Генерация полного отчета об оценке."""
        report = {
            'timestamp': np.datetime64('now').astype(str),
            'classification': classification_results,
            'regression': regression_results,
            'summary': {
                'classification_accuracy': classification_results['classification_report']['accuracy'],
                'regression_mse': regression_results['total_mse'],
                'regression_mae': regression_results['total_mae']
            }
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Отчет сохранен в {save_path}")
        return report


if __name__ == "__main__":
    # Пример использования
    print("🧪 Тестирование ModelEvaluator")
    print("=" * 50)
    
    # Создаем тестовые данные
    num_samples = 100
    X_test = np.random.rand(num_samples, 224, 224, 3)
    
    # Классификация: 5 классов
    y_cls_true = np.random.randint(0, 5, size=num_samples)
    y_cls_test = tf.keras.utils.to_categorical(y_cls_true, 5)
    
    # Регрессия: 15 компонентов
    y_reg_test = np.random.rand(num_samples, 15)
    
    class_names = ['мрамор', 'кварц', 'гранит', 'слюда', 'известняк']
    component_names = [f'компонент_{i}' for i in range(15)]
    
    # Создаем заглушку модели (в реальности нужно загрузить обученную модель)
    from .terrazite_model import TerraziteRecipeModel
    model = TerraziteRecipeModel()
    model.build_model()
    
    # Создаем evaluator
    evaluator = ModelEvaluator(model=model)
    
    # Оценка классификации
    print("Оценка классификации...")
    cls_results = evaluator.evaluate_classification(
        X_test[:10],  # берем только 10 для скорости
        y_cls_test[:10],
        class_names,
        save_dir='test_evaluation'
    )
    
    # Оценка регрессии
    print("\nОценка регрессии...")
    reg_results = evaluator.evaluate_regression(
        X_test[:10],
        y_reg_test[:10],
        component_names,
        save_dir='test_evaluation'
    )
    
    # Генерация отчета
    report = evaluator.generate_report(cls_results, reg_results, 'test_evaluation/report.json')
    
    print("\n✅ ModelEvaluator готов к работе!")
