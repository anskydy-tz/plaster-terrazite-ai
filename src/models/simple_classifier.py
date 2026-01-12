"""
Упрощенный классификатор для быстрого прототипирования.
Используется, когда данных мало.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

logger = logging.getLogger(__name__)


class SimpleAggregateClassifier:
    """
    Простой классификатор на основе случайного леса для определения
    типа декоративного заполнителя по гистограммам цвета.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.classes_ = []
        
    def extract_color_histogram(self, image, bins=32):
        """
        Извлечение гистограммы цветов из изображения.
        """
        # Разделяем на каналы
        channels = []
        for i in range(3):
            channel = image[:, :, i]
            hist, _ = np.histogram(channel, bins=bins, range=(0, 256))
            channels.append(hist)
        # Объединяем
        features = np.concatenate(channels)
        return features
    
    def fit(self, X, y):
        """
        Обучение модели.
        
        Args:
            X: список изображений (numpy arrays)
            y: список меток (тип заполнителя)
        """
        logger.info("Извлечение гистограмм цветов...")
        # Преобразуем изображения в гистограммы
        X_features = []
        for img in X:
            features = self.extract_color_histogram(img)
            X_features.append(features)
        
        X_features = np.array(X_features)
        self.feature_names = [f'hist_{i}' for i in range(X_features.shape[1])]
        
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        logger.info(f"Обучение RandomForest на {len(X)} образцах...")
        self.model.fit(X_features, y_encoded)
        
        logger.info(f"Классы: {self.classes_}")
        return self
    
    def predict(self, X):
        """
        Предсказание меток для изображений.
        """
        X_features = []
        for img in X:
            features = self.extract_color_histogram(img)
            X_features.append(features)
        
        X_features = np.array(X_features)
        y_encoded = self.model.predict(X_features)
        y = self.label_encoder.inverse_transform(y_encoded)
        return y
    
    def predict_proba(self, X):
        """
        Вероятности для каждого класса.
        """
        X_features = []
        for img in X:
            features = self.extract_color_histogram(img)
            X_features.append(features)
        
        X_features = np.array(X_features)
        return self.model.predict_proba(X_features)
    
    def evaluate(self, X, y):
        """
        Оценка точности модели.
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"Точность: {accuracy:.2%}")
        logger.info("\nОтчет по классификации:")
        logger.info(classification_report(y, y_pred))
        
        return accuracy
    
    def save(self, path):
        """
        Сохранение модели и энкодера.
        """
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'classes': self.classes_
        }, path)
        logger.info(f"Модель сохранена в {path}")
    
    def load(self, path):
        """
        Загрузка модели и энкодера.
        """
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        self.classes_ = data['classes']
        logger.info(f"Модель загружена из {path}")


if __name__ == "__main__":
    # Пример использования
    print("🧪 Тестирование SimpleAggregateClassifier")
    print("=" * 50)
    
    # Создаем случайные данные для теста
    num_samples = 50
    images = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(num_samples)]
    labels = np.random.choice(['мрамор', 'кварц', 'гранит'], size=num_samples)
    
    # Создаем и обучаем модель
    clf = SimpleAggregateClassifier(n_estimators=50)
    clf.fit(images, labels)
    
    # Предсказание
    test_images = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
    predictions = clf.predict(test_images)
    
    print(f"Предсказания: {predictions}")
    print("\n✅ Классификатор готов!")
