"""
Обработчики для API эндпоинтов
"""
import json
import logging
import time
from typing import Dict, List, Optional
from fastapi import UploadFile, HTTPException
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


async def predict_recipe_handler(model, image_file: UploadFile, include_similar: bool, 
                               max_similar: int, recipe_database: List) -> Dict:
    """Обработчик предсказания рецепта"""
    start_time = time.time()
    
    try:
        # Чтение и обработка изображения
        image_data = await image_file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Ресайз до размера модели
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        
        # Предсказание
        prediction = model.predict(image_array)
        
        # Поиск похожих рецептов
        similar = []
        if include_similar and recipe_database:
            similar = find_similar_recipes(prediction, recipe_database, max_similar)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "recipe_id": generate_recipe_id(prediction),
            "aggregate_type": prediction['aggregate_type'],
            "confidence": prediction['confidence'],
            "components": format_components(prediction['recipe_components']),
            "similar_recipes": similar,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def upload_image_handler(image: UploadFile, recipe_data: Optional[str]) -> Dict:
    """Обработчик загрузки изображения"""
    try:
        # Сохраняем изображение
        filename = f"{int(time.time())}_{image.filename}"
        save_path = f"static/uploads/{filename}"
        
        with open(save_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Если есть данные рецепта, сохраняем их
        recipe_id = None
        if recipe_data:
            recipe_dict = json.loads(recipe_data)
            recipe_id = recipe_dict.get('recipe_id', f"REC_{int(time.time())}")
            save_recipe_data(recipe_id, recipe_dict, filename)
        
        return {
            "filename": filename,
            "url": f"/static/uploads/{filename}",
            "recipe_id": recipe_id or "not_provided",
            "message": "Файл успешно загружен"
        }
        
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_recipe_handler(recipe_id: str, recipe_database: List) -> Dict:
    """Обработчик получения рецепта по ID"""
    for recipe in recipe_database:
        if recipe.get('recipe_id') == recipe_id:
            return recipe
    
    raise HTTPException(status_code=404, detail=f"Рецепт {recipe_id} не найден")


def get_similar_recipes_handler(recipe_id: str, recipe_database: List, max_results: int) -> List[Dict]:
    """Обработчик поиска похожих рецептов"""
    target_recipe = None
    for recipe in recipe_database:
        if recipe.get('recipe_id') == recipe_id:
            target_recipe = recipe
            break
    
    if not target_recipe:
        raise HTTPException(status_code=404, detail=f"Рецепт {recipe_id} не найден")
    
    # Простой алгоритм поиска похожих (можно улучшить)
    similar = []
    for recipe in recipe_database:
        if recipe['recipe_id'] != recipe_id:
            similarity = calculate_similarity(target_recipe, recipe)
            if similarity > 0.5:  # Порог схожести
                similar.append({
                    **recipe,
                    "similarity_score": similarity
                })
    
    # Сортировка по схожести
    similar.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return similar[:max_results]


def health_check_handler(model_loaded: bool) -> Dict:
    """Обработчик проверки здоровья"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "database_records": 0,  # Можно добавить реальное количество
        "timestamp": time.time()
    }


# Вспомогательные функции
def generate_recipe_id(prediction: Dict) -> str:
    """Генерация ID рецепта на основе предсказания"""
    agg_type = prediction['aggregate_type'][:3].upper()
    timestamp = int(time.time()) % 10000
    return f"PRED_{agg_type}_{timestamp}"


def format_components(components: List[float]) -> List[Dict]:
    """Форматирование компонентов рецепта"""
    # Заглушка - нужно сопоставить с реальными названиями компонентов
    component_names = [
        "Цемент белый", "Цемент серый", "Известь", 
        "Песок 0-0.63мм", "Доломит", "Мрамор белый",
        "Мрамор черный", "Пигмент красный", "Пигмент желтый",
        "Пигмент зеленый", "Пластификатор", "Метилцеллюлоза"
    ]
    
    formatted = []
    for i, (name, weight) in enumerate(zip(component_names, components)):
        if weight > 0:  # Показываем только ненулевые компоненты
            formatted.append({
                "name": name,
                "weight_kg": weight,
                "percentage": (weight / sum(components)) * 100 if sum(components) > 0 else 0
            })
    
    return formatted


def find_similar_recipes(prediction: Dict, database: List, max_results: int) -> List[Dict]:
    """Поиск похожих рецептов в базе"""
    # Простая реализация - можно улучшить
    similar = []
    for recipe in database[:max_results]:  # Пока берем первые N
        similar.append({
            "recipe_id": recipe.get("recipe_id", "unknown"),
            "name": recipe.get("name", "Неизвестный рецепт"),
            "similarity_score": 0.8  # Заглушка
        })
    
    return similar


def save_recipe_data(recipe_id: str, recipe_data: Dict, image_filename: str):
    """Сохранение данных рецепта"""
    # Здесь будет сохранение в базу данных или файл
    pass


def calculate_similarity(recipe1: Dict, recipe2: Dict) -> float:
    """Расчет схожести рецептов"""
    # Простая реализация - можно улучшить
    return 0.7  # Заглушка
