"""
Pydantic схемы для валидации данных API
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Any
from datetime import datetime


class RecipeComponent(BaseModel):
    """Схема для компонента рецепта"""
    name: str
    weight_kg: float = Field(..., ge=0, description="Вес в кг")
    percentage: float = Field(..., ge=0, le=100, description="Процент от общей массы")


class PredictionRequest(BaseModel):
    """Схема запроса на предсказание"""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    include_similar: bool = True
    max_similar: int = 5


class PredictionResponse(BaseModel):
    """Схема ответа с предсказанием"""
    recipe_id: str
    aggregate_type: str
    confidence: float = Field(..., ge=0, le=100)
    components: List[RecipeComponent]
    similar_recipes: List[Dict] = []
    processing_time_ms: float


class UploadResponse(BaseModel):
    """Схема ответа на загрузку"""
    filename: str
    url: str
    recipe_id: str
    message: str


class HealthResponse(BaseModel):
    """Схема ответа health check"""
    status: str
    model_loaded: bool
    database_records: int
    timestamp: datetime


class RecipeDetailResponse(BaseModel):
    """Схема детального рецепта"""
    recipe_id: str
    name: str
    type: str
    components: Dict[str, float]
    total_weight: float
    main_aggregate: str
    image_url: Optional[str]
    created_at: datetime
