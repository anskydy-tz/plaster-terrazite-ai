"""
FastAPI приложение для Terrazite AI API
Поддержка категорий компонентов и анализа рецептов
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config, setup_config
from src.utils.logger import setup_logger
from src.data.loader import RecipeLoader, DataLoader
from src.data.component_analyzer import ComponentAnalyzer
from src.models.terrazite_model import TerraziteModel, create_model

# Настройка логгера
logger = setup_logger(__name__)

# Создание приложения FastAPI
app = FastAPI(
    title="Terrazite AI API",
    description="API для подбора рецепта терразитовой штукатурки по изображению",
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для хранения состояния
model = None
recipe_loader = None
component_analyzer = None


def get_model():
    """Зависимость для получения модели"""
    global model
    if model is None:
        logger.info("Инициализация модели...")
        model = create_model(
            model_type='terrazite',
            num_categories=config.model.num_categories,
            num_components=config.model.num_components,
            hidden_size=config.model.hidden_size,
            dropout_rate=config.model.dropout_rate
        )
        # Загрузка весов, если есть
        checkpoint_dir = Path(config.project_root) / config.training.checkpoint_dir
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.pth"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                model.load_state_dict(torch.load(latest_checkpoint))
                logger.info(f"Модель загружена из {latest_checkpoint}")
    return model


def get_recipe_loader():
    """Зависимость для получения загрузчика рецептов"""
    global recipe_loader
    if recipe_loader is None:
        logger.info("Инициализация загрузчика рецептов...")
        # Ищем Excel файл
        excel_path = Path(config.project_root) / config.data.excel_file
        if not excel_path.exists():
            # Пробуем другие пути
            possible_paths = [
                excel_path,
                Path("data/raw/recipes.xlsx"),
                Path("Рецептуры терразит.xlsx")
            ]
            for path in possible_paths:
                if path.exists():
                    excel_path = path
                    break
        
        if excel_path.exists():
            recipe_loader = RecipeLoader(str(excel_path))
            recipe_loader.load_excel()
        else:
            logger.warning("Excel файл не найден. Загрузчик рецептов не инициализирован.")
    return recipe_loader


def get_component_analyzer():
    """Зависимость для получения анализатора компонентов"""
    global component_analyzer
    if component_analyzer is None:
        logger.info("Инициализация анализатора компонентов...")
        excel_path = Path(config.project_root) / config.data.excel_file
        if excel_path.exists():
            component_analyzer = ComponentAnalyzer(str(excel_path))
            component_analyzer.load_excel()
            component_analyzer.analyze_components()
        else:
            logger.warning("Excel файл не найден. Анализатор компонентов не инициализирован.")
    return component_analyzer


@app.on_event("startup")
async def startup_event():
    """Запуск приложения: инициализация компонентов"""
    logger.info(f"Запуск Terrazite AI API v{config.version}")
    logger.info(f"Режим: {config.mode}")
    
    # Создаем необходимые директории
    (Path(config.project_root) / "uploads").mkdir(exist_ok=True)
    (Path(config.project_root) / "exports").mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Terrazite AI API",
        "version": config.version,
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "categories": "/api/categories",
            "components": "/api/components",
            "recipes": "/api/recipes",
            "predict": "/api/predict (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    model_status = "loaded" if model is not None else "not loaded"
    recipes_status = "loaded" if recipe_loader is not None else "not loaded"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": model_status,
        "recipes": recipes_status,
        "categories": config.data.recipe_categories
    }


@app.get("/api/categories")
async def get_categories():
    """Получение списка категорий рецептов"""
    return {
        "categories": config.data.recipe_categories,
        "count": len(config.data.recipe_categories)
    }


@app.get("/api/categories/{category_name}")
async def get_category_info(category_name: str):
    """Получение информации о категории"""
    if category_name not in config.data.recipe_categories:
        raise HTTPException(status_code=404, detail=f"Категория '{category_name}' не найдена")
    
    # Получаем загрузчик рецептов
    loader = get_recipe_loader()
    if loader is None:
        raise HTTPException(status_code=500, detail="Загрузчик рецептов не инициализирован")
    
    # Получаем рецепты категории
    category_recipes = loader.get_recipe_by_category(category_name)
    
    # Анализируем компоненты категории
    analyzer = get_component_analyzer()
    typical_components = []
    if analyzer is not None and category_name in analyzer.analysis_results.get('unique_components_by_category', {}):
        typical_components = analyzer.analysis_results['unique_components_by_category'][category_name][:10]
    
    return {
        "name": category_name,
        "description": config.get_category_info(category_name).get("description", ""),
        "recipe_count": len(category_recipes),
        "typical_components": typical_components,
        "recipes": [recipe.name for recipe in category_recipes[:10]]  # Только 10 первых
    }


@app.get("/api/components")
async def get_components(
    group: Optional[str] = Query(None, description="Фильтр по группе компонентов"),
    category: Optional[str] = Query(None, description="Фильтр по категории рецепта")
):
    """Получение списка компонентов с возможностью фильтрации"""
    analyzer = get_component_analyzer()
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Анализатор компонентов не инициализирован")
    
    # Получаем все компоненты
    all_components = set()
    if analyzer.df is not None:
        component_columns = [col for col in analyzer.df.columns 
                           if col not in ['recipe_name', 'category'] 
                           and not str(col).startswith('Unnamed')]
        all_components.update(component_columns)
    
    # Фильтрация по группе
    filtered_components = list(all_components)
    if group and group in config.data.component_groups:
        filtered_components = config.data.component_groups[group]
    
    # Фильтрация по категории
    if category and category in config.data.recipe_categories:
        loader = get_recipe_loader()
        if loader is not None:
            category_recipes = loader.get_recipe_by_category(category)
            category_components = set()
            for recipe in category_recipes:
                category_components.update(recipe.components.keys())
            filtered_components = [c for c in filtered_components if c in category_components]
    
    return {
        "components": filtered_components[:100],  # Ограничиваем 100 компонентами
        "total": len(filtered_components),
        "group": group,
        "category": category
    }


@app.get("/api/components/{component_name}")
async def get_component_info(component_name: str):
    """Получение информации о компоненте"""
    # Определяем группу компонента
    group = config.get_component_group(component_name)
    
    # Ищем использование компонента в рецептах
    loader = get_recipe_loader()
    usage_count = 0
    average_value = 0
    categories_using = set()
    
    if loader is not None and loader.df is not None:
        if component_name in loader.df.columns:
            # Считаем использование
            usage_series = loader.df[component_name].apply(loader._parse_float)
            usage_count = (usage_series > 0).sum()
            if usage_count > 0:
                average_value = usage_series[usage_series > 0].mean()
            
            # Определяем категории, где используется компонент
            for category in config.data.recipe_categories:
                category_df = loader.df[loader.df['category'] == category]
                if component_name in category_df.columns:
                    category_usage = (category_df[component_name].apply(loader._parse_float) > 0).sum()
                    if category_usage > 0:
                        categories_using.add(category)
    
    return {
        "name": component_name,
        "group": group,
        "usage_count": usage_count,
        "average_value_kg": round(average_value, 2),
        "categories_using": list(categories_using),
        "description": f"Компонент группы '{group}'" if group else "Компонент без группы"
    }


@app.get("/api/recipes")
async def get_recipes(
    category: Optional[str] = Query(None, description="Фильтр по категории"),
    search: Optional[str] = Query(None, description="Поиск по названию"),
    limit: int = Query(50, ge=1, le=500, description="Лимит результатов"),
    offset: int = Query(0, ge=0, description="Смещение")
):
    """Получение списка рецептов с фильтрацией"""
    loader = get_recipe_loader()
    if loader is None:
        raise HTTPException(status_code=500, detail="Загрузчик рецептов не инициализирован")
    
    # Получаем все рецепты
    all_recipes = loader.get_all_recipes(include_components=False)
    
    # Фильтрация по категории
    if category and category in config.data.recipe_categories:
        filtered_recipes = [r for r in all_recipes if r.category == category]
    else:
        filtered_recipes = all_recipes
    
    # Поиск по названию
    if search:
        filtered_recipes = [r for r in filtered_recipes if search.lower() in r.name.lower()]
    
    # Пагинация
    total = len(filtered_recipes)
    paginated_recipes = filtered_recipes[offset:offset + limit]
    
    # Формируем ответ
    recipes_data = []
    for recipe in paginated_recipes:
        recipes_data.append({
            "name": recipe.name,
            "category": recipe.category,
            "component_count": len(recipe.components),
            "total_weight": sum(recipe.components.values())
        })
    
    return {
        "recipes": recipes_data,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }


@app.get("/api/recipes/{recipe_name}")
async def get_recipe_detail(recipe_name: str):
    """Получение детальной информации о рецепте"""
    loader = get_recipe_loader()
    if loader is None:
        raise HTTPException(status_code=500, detail="Загрузчик рецептов не инициализирован")
    
    # Ищем рецепт по имени
    found_recipe = None
    for recipe in loader.get_all_recipes():
        if recipe.name == recipe_name:
            found_recipe = recipe
            break
    
    if found_recipe is None:
        raise HTTPException(status_code=404, detail=f"Рецепт '{recipe_name}' не найден")
    
    # Группируем компоненты по группам
    components_by_group = {}
    for component, value in found_recipe.components.items():
        group = config.get_component_group(component) or "other"
        if group not in components_by_group:
            components_by_group[group] = []
        components_by_group[group].append({
            "name": component,
            "value_kg": value,
            "percentage": (value / found_recipe.total_weight * 100) if found_recipe.total_weight > 0 else 0
        })
    
    # Сортируем компоненты по значению
    for group in components_by_group:
        components_by_group[group].sort(key=lambda x: x["value_kg"], reverse=True)
    
    return {
        "name": found_recipe.name,
        "category": found_recipe.category,
        "total_weight": found_recipe.total_weight,
        "component_count": len(found_recipe.components),
        "components": found_recipe.components,
        "components_by_group": components_by_group,
        "metadata": found_recipe.metadata
    }


@app.post("/api/predict")
async def predict_from_image(
    file: UploadFile = File(..., description="Изображение терразитовой штукатурки"),
    include_similar: bool = Query(True, description="Включить похожие рецепты"),
    similarity_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Порог сходства")
):
    """Предсказание рецепта по изображению"""
    # Проверяем тип файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Загружаем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Здесь должен быть код для предсказания с помощью модели
        # Временно возвращаем заглушку
        
        # Генерируем предсказания на основе имени файла
        import hashlib
        file_hash = hashlib.md5(contents).hexdigest()
        seed = int(file_hash, 16) % 10000
        
        # Случайная категория
        np.random.seed(seed)
        category_idx = np.random.randint(0, len(config.data.recipe_categories))
        predicted_category = config.data.recipe_categories[category_idx]
        
        # Случайные компоненты
        analyzer = get_component_analyzer()
        if analyzer is not None:
            component_features = analyzer.get_component_features()
            component_names = list(component_features['component_to_idx'].keys())
        else:
            component_names = [
                "Цемент белый ПЦ500",
                "Цемент серый ПЦ500, кг",
                "Песок лужский фр.0-0,63мм, кг",
                "Доломитовая мука, кг"
            ]
        
        num_components = min(8, len(component_names))
        selected_indices = np.random.choice(len(component_names), num_components, replace=False)
        
        predicted_components = {}
        total_weight = 0
        for idx in selected_indices:
            component_name = component_names[idx]
            value = np.random.uniform(50, 300)
            predicted_components[component_name] = round(value, 2)
            total_weight += value
        
        # Ищем похожие рецепты
        similar_recipes = []
        if include_similar:
            loader = get_recipe_loader()
            if loader is not None:
                # Векторизуем предсказанные компоненты
                target_vector = loader.vectorize_components(predicted_components)
                
                # Ищем похожие среди рецептов той же категории
                category_recipes = loader.get_recipe_by_category(predicted_category)
                
                for recipe in category_recipes[:20]:  # Ограничиваем поиск 20 рецептами
                    recipe_vector = loader.vectorize_components(recipe.components)
                    
                    # Косинусное сходство
                    dot_product = np.dot(target_vector, recipe_vector)
                    norm_target = np.linalg.norm(target_vector)
                    norm_recipe = np.linalg.norm(recipe_vector)
                    
                    if norm_target > 0 and norm_recipe > 0:
                        similarity = dot_product / (norm_target * norm_recipe)
                        
                        if similarity >= similarity_threshold:
                            similar_recipes.append({
                                "name": recipe.name,
                                "similarity": float(similarity),
                                "category": recipe.category,
                                "component_count": len(recipe.components)
                            })
                
                # Сортируем по сходству
                similar_recipes.sort(key=lambda x: x["similarity"], reverse=True)
                similar_recipes = similar_recipes[:5]
        
        # Сохраняем изображение
        upload_dir = Path(config.project_root) / "uploads"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"prediction_{timestamp}_{file.filename}"
        image_path = upload_dir / image_filename
        image.save(image_path)
        
        return {
            "prediction_id": file_hash[:8],
            "timestamp": datetime.now().isoformat(),
            "image_filename": image_filename,
            "predicted_category": predicted_category,
            "confidence": float(np.random.uniform(0.7, 0.95)),
            "total_weight": round(total_weight, 2),
            "predicted_components": predicted_components,
            "similar_recipes": similar_recipes
        }
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")


@app.post("/api/find-similar")
async def find_similar_recipes(
    components: Dict[str, float],
    top_k: int = Query(5, ge=1, le=20, description="Количество возвращаемых рецептов")
):
    """Поиск похожих рецептов по компонентам"""
    loader = get_recipe_loader()
    if loader is None:
        raise HTTPException(status_code=500, detail="Загрузчик рецептов не инициализирован")
    
    try:
        similar = loader.find_similar_recipes(components, top_k=top_k)
        
        result = []
        for recipe, similarity in similar:
            result.append({
                "name": recipe.name,
                "category": recipe.category,
                "similarity": float(similarity),
                "component_count": len(recipe.components),
                "total_weight": sum(recipe.components.values())
            })
        
        return {
            "query_components": components,
            "similar_recipes": result,
            "total_found": len(result)
        }
        
    except Exception as e:
        logger.error(f"Ошибка при поиске похожих рецептов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


@app.get("/api/analysis/component-stats")
async def get_component_statistics():
    """Получение статистики по компонентам"""
    analyzer = get_component_analyzer()
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Анализатор компонентов не инициализирован")
    
    if not analyzer.analysis_results:
        analyzer.analyze_components()
    
    # Подготавливаем данные для ответа
    stats = {
        "categories": analyzer.analysis_results.get("category_stats", {}),
        "component_frequency": {},
        "unique_components_by_category": analyzer.analysis_results.get("unique_components_by_category", {}),
        "component_groups_by_category": analyzer.analysis_results.get("component_groups_by_category", {})
    }
    
    # Преобразуем defaultdict в dict
    for category, components in analyzer.analysis_results.get("component_frequency", {}).items():
        stats["component_frequency"][category] = dict(components)
    
    return stats


@app.get("/api/analysis/category-comparison")
async def compare_categories(
    category1: str = Query(..., description="Первая категория"),
    category2: str = Query(..., description="Вторая категория")
):
    """Сравнение двух категорий рецептов"""
    if category1 not in config.data.recipe_categories:
        raise HTTPException(status_code=404, detail=f"Категория '{category1}' не найдена")
    
    if category2 not in config.data.recipe_categories:
        raise HTTPException(status_code=404, detail=f"Категория '{category2}' не найдена")
    
    loader = get_recipe_loader()
    analyzer = get_component_analyzer()
    
    if loader is None or analyzer is None:
        raise HTTPException(status_code=500, detail="Загрузчик или анализатор не инициализирован")
    
    # Получаем рецепты категорий
    recipes1 = loader.get_recipe_by_category(category1)
    recipes2 = loader.get_recipe_by_category(category2)
    
    # Анализируем компоненты категорий
    if not analyzer.analysis_results:
        analyzer.analyze_components()
    
    # Находим общие и уникальные компоненты
    components1 = set()
    for recipe in recipes1:
        components1.update(recipe.components.keys())
    
    components2 = set()
    for recipe in recipes2:
        components2.update(recipe.components.keys())
    
    common_components = components1.intersection(components2)
    unique_to_1 = components1 - components2
    unique_to_2 = components2 - components1
    
    # Среднее количество компонентов
    avg_components1 = np.mean([len(r.components) for r in recipes1]) if recipes1 else 0
    avg_components2 = np.mean([len(r.components) for r in recipes2]) if recipes2 else 0
    
    # Средний общий вес
    avg_weight1 = np.mean([sum(r.components.values()) for r in recipes1]) if recipes1 else 0
    avg_weight2 = np.mean([sum(r.components.values()) for r in recipes2]) if recipes2 else 0
    
    return {
        "categories": [category1, category2],
        "recipe_counts": [len(recipes1), len(recipes2)],
        "avg_components": [float(avg_components1), float(avg_components2)],
        "avg_total_weight": [float(avg_weight1), float(avg_weight2)],
        "common_components": list(common_components)[:20],  # Ограничиваем 20 компонентами
        "unique_to_first": list(unique_to_1)[:20],
        "unique_to_second": list(unique_to_2)[:20],
        "comparison": {
            "component_overlap": len(common_components) / (len(components1.union(components2)) or 1),
            "recipes_difference": abs(len(recipes1) - len(recipes2)) / max(len(recipes1), len(recipes2), 1)
        }
    }


@app.get("/api/export/recipes")
async def export_recipes(
    format: str = Query("json", regex="^(json|csv)$", description="Формат экспорта"),
    category: Optional[str] = Query(None, description="Фильтр по категории")
):
    """Экспорт рецептов в различных форматах"""
    loader = get_recipe_loader()
    if loader is None:
        raise HTTPException(status_code=500, detail="Загрузчик рецептов не инициализирован")
    
    # Получаем рецепты
    if category and category in config.data.recipe_categories:
        recipes = loader.get_recipe_by_category(category)
    else:
        recipes = loader.get_all_recipes()
    
    # Подготавливаем данные
    export_data = []
    for recipe in recipes:
        recipe_data = {
            "name": recipe.name,
            "category": recipe.category,
            "component_count": len(recipe.components),
            "total_weight": sum(recipe.components.values()),
            "components": recipe.components
        }
        export_data.append(recipe_data)
    
    # Экспорт в JSON
    if format == "json":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"terrazite_recipes_{timestamp}.json"
        filepath = Path(config.project_root) / "exports" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/json"
        )
    
    # Экспорт в CSV
    elif format == "csv":
        import pandas as pd
        
        # Создаем плоскую структуру для CSV
        csv_data = []
        for recipe in export_data:
            # Основная информация
            row = {
                "name": recipe["name"],
                "category": recipe["category"],
                "component_count": recipe["component_count"],
                "total_weight": recipe["total_weight"]
            }
            
            # Добавляем топ-10 компонентов
            components_sorted = sorted(recipe["components"].items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (component, value) in enumerate(components_sorted, 1):
                row[f"component_{i}_name"] = component
                row[f"component_{i}_value"] = value
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"terrazite_recipes_{timestamp}.csv"
        filepath = Path(config.project_root) / "exports" / filename
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="text/csv"
        )


@app.get("/api/status")
async def api_status():
    """Полный статус API"""
    model_loaded = model is not None
    recipes_loaded = recipe_loader is not None
    analyzer_loaded = component_analyzer is not None
    
    # Информация о данных
    recipe_count = 0
    if recipes_loaded:
        recipe_count = len(recipe_loader.get_all_recipes())
    
    component_count = 0
    if analyzer_loaded and hasattr(analyzer_loader, 'component_features'):
        component_count = analyzer_loader.component_features.get('total_components', 0)
    
    return {
        "api": {
            "version": config.version,
            "status": "running",
            "uptime": "N/A"  # Здесь можно добавить расчет времени работы
        },
        "components": {
            "model": {
                "loaded": model_loaded,
                "type": config.model.model_name,
                "categories": config.model.num_categories,
                "components": config.model.num_components
            },
            "recipe_loader": {
                "loaded": recipes_loaded,
                "recipe_count": recipe_count,
                "categories_loaded": config.data.recipe_categories
            },
            "component_analyzer": {
                "loaded": analyzer_loaded,
                "component_count": component_count
            }
        },
        "resources": {
            "endpoints": len(app.routes),
            "memory_usage": "N/A"  # Здесь можно добавить информацию об использовании памяти
        }
    }


# Запуск приложения
if __name__ == "__main__":
    # Загружаем конфигурацию
    setup_config()
    
    # Запускаем сервер
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        workers=config.api.workers
    )
