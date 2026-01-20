.PHONY: help install test train pipeline clean

# Цвета для вывода
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
NC=\033[0m # No Color

# Помощь
help:
	@echo "$(YELLOW)🚀 Terrazite AI - Команды управления$(NC)"
	@echo ""
	@echo "$(GREEN)📦 Установка:$(NC)"
	@echo "  make install           - Установить все зависимости"
	@echo "  make install-dev       - Установить зависимости для разработки"
	@echo "  make install-ml        - Установить ML зависимости"
	@echo ""
	@echo "$(GREEN)🧪 Тестирование:$(NC)"
	@echo "  make test              - Запустить все тесты"
	@echo "  test-unit             - Запустить unit-тесты"
	@echo "  test-integration      - Запустить интеграционные тесты"
	@echo "  test-coverage         - Запустить тесты с покрытием"
	@echo ""
	@echo "$(GREEN)🔄 Пайплайн данных:$(NC)"
	@echo "  make create-data       - Создать тестовые данные"
	@echo "  make process-data      - Обработать Excel файл"
	@echo "  make create-manifest   - Создать манифест данных"
	@echo "  make prepare-dataset   - Подготовить датасет изображений"
	@echo ""
	@echo "$(GREEN)🤖 Обучение:$(NC)"
	@echo "  make train             - Обучить модель (полный цикл)"
	@echo "  make train-quick       - Быстрое обучение (5 эпох)"
	@echo "  make evaluate          - Оценить модель"
	@echo ""
	@echo "$(GREEN)🚀 Пайплайн:$(NC)"
	@echo "  make pipeline          - Полный пайплайн (данные → обучение)"
	@echo "  make pipeline-quick    - Быстрый пайплайн (тестовый)"
	@echo ""
	@echo "$(GREEN)📊 Сервисы:$(NC)"
	@echo "  make run-api           - Запустить API сервер"
	@echo "  make run-ui            - Запустить веб-интерфейс"
	@echo "  make run-all           - Запустить все сервисы"
	@echo ""
	@echo "$(GREEN)🧹 Очистка:$(NC)"
	@echo "  make clean             - Очистить временные файлы"
	@echo "  make clean-all         - Очистить всё (включая данные)"
	@echo ""

# Установка
install:
	@echo "$(YELLOW)📦 Установка всех зависимостей...$(NC)"
	pip install -r requirements.txt
	pip install -r requirements-ml.txt

install-dev:
	@echo "$(YELLOW)🔧 Установка зависимостей для разработки...$(NC)"
	pip install -r requirements-dev.txt

install-ml:
	@echo "$(YELLOW)🧠 Установка ML зависимостей...$(NC)"
	pip install -r requirements-ml.txt

# Тестирование
test:
	@echo "$(YELLOW)🧪 Запуск всех тестов...$(NC)"
	python run_tests.py

test-unit:
	@echo "$(YELLOW)🧪 Запуск unit-тестов...$(NC)"
	python -m pytest tests/ -v -m "not integration"

test-integration:
	@echo "$(YELLOW)🧪 Запуск интеграционных тестов...$(NC)"
	python -m pytest tests/ -v -m "integration"

test-coverage:
	@echo "$(YELLOW)📊 Запуск тестов с покрытием...$(NC)"
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Пайплайн данных
create-data:
	@echo "$(YELLOW)📄 Создание тестовых данных...$(NC)"
	python create_test_excel.py

process-data:
	@echo "$(YELLOW)📊 Обработка Excel файла...$(NC)"
	python scripts/process_excel.py

create-manifest:
	@echo "$(YELLOW)📋 Создание манифеста данных...$(NC)"
	python create_data_manifest.py

prepare-dataset:
	@echo "$(YELLOW)🖼️ Подготовка датасета изображений...$(NC)"
	python scripts/prepare_image_dataset.py --create-mapping

# Обучение
train:
	@echo "$(YELLOW)🤖 Обучение модели (50 эпох)...$(NC)"
	python scripts/train_model.py --epochs 50 --batch-size 32 --plot

train-quick:
	@echo "$(YELLOW)🤖 Быстрое обучение модели (5 эпох)...$(NC)"
	python scripts/train_model.py --epochs 5 --batch-size 4 --plot

evaluate:
	@echo "$(YELLOW)📊 Оценка модели...$(NC)"
	python scripts/train_model.py --test-only

# Полный пайплайн
pipeline:
	@echo "$(YELLOW)🚀 Запуск полного пайплайна...$(NC)"
	python scripts/run_pipeline.py

pipeline-quick:
	@echo "$(YELLOW)🚀 Запуск быстрого пайплайна...$(NC)"
	python scripts/run_pipeline.py --quick

# Сервисы
run-api:
	@echo "$(YELLOW)🌐 Запуск API сервера...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	@echo "$(YELLOW)🎨 Запуск веб-интерфейса...$(NC)"
	streamlit run streamlit_app.py

run-all:
	@echo "$(YELLOW)🚀 Запуск всех сервисов...$(NC)"
	@echo "  API: http://localhost:8000"
	@echo "  UI: http://localhost:8501"
	@make -j 2 run-api run-ui

# Очистка
clean:
	@echo "$(YELLOW)🧹 Очистка временных файлов...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	@echo "$(YELLOW)🧹 Очистка всех данных...$(NC)"
	rm -rf data/processed/* 2>/dev/null || true
	rm -rf checkpoints/* 2>/dev/null || true
	rm -rf logs/* 2>/dev/null || true
	rm -rf reports/* 2>/dev/null || true
	rm -rf coverage_html/* 2>/dev/null || true
	rm -rf uploads/* 2>/dev/null || true
	rm -rf exports/* 2>/dev/null || true

# По умолчанию
.DEFAULT_GOAL := help
