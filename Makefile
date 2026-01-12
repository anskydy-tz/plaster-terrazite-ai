# Makefile для проекта Terrazite Plaster AI

.PHONY: help install install-dev install-ml venv test lint format clean run-api run-frontend build

# Цвета для вывода
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Показать справку
	@echo "$(YELLOW)Terrazite Plaster AI - система подбора рецептов терразитовой штукатурки$(NC)"
	@echo "Использование: make [цель]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить основные зависимости
	@echo "$(YELLOW)Установка основных зависимостей...$(NC)"
	pip install -r requirements.txt

install-dev: ## Установить зависимости для разработки
	@echo "$(YELLOW)Установка зависимостей для разработки...$(NC)"
	pip install -r requirements-dev.txt

install-ml: ## Установить зависимости для машинного обучения
	@echo "$(YELLOW)Установка ML зависимостей...$(NC)"
	pip install -r requirements-ml.txt

install-all: install install-dev install-ml ## Установить все зависимости
	@echo "$(GREEN)✓ Все зависимости установлены$(NC)"

venv: ## Создать виртуальное окружение
	@echo "$(YELLOW)Создание виртуального окружения...$(NC)"
	python -m venv venv
	@echo "$(GREEN)✓ Виртуальное окружение создано$(NC)"
	@echo "Активируйте его командой:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"

test: ## Запустить все тесты
	@echo "$(YELLOW)Запуск тестов...$(NC)"
	python run_tests.py

test-unit: ## Запустить unit-тесты
	@echo "$(YELLOW)Запуск unit-тестов...$(NC)"
	python -m pytest tests/ -v -m "not integration"

test-integration: ## Запустить интеграционные тесты
	@echo "$(YELLOW)Запуск интеграционных тестов...$(NC)"
	python -m pytest tests/ -v -m "integration"

test-coverage: ## Запустить тесты с покрытием кода
	@echo "$(YELLOW)Запуск тестов с покрытием...$(NC)"
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint: ## Проверить код с помощью линтеров
	@echo "$(YELLOW)Проверка кода...$(NC)"
	python -m black --check src/ tests/
	python -m flake8 src/ tests/
	python -m mypy src/

format: ## Форматировать код
	@echo "$(YELLOW)Форматирование кода...$(NC)"
	python -m black src/ tests/
	python -m isort src/ tests/

clean: ## Очистить проект от временных файлов
	@echo "$(YELLOW)Очистка проекта...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache logs/*.log 2>/dev/null || true
	@echo "$(GREEN)✓ Проект очищен$(NC)"

run-api: ## Запустить API сервер
	@echo "$(YELLOW)Запуск API сервера...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend: ## Запустить Streamlit интерфейс
	@echo "$(YELLOW)Запуск Streamlit интерфейса...$(NC)"
	streamlit run streamlit_app.py

run-jupyter: ## Запустить Jupyter notebook
	@echo "$(YELLOW)Запуск Jupyter notebook...$(NC)"
	jupyter notebook notebooks/

process-data: ## Обработать данные из Excel
	@echo "$(YELLOW)Обработка данных из Excel...$(NC)"
	python scripts/process_excel_data.py

train-model: ## Обучить модель
	@echo "$(YELLOW)Обучение модели...$(NC)"
	python scripts/train_model.py

docker-build: ## Собрать Docker образ
	@echo "$(YELLOW)Сборка Docker образа...$(NC)"
	docker build -t terrazite-ai:latest .

docker-run: ## Запустить Docker контейнер
	@echo "$(YELLOW)Запуск Docker контейнера...$(NC)"
	docker run -p 8000:8000 terrazite-ai:latest

update-deps: ## Обновить зависимости
	@echo "$(YELLOW)Обновление зависимостей...$(NC)"
	pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U

setup-precommit: ## Настроить pre-commit хуки
	@echo "$(YELLOW)Настройка pre-commit...$(NC)"
	pre-commit install
	pre-commit autoupdate

# Короткие алиасы
i: install
id: install-dev
im: install-ml
t: test
l: lint
f: format
c: clean
ra: run-api
rf: run-frontend
