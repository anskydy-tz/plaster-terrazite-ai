# Makefile для проекта Terrazite AI
# Автоматизация основных задач: установка, подготовка данных, обучение, тестирование

.PHONY: help install data prepare train test clean all quick api streamlit docs

# Цвета для вывода
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Параметры по умолчанию
PYTHON := python
PIP := pip
VENV_DIR := venv
CHECKPOINTS_DIR := checkpoints
REPORTS_DIR := reports
DATA_DIR := data
LOGS_DIR := logs

help:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)  Terrazite AI - Makefile Commands$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(YELLOW)УСТАНОВКА И НАСТРОЙКА:$(NC)"
	@echo "  make help          - Показать эту справку"
	@echo "  make install       - Установить все зависимости"
	@echo "  make install-dev   - Установить зависимости для разработки"
	@echo "  make install-min   - Установить минимальные зависимости"
	@echo "  make venv          - Создать виртуальное окружение"
	@echo ""
	@echo "$(YELLOW)ПОДГОТОВКА ДАННЫХ:$(NC)"
	@echo "  make data          - Полный цикл подготовки данных"
	@echo "  make process-excel - Только обработка Excel"
	@echo "  make manifest      - Только создание манифестов"
	@echo "  make prepare       - Только подготовка изображений"
	@echo "  make test-images   - Создать тестовые изображения"
	@echo ""
	@echo "$(YELLOW)ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ:$(NC)"
	@echo "  make train         - Запустить обучение модели"
	@echo "  make train-quick   - Быстрое тестовое обучение"
	@echo "  make train-gpu     - Обучение на GPU"
	@echo "  make test          - Запустить все тесты"
	@echo "  make test-basic    - Базовые тесты модели"
	@echo "  make test-full     - Полное тестирование пайплайна"
	@echo ""
	@echo "$(YELLOW)ЗАПУСК СИСТЕМЫ:$(NC)"
	@echo "  make api           - Запустить API сервер"
	@echo "  make streamlit     - Запустить веб-интерфейс"
	@echo "  make run           - Запустить всё (API + Streamlit)"
	@echo ""
	@echo "$(YELLOW)ОЧИСТКА И ОБСЛУЖИВАНИЕ:$(NC)"
	@echo "  make clean         - Очистить временные файлы"
	@echo "  make clean-data    - Очистить обработанные данные"
	@echo "  make clean-all     - Полная очистка (кроме исходников)"
	@echo "  make docs          - Сгенерировать документацию"
	@echo "  make lint          - Проверить код линтером"
	@echo "  make format        - Отформатировать код"
	@echo ""
	@echo "$(YELLOW)ПРИМЕРЫ:$(NC)"
	@echo "  make all           - Выполнить полный цикл (подготовка + обучение)"
	@echo "  make quick         - Быстрый тестовый прогон всего пайплайна"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# УСТАНОВКА И НАСТРОЙКА
# ─────────────────────────────────────────────────────────────────────────────

venv:
	@echo "$(GREEN)🔧 Создание виртуального окружения...$(NC)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(GREEN)✅ Виртуальное окружение создано. Активируйте: source $(VENV_DIR)/bin/activate$(NC)"

install:
	@echo "$(GREEN)📦 Установка зависимостей...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo "$(GREEN)✅ Зависимости установлены$(NC)"

install-dev:
	@echo "$(GREEN)📦 Установка зависимостей для разработки...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✅ Зависимости для разработки установлены$(NC)"

install-min:
	@echo "$(GREEN)📦 Установка минимальных зависимостей...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[minimal]"
	@echo "$(GREEN)✅ Минимальные зависимости установлены$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────

process-excel:
	@echo "$(GREEN)📊 Обработка Excel файла...$(NC)"
	$(PYTHON) scripts/process_excel.py

manifest:
	@echo "$(GREEN)📋 Создание манифестов данных...$(NC)"
	$(PYTHON) scripts/create_data_manifest.py

prepare:
	@echo "$(GREEN)🖼️  Подготовка датасета изображений...$(NC)"
	$(PYTHON) scripts/prepare_image_dataset.py --create-mapping

test-images:
	@echo "$(GREEN)🎨 Создание тестовых изображений...$(NC)"
	$(PYTHON) scripts/create_test_images.py

data: process-excel manifest prepare
	@echo "$(GREEN)✅ Все данные подготовлены успешно!$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────────────

train:
	@echo "$(GREEN)🚀 Запуск обучения модели...$(NC)"
	$(PYTHON) scripts/train_model.py --plot

train-quick:
	@echo "$(GREEN)⚡ Быстрое тестовое обучение...$(NC)"
	$(PYTHON) scripts/train_model.py --quick-test --plot

train-gpu:
	@echo "$(GREEN)🚀 Запуск обучения на GPU...$(NC)"
	$(PYTHON) scripts/train_model.py --device cuda --plot

test-basic:
	@echo "$(GREEN)🔍 Запуск базовых тестов модели...$(NC)"
	$(PYTHON) test_model_basic.py

test-full:
	@echo "$(GREEN)🔍 Запуск полного тестирования пайплайна...$(NC)"
	$(PYTHON) test_full_pipeline.py

test: test-basic test-full
	@echo "$(GREEN)✅ Все тесты пройдены успешно!$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК СИСТЕМЫ
# ─────────────────────────────────────────────────────────────────────────────

api:
	@echo "$(GREEN)🌐 Запуск API сервера...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	@echo "$(GREEN)🎨 Запуск Streamlit интерфейса...$(NC)"
	streamlit run streamlit_app.py

run:
	@echo "$(GREEN)🚀 Запуск полной системы...$(NC)"
	@echo "$(YELLOW)API сервер будет доступен на http://localhost:8000$(NC)"
	@echo "$(YELLOW)Streamlit интерфейс будет доступен на http://localhost:8501$(NC)"
	@make -j2 api streamlit

# ─────────────────────────────────────────────────────────────────────────────
# ОЧИСТКА И ОБСЛУЖИВАНИЕ
# ─────────────────────────────────────────────────────────────────────────────

clean:
	@echo "$(YELLOW)🧹 Очистка временных файлов...$(NC)"
	rm -rf $(REPORTS_DIR)/*
	rm -rf $(LOGS_DIR)/*
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "$(GREEN)✅ Очистка завершена$(NC)"

clean-data:
	@echo "$(YELLOW)🧹 Очистка обработанных данных...$(NC)"
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/raw/test_images
	@echo "$(GREEN)✅ Данные очищены$(NC)"

clean-all: clean clean-data
	@echo "$(YELLOW)🧹 Полная очистка...$(NC)"
	rm -rf $(CHECKPOINTS_DIR)/*
	rm -rf $(VENV_DIR)
	@echo "$(GREEN)✅ Полная очистка завершена$(NC)"

docs:
	@echo "$(GREEN)📚 Генерация документации...$(NC)"
	# pdoc --html --output-dir docs src --force
	@echo "$(YELLOW)⚠️  Документация будет доступна в docs/$(NC)"

lint:
	@echo "$(GREEN)🔍 Проверка кода линтером...$(NC)"
	ruff check src/ scripts/
	@echo "$(GREEN)✅ Линтер завершил проверку$(NC)"

format:
	@echo "$(GREEN)✨ Форматирование кода...$(NC)"
	black src/ scripts/
	@echo "$(GREEN)✅ Форматирование завершено$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# КОМБИНИРОВАННЫЕ КОМАНДЫ
# ─────────────────────────────────────────────────────────────────────────────

quick: data train-quick test-basic
	@echo "$(GREEN)🎉 Быстрый прогон завершен успешно!$(NC)"

all: data train test
	@echo "$(GREEN)🎉 Полный цикл выполнен успешно!$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# ДОПОЛНИТЕЛЬНЫЕ КОМАНДЫ
# ─────────────────────────────────────────────────────────────────────────────

check-data:
	@echo "$(GREEN)🔍 Проверка наличия данных...$(NC)"
	@if [ -f "data/raw/recipes.xlsx" ]; then \
		echo "$(GREEN)  ✅ Excel файл найден$(NC)"; \
	else \
		echo "$(RED)  ❌ Excel файл не найден: data/raw/recipes.xlsx$(NC)"; \
	fi
	@if [ -d "data/raw/images" ]; then \
		IMG_COUNT=$$(find data/raw/images -name "*.jpg" -o -name "*.png" | wc -l); \
		echo "$(GREEN)  ✅ Изображения найдены: $$IMG_COUNT$(NC)"; \
	else \
		echo "$(RED)  ❌ Директория с изображениями не найдена: data/raw/images$(NC)"; \
	fi

status:
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)  Terrazite AI - Статус проекта$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@make check-data
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
