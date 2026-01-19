.PHONY: install install-dev install-ml test lint run-api run-ui clean

# Установка минимальных зависимостей
install:
	pip install -r requirements-minimal.txt

# Установка для разработки
install-dev:
	pip install -r requirements-dev.txt

# Установка ML зависимостей
install-ml:
	pip install -r requirements-ml.txt

# Запуск всех тестов
test:
	pytest tests/ -v --cov=src --cov-report=html

# Проверка стиля кода
lint:
	black src/ scripts/ --check
	flake8 src/ scripts/
	mypy src/

# Форматирование кода
format:
	black src/ scripts/
	isort src/ scripts/

# Запуск API сервера
run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Запуск Streamlit интерфейса
run-ui:
	streamlit run streamlit_app.py

# Запуск всей системы (два процесса)
run-all:
	@echo "Запуск API и интерфейса..."
	@make -j 2 run-api run-ui

# Обработка Excel файла
process-data:
	python scripts/process_excel.py

# Очистка кэша и временных файлов
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Создание виртуального окружения
venv:
	python -m venv venv
	@echo "Активируйте виртуальное окружение:"
	@echo "  На Windows: venv\Scripts\activate"
	@echo "  На Mac/Linux: source venv/bin/activate"

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  make install     - Установка минимальных зависимостей"
	@echo "  make install-dev - Установка для разработки"
	@echo "  make install-ml  - Установка ML зависимостей"
	@echo "  make test        - Запуск тестов"
	@echo "  make lint        - Проверка стиля кода"
	@echo "  make format      - Форматирование кода"
	@echo "  make run-api     - Запуск API сервера"
	@echo "  make run-ui      - Запуск Streamlit интерфейса"
	@echo "  make run-all     - Запуск всей системы"
	@echo "  make process-data - Обработка Excel файла"
	@echo "  make clean       - Очистка временных файлов"
	@echo "  make venv        - Создание виртуального окружения"
