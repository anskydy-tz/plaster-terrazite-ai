#!/bin/bash
# Скрипт установки проекта Terrazite AI

set -e  # Выход при ошибке

echo "🚀 Установка Terrazite AI"
echo "=================================="
echo "Система подбора рецептов терразитовой штукатурки"
echo "Версия: 1.2.0"
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.9 или выше."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION обнаружен"

# Проверка версии Python
REQUIRED_VERSION="3.9.0"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "✅ Версия Python соответствует требованиям (≥ $REQUIRED_VERSION)"
else
    echo "⚠️  Версия Python $PYTHON_VERSION. Рекомендуется ≥ $REQUIRED_VERSION"
fi

# Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo "📁 Создание виртуального окружения..."
    python3 -m venv venv
    echo "✅ Виртуальное окружение создано"
else
    echo "📁 Виртуальное окружение уже существует"
fi

# Активация виртуального окружения
echo "🔧 Активация виртуального окружения..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Не удалось активировать виртуальное окружение"
    exit 1
fi

# Обновление pip
echo "📦 Обновление pip..."
pip install --upgrade pip wheel setuptools

# Установка зависимостей в зависимости от параметров
echo ""
echo "📦 Установка зависимостей..."

case "$1" in
    "--minimal")
        echo "📊 Установка минимальных зависимостей..."
        if [ -f "requirements-minimal.txt" ]; then
            pip install -r requirements-minimal.txt
            echo "✅ Минимальные зависимости установлены"
        else
            echo "⚠️  Файл requirements-minimal.txt не найден, используем requirements.txt"
            pip install -r requirements.txt
        fi
        ;;
    "--ml")
        echo "🧠 Установка ML зависимостей..."
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        fi
        if [ -f "requirements-ml.txt" ]; then
            pip install -r requirements-ml.txt
            echo "✅ ML зависимости установлены"
        fi
        ;;
    "--dev"|"")
        echo "🔧 Установка зависимостей для разработки..."
        if [ -f "requirements-dev.txt" ]; then
            pip install -r requirements-dev.txt
            echo "✅ Зависимости для разработки установлены"
        else
            # Поэтапная установка
            if [ -f "requirements.txt" ]; then
                pip install -r requirements.txt
            fi
            if [ -f "requirements-ml.txt" ]; then
                pip install -r requirements-ml.txt
            fi
            # Дополнительные dev зависимости
            pip install streamlit==1.28.0 plotly==5.17.0 jupyter notebook
        fi
        ;;
    *)
        echo "❌ Неизвестный параметр: $1"
        echo "Использование: $0 [--minimal|--ml|--dev]"
        exit 1
        ;;
esac

# Создание необходимых папок
echo ""
echo "📁 Создание структуры папок..."
mkdir -p data/raw data/processed data/raw/images
mkdir -p uploads exports checkpoints logs reports/visualizations
mkdir -p notebooks tests docs

echo "✅ Структура папок создана"

# Копирование примера .env файла
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📄 Создание .env файла из примера..."
    cp .env.example .env
    echo "⚠️  Отредактируйте файл .env под ваши настройки"
elif [ ! -f ".env" ]; then
    echo "📄 Создание базового .env файла..."
    cat > .env << EOF
# Конфигурация Terrazite AI
PROJECT_NAME=Terrazite AI
MODE=development
DEBUG=True

# Пути к данным
EXCEL_FILE=data/raw/Рецептуры терразит.xlsx
IMAGES_DIR=data/raw/images

# Настройки API
API_HOST=0.0.0.0
API_PORT=8000

# Настройки ML модели
MODEL_BACKBONE=resnet50
LEARNING_RATE=0.001
BATCH_SIZE=32
EOF
    echo "✅ Базовый .env файл создан"
fi

# Проверка наличия Excel файла
echo ""
echo "🔍 Проверка наличия данных..."
EXCEL_FILES=$(find data/raw -name "*.xlsx" -o -name "*.xls" 2>/dev/null | head -5)

if [ -n "$EXCEL_FILES" ]; then
    echo "✅ Найдены Excel файлы:"
    echo "$EXCEL_FILES" | while read file; do
        echo "   - $(basename "$file")"
    done
else
    echo "⚠️  Excel файлы с рецептами не найдены в data/raw/"
    echo "   Поместите файлы:"
    echo "   - Рецептуры терразит.xlsx"
    echo "   - Или другие файлы с рецептами"
fi

# Проверка установки
echo ""
echo "🔍 Проверка установки..."
if python -c "import fastapi, pandas, numpy, streamlit" &>/dev/null; then
    echo "✅ Основные зависимости работают"
else
    echo "⚠️  Возникли проблемы с некоторыми зависимостями"
fi

echo ""
echo "🎉 Установка завершена!"
echo "=================================="
echo ""
echo "📋 Команды для запуска:"

# Проверка наличия Makefile
if [ -f "Makefile" ]; then
    echo "  make help                                   # Показать все команды"
    echo "  make install-dev                           # Установить зависимости для разработки"
    echo "  make run-api                               # Запустить API сервер"
    echo "  make run-ui                                # Запустить веб-интерфейс"
    echo "  make run-all                               # Запустить всю систему"
    echo "  make process-data                          # Обработать Excel файл"
    echo "  make test                                  # Запустить тесты"
else
    echo "  source venv/bin/activate                    # Активировать окружение"
    echo "  uvicorn src.api.main:app --reload          # Запустить API"
    echo "  streamlit run streamlit_app.py             # Запустить интерфейс"
    echo "  python scripts/process_excel.py            # Обработать данные"
fi

echo ""
echo "🌐 Веб-интерфейсы после запуска:"
echo "  📚 API документация: http://localhost:8000/docs"
echo "  🎨 Streamlit интерфейс: http://localhost:8501"
echo ""
echo "📁 Структура проекта:"
echo "  📊 Данные: data/raw/ (поместите Excel файлы сюда)"
echo "  🧠 Модели: src/models/"
echo "  📝 Исходный код: src/"
echo "  🧪 Скрипты: scripts/"
echo ""
echo "🚀 Быстрый старт:"
echo "  1. Поместите Excel файл с рецептами в data/raw/"
echo "  2. Запустите: python scripts/process_excel.py"
echo "  3. Запустите: make run-all"
echo ""
echo "📄 Документация:"
echo "  Полный отчет: PROJECT_COMPLETION_REPORT.md"
echo "  Конфигурация: src/utils/config.py"
echo ""
echo "⚠️  Не забудьте активировать виртуальное окружение:"
echo "    source venv/bin/activate"
