.PHONY: create-manifest prepare-dataset train-model

# Создание манифеста данных
create-manifest:
	python scripts/create_data_manifest.py

# Подготовка датасета изображений
prepare-dataset:
	python scripts/prepare_image_dataset.py --create-mapping

# Обучение модели
train-model:
	python scripts/train_model.py --epochs 50 --batch-size 32 --plot

# Полный пайплайн
full-pipeline: create-manifest prepare-dataset train-model

# Быстрый тест
quick-test:
	python scripts/create_data_manifest.py --analyze-types
	python scripts/prepare_image_dataset.py --copy-only --no-augmentation
	python scripts/train_model.py --epochs 5 --batch-size 4 --plot
