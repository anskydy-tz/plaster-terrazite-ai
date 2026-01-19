"""
Установка проекта Terrazite AI
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="terrazite-ai",
    version="1.2.0",
    author="Terrazite AI Team",
    author_email="team@terrazite-ai.com",
    description="Система для подбора рецепта терразитовой штукатурки по изображению",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/terrazite-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "ml": ["-r requirements-ml.txt"],
        "dev": ["-r requirements-dev.txt"],
        "minimal": ["-r requirements-minimal.txt"],
    },
    entry_points={
        "console_scripts": [
            "terrazite-api=src.api.main:main",
            "terrazite-process=scripts.process_excel:main",
        ],
    },
    include_package_data=True,
)
