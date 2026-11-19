#!/usr/bin/env python3
"""
Скрипт для индексации проекта и сохранения результатов в постоянное хранилище
"""

import os
import sys

# Путь к проекту
project_path = "/mnt/sdb1/projects/dl3"
index_cache_path = os.path.join(project_path, ".index_cache")

def setup_index_storage():
    """Настройка постоянного хранилища индекса"""
    if not os.path.exists(index_cache_path):
        os.makedirs(index_cache_path)
        print(f"Создан каталог для хранения индекса: {index_cache_path}")
    else:
        print(f"Каталог хранения индекса уже существует: {index_cache_path}")

def index_project():
    """Индексация проекта"""
    print("Индексация проекта...")
    # Здесь будет логика индексации файлов
    # Для реальной реализации нужно использовать соответствующие инструменты
    
    # Пример: перечисление Python файлов
    python_files = []
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, project_path)
                python_files.append(relative_path)
    
    print(f"Найдено Python файлов: {len(python_files)}")
    
    # Сохраняем список файлов в кэш
    index_file = os.path.join(index_cache_path, "project_files.txt")
    with open(index_file, 'w') as f:
        for file in python_files:
            f.write(f"{file}\n")
    
    print(f"Индекс сохранен в: {index_file}")

if __name__ == "__main__":
    setup_index_storage()
    index_project()
    print("Индексация проекта завершена!")