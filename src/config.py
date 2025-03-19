import os
import torch

# Базовый каталог проекта (на уровень выше, чем src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Пути к данным
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'Test')  # замените на нужную папку (Test/Train)
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations', 'instances_Test.json')

# Каталоги для чекпоинтов и логов
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Гиперпараметры обучения
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

# Конфигурация модели
NUM_CLASSES = 5

# Настройка устройства (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
