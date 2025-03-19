```markdown
# DeepLabV3 Segmentation Project

Этот проект реализует задачу семантической сегментации изображений с использованием модели DeepLabV3-ResNet50. В проекте используется кастомный датасет на основе формата COCO и реализованы как обучение модели, так и инференс (применение модели на новых изображениях). Проект оптимизирован для работы с GPU (CUDA).

## Структура проекта

```
project/
├── checkpoints/                # Сохранённые модели (чекпоинты)
├── data/                       # Директория с данными
│   ├── images/                # Изображения (например, Test, Train)
│   └── annotations/           # Аннотации в формате COCO (например, instances_Test.json)
├── logs/                       # Логи обучения и работы проекта
├── models/
│   └── deeplabv3.py            # Определение модели DeepLabV3-ResNet50
├── src/
│   ├── config.py               # Файл с настройками (пути, гиперпараметры, устройство)
│   ├── datasets/
│   │   └── custom_dataset.py   # Кастомный датасет для семантической сегментации (COCO)
│   ├── inference/
│   │   └── inf_2.py            # Скрипт инференса (применение модели на изображениях)
│   ├── training/
│   │   └── train.py            # Скрипт обучения модели
│   └── utils/
│       └── logger.py           # Модуль для логирования
├── requirements.txt            # Зависимости проекта
└── README.md                   # Это описание проекта
```

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://your-repository-url.git
   cd your-repository-folder
   ```

2. **Создайте и активируйте виртуальное окружение (рекомендуется):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
   ```

3. **Установите зависимости:**

   ```bash
   pip install -r requirements.txt
   ```

   Обратите внимание: данный requirements.txt рассчитан на использование CUDA. Убедитесь, что у вас установлены подходящие драйверы NVIDIA и версия CUDA, соответствующая используемой версии PyTorch.

## Использование

### Обучение модели

Запустите скрипт обучения:

```bash
python src/training/train.py
```

Логи работы будут сохраняться в папке `logs/`, а чекпоинты модели — в папке `checkpoints/`.

### Инференс

Для применения модели на новых изображениях используйте скрипт инференса:

```bash
python src/inference/inf_2.py
```

Скрипт загрузит лучший чекпоинт из `checkpoints/` и сохранит предсказанную маску в файл `pred_mask.png`.

## Конфигурация

Все основные параметры (пути, гиперпараметры, устройство) заданы в файле `src/config.py`. При необходимости вы можете изменить их для адаптации проекта под свои данные и условия обучения.

## Зависимости

Проект использует следующие библиотеки:

- PyTorch (с поддержкой CUDA)
- torchvision
- albumentations
- opencv-python
- tensorboard
- pycocotools
- Pillow

Убедитесь, что у вас установлены драйверы NVIDIA и подходящая версия CUDA.

## Лицензия

[Укажите лицензию вашего проекта]
```

---

### requirements.txt

```plaintext
# PyTorch с поддержкой CUDA (укажите нужную версию и cuda-версию, пример для CUDA 11.3)
torch>=1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
torchvision>=0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Для работы с изображениями и визуализации
Pillow>=8.0.0
opencv-python>=4.5.0

# Аугментации и предобработка
albumentations>=1.0.0
albumentations[imgaug]==1.2.1
albumentations>=1.2.1
albumentations[extra]>=1.2.1

# Для работы с форматом COCO
pycocotools>=2.0.0

# TensorBoard для мониторинга обучения
tensorboard>=2.9.0

# Дополнительные библиотеки (если необходимо)
numpy>=1.18.0
```

> **Примечание:**
> - Проверьте актуальные версии библиотек, соответствующие вашему оборудованию и CUDA. Если у вас другая версия CUDA, замените `cu113` на нужную (например, `cu116` для CUDA 11.6).
> - Иногда для установки `pycocotools` на Windows требуется [pycocotools-windows](https://pypi.org/project/pycocotools-windows/) или аналогичный пакет.

---

Эти файлы README.md и requirements.txt помогут правильно настроить и запустить ваш проект, используя CUDA и необходимые библиотеки для обучения и инференса модели сегментации.