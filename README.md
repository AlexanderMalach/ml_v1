```markdown
# ML_v1: Semantic Segmentation with DeepLabV3

Этот проект реализует задачу семантической сегментации изображений с использованием модели DeepLabV3-ResNet50. Проект включает кастомный датасет, построенный по формату COCO, и предоставляет скрипты для обучения и инференса модели. Проект оптимизирован для работы с GPU (CUDA).

## Репозиторий

# ML_v1: Semantic Segmentation with DeepLabV3

Этот проект реализует задачу семантической сегментации изображений с использованием модели DeepLabV3-ResNet50. Проект включает кастомный датасет, построенный по формату COCO, и предоставляет скрипты для обучения и инференса модели. Проект оптимизирован для работы с GPU (CUDA).

## Репозиторий

Репозиторий находится по адресу:  
[https://github.com/AlexanderMalach/ml_v1.git](https://github.com/AlexanderMalach/ml_v1.git)

## Структура проекта

```
ml_v1/
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


## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/AlexanderMalach/ml_v1.git
   cd ml_v1
   ```

2. **Создайте и активируйте виртуальное окружение (рекомендуется):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
   ```

3. **Установите зависимости:**

   Убедитесь, что у вас установлены драйверы NVIDIA и подходящая версия CUDA. Команды для установки зависимостей:
   
   ```bash
   pip install -r requirements.txt
   ```

   Если у вас другая версия CUDA, откройте файл `requirements.txt` и замените суффикс (например, `cu113`) на нужное значение (например, `cu116` для CUDA 11.6). Для подбора корректных версий воспользуйтесь [PyTorch Get Started](https://pytorch.org/get-started/locally/).

## Использование

### Обучение модели

Запустите скрипт обучения:

```bash
python src/training/train.py
```

Логи сохраняются в папке `logs/`, а чекпоинты модели – в папке `checkpoints/`.

### Инференс

Для применения модели на новых изображениях используйте скрипт инференса:

```bash
python src/inference/inf_2.py
```

Скрипт загрузит лучший чекпоинт из `checkpoints/` и сохранит предсказанную маску в файл `pred_mask.png`.

## Конфигурация

Основные параметры (пути, гиперпараметры, устройство) заданы в файле `src/config.py`. При необходимости вы можете изменить их для адаптации проекта под свои данные и условия обучения.

## Зависимости

Проект использует следующие библиотеки:
- **PyTorch** (с поддержкой CUDA)
- **torchvision**
- **albumentations**
- **opencv-python**
- **tensorboard**
- **pycocotools**
- **Pillow**
- **numpy**

Убедитесь, что у вас установлены подходящие драйверы NVIDIA и версия CUDA, соответствующая версии PyTorch, указанной в файле `requirements.txt`.

## Лицензия

[Укажите лицензию вашего проекта]
```

Этот README.md содержит всю необходимую информацию для установки, использования и понимания структуры вашего проекта. Вы можете дополнительно изменить или расширить его разделы в соответствии с потребностями.
