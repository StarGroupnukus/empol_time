import logging
import mimetypes
import os
from datetime import datetime
import numpy as np
import requests
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

def setup_logger(name, log_file, level=logging.DEBUG):
    """Настройка и создание логгера с заданными параметрами."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Файловый обработчик с фильтром
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Создаем и настраиваем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Уровень DEBUG для сбора всех сообщений
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

os.makedirs('logs', exist_ok=True)
logger = setup_logger("flog", "logs/flog.log")


def extract_date_from_filename(filename):
    """
    Извлекает дату из имени файла.

    Args:
    filename (str): Имя файла.

    Returns:
    str: Дата в формате YYYYMMDD.
    """
    try:
        date_str = filename.split("_")[
            2
        ]  # Предполагается, что дата находится в третьей части имени файла
        date_obj = datetime.strptime(date_str, "%Y%m%d%H%M%S%f")
        return date_obj
    except Exception as e:
        print(f"Произошла ошибка при извлечении даты из имени файла: {e}")
        return None



def send_report(camera_id, person_id, image_id, file_path, time, score, logger=logger):
    file_name = file_path.split("/")[-1]
    os.rename(file_path, f'{os.getenv("USERS_FOLDER_PATH")}/{person_id}/attendances/{file_name}')
    url = os.getenv("REPORT_URL")
    token = os.getenv("TOKEN_FOR_API")
    data = {
        "image_id": str(image_id),
        "device_id": str(camera_id),
        "images": file_name,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score": str(score),
    }

    try:
        with requests.post(
                url, data=data, headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}"
                },
                timeout=10
        ) as response:
            logger.info(response.status_code)
            logger.info(f"{image_id} -- {score}")
            if response.status_code != 200:
                document = {
                    "camera_id": str(camera_id),
                    "image_id": str(image_id),
                    "score": str(score),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "file_path": file_path,
                    "status_code": response.status_code,
                    "send_time": datetime.now(),
                }

                # Подключение к MongoDB
                client = MongoClient(os.getenv("MONGODB_LOCAL"))
                # Укажите имя базы данных и коллекции
                db = client[os.getenv("DB_NAME")]
                collection = db["send_report"]
                collection.insert_one(document)
            logger.warning(f"{image_id} -- {score}")
    except Exception as e:
        document = {
            "camera_id": str(camera_id),
            "image_id": str(image_id),
            "score": str(score),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path,
            "send_time": datetime.now(),
        }

        # Подключение к MongoDB
        client = MongoClient(os.getenv("MONGODB_LOCAL"))
        # Укажите имя базы данных и коллекции
        db = client[os.getenv("DB_NAME")]
        collection = db["send_report"]
        collection.insert_one(document)
        logger.error(e)


def get_faces_data(faces):
    if not faces:
        return None

    max_face = max(faces, key=lambda face: calculate_rectangle_area(face["bbox"]))
    return max_face


def calculate_rectangle_area(bbox):
    # Проверяем, что список bbox содержит четыре элемента
    if len(bbox) != 4:
        raise ValueError(
            "bbox должен содержать четыре координаты: x_min, y_min, x_max, y_max"
        )

    # Вычисляем ширину и высоту прямоугольника
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Вычисляем площадь прямоугольника
    area = width * height

    return area

def compute_sim(feat1, feat2, logger=logger):
    from numpy.linalg import norm

    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim
    except Exception as e:
        logger.error(e)
        return None


