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



def send_report(camera_id, id, file_paths, time, score, status, org_id, flogger=logger):
    url = os.getenv("REPORT_URL")
    data = {
        "camera_id": str(camera_id),
        "child_id": str(id),
        "score": str(score),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "1" if status else "0",
    }
    files_data = []

    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                mime_type, _ = mimetypes.guess_type(file_name)
                files_data.append(
                    ("images[]", (file_name, open(file_path, "rb"), mime_type))
                )
            else:
                flogger.error(f"Файл {file_path} не существует.")

        files = tuple(files_data)
        with requests.post(
            url, data=data, files=files, headers={"Accept": "application/json"}, timeout=10
        ) as response:
            flogger.info(response.status_code)
            if response.status_code != 200:
                document = {
                    "camera_id": str(camera_id),
                    "child_id": str(id),
                    "score": str(score),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "1" if status else "0",
                    "file_path": file_paths[0],
                    "status_code": response.status_code,
                    "send_time": datetime.now(),
                }

                # Подключение к MongoDB
                client = MongoClient(os.getenv("MONGODB_LOCAL"))
                # Укажите имя базы данных и коллекции
                db = client["face_project"]
                collection = db[f"send_report{org_id}"]
                collection.insert_one(document)
            flogger.warning(f"{id} -- {score}")
    except Exception as e:
        document = {
            "camera_id": str(camera_id),
            "child_id": str(id),
            "score": str(score),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "1" if status else "0",
            "file_path": file_paths[0],
            "send_time": datetime.now(),
        }

        # Подключение к MongoDB
        client = MongoClient(os.getenv("MONGODB_LOCAL"))
        # Укажите имя базы данных и коллекции
        db = client["face_project"]
        collection = db[f"send_report{org_id}"]
        collection.insert_one(document)
        flogger.error(e)
    finally:
        for _, (filename, file, _) in files_data:
            file.close()


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

