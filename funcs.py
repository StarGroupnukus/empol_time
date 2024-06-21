import logging
import os
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from numpy.linalg import norm

load_dotenv()


def setup_logger(name, log_file, level=logging.DEBUG):
    """Настройка и создание логгера с заданными параметрами."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


os.makedirs('logs', exist_ok=True)
logger = setup_logger("flog", "logs/flog.log")


def extract_date_from_filename(filename):
    """Извлекает дату из имени файла."""
    try:
        date_str = filename.split("_")[2]
        return datetime.strptime(date_str, "%Y%m%d%H%M%S%f")
    except Exception as e:
        logger.error(f"Произошла ошибка при извлечении даты из имени файла: {e}")
        return None


def send_report(camera_id, person_id, file_path, time, score, logger=logger):
    file_name = os.path.basename(file_path)
    folder = os.path.join(os.getenv("USERS_FOLDER_PATH"), str(person_id), "attendances")
    os.makedirs(folder, exist_ok=True)
    os.rename(file_path, os.path.join(folder, file_name))

    url = os.getenv("REPORT_URL")
    token = os.getenv("TOKEN_FOR_API")
    data = {
        "user_id": str(person_id),
        "device_id": str(camera_id),
        "images[]": [file_name],
        "time": time.strftime("%H:%M:%S"),
        "score": str(score),
    }

    try:
        response = requests.post(url, data=data, headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"
        }, timeout=10)
        logger.info(f"{person_id} -- {score} sent {response.status_code}")
        if response.status_code != 201:
            logger.error(f"Error: {response.status_code} for {person_id}")
        else:
            logger.info(f"Report sent successfully for {person_id}")
    except Exception as e:
        logger.error(f"Exception while sending report: {e}")


def get_faces_data(faces):
    """Возвращает данные о лице с максимальной площадью прямоугольника."""
    if not faces:
        return None
    return max(faces, key=lambda face: calculate_rectangle_area(face["bbox"]))


def calculate_rectangle_area(bbox):
    """Вычисляет площадь прямоугольника."""
    if len(bbox) != 4:
        raise ValueError("bbox должен содержать четыре координаты: x_min, y_min, x_max, y_max")
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_sim(feat1, feat2, logger=logger):
    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim
    except Exception as e:
        logger.error(e)
        return None
