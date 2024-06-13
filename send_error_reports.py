import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from time import sleep

from funcs import send_report, setup_logger
import logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Подключение к MongoDB
client = MongoClient(os.getenv('MONGODB_LOCAL'))
db = client[os.getenv('DB_NAME')]

logger = setup_logger('send_report', 'logs/send_report.log')

def retry_failed_requests():
    while True:

        # Получите все коллекции, начинающиеся с 'send_report'
        collections = [name for name in db.list_collection_names() if name.startswith('send_report')]

           # Пройдите по каждой коллекции
        for collection_name in collections:
            collection = db[collection_name]

            # Получите все документы в коллекции
            failed_requests = list(collection.find())

            # Повторите каждый запрос
            for request in failed_requests:
                camera_id = request['camera_id']
                image_id = request['image_id']
                score = request['score']
                time = request['time']
                file_path = request['file_path']

                # Попробуйте отправить запрос снова
                try:
                    send_report(camera_id, image_id, file_path, datetime.strptime(time, '%Y-%m-%d %H:%M:%S'), score,
                                logger)

                    # Если запрос успешно выполнен, удалите документ из базы данных
                    collection.delete_one({'_id': request['_id']})

                except Exception as e:
                    logger.info(f"Failed to retry request for camera_id: {camera_id}, child_id: {child_id}. Error: {e}")

        # Ждите один час перед повторением
        sleep(3600)

if __name__ == '__main__':
    try:
        # Начните повторять неудачные запросы
        retry_failed_requests()
    except Exception as e:
        logger.error(f'Завершился ошибкой {e}')
        sleep(3600)

