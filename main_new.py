import os
import threading
import time
from distutils.command.config import config

import cv2
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from pymongo import MongoClient

from download_file import new_create_indexes, update_database, create_indexes, update_database_to_db
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger, send_report, send_report_client

load_dotenv()


client_index = None
client_indices = None
employee_index = None
employee_indices = None


class Config:
    CHECK_NEW_CLIENT = 0.5
    THRESHOLD_IS_DB = 60
    POSE_THRESHOLD = 40
    DET_SCORE_THRESH = 0.65
    IMAGE_COUNT = 10
    THRESHOLD_ADD_DB = 65
    DIMENSIONS = 512
    INDEX_UPDATE_THRESHOLD = 5
    logger = setup_logger('MainRunner', 'logs/main.log')
    INIT_IMAGE_PATH = './pavel.png'


class Database:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.db = self.client.biz_count
        self.employees = self.db.employees
        self.clients = self.db.clients
        self.counters = self.db.counters
        self.initialize_counter('client_id')

    def init_clients_db(self):
        image = cv2.imread(Config.INIT_IMAGE_PATH)
        face_data = FaceProcessor().app.get(image)[0]
        client_data = {
            "person_id": 0,
            "embedding": face_data.embedding.tolist(),
        }
        self.clients.insert_one(client_data)

    def initialize_counter(self, counter_id):
        if self.counters.find_one({'_id': counter_id}) is None:
            self.counters.insert_one({'_id': counter_id, 'seq': 0})
            self.init_clients_db()

    def increment_counter(self, counter_id):
        return self.counters.find_one_and_update(
            {'_id': counter_id},
            {'$inc': {'seq': 1}},
            upsert=True,
            return_document=True
        )['seq']


import threading

class FaceProcessor:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if FaceProcessor._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.app = FaceAnalysis()
            self.app.prepare(ctx_id=0)
            FaceProcessor._instance = self

    @staticmethod
    def get_instance():
        if FaceProcessor._instance is None:
            with FaceProcessor._lock:
                if FaceProcessor._instance is None:
                    FaceProcessor()
        return FaceProcessor._instance

    def get_faces(self, image):
        return self.app.get(image)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        return self.get_faces(image)

class IndexManager:
    def __init__(self, org_name):
        global client_index, client_indices, employee_index, employee_indices
        self.org_name = org_name
        self.lock = threading.Lock()
        if client_index is None or client_indices is None:
            client_index, client_indices = create_indexes(Database().clients)
        if employee_index is None or employee_indices is None:
            employee_index, employee_indices = update_database_to_db(Database().employees, FaceProcessor.get_instance().app)

    def update_client_index(self, new_clients):
        global client_index, client_indices
        embeddings = [np.array(client["embedding"]) for client in new_clients]
        client_ids = [client["person_id"] for client in new_clients]

        vectors = np.array(embeddings).astype('float32')
        faiss.normalize_L2(vectors)

        with self.lock:
            client_index.add(vectors)
            client_indices.extend(client_ids)

    def search_employee(self, embedding):
        global employee_index, employee_indices
        query = np.array(embedding).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        with self.lock:
            scores, ids = employee_index.search(query, 1)

        if len(scores) == 0 or len(ids) == 0 or len(ids[0]) == 0:
            return 0, 0
        person_id = int(employee_indices[ids[0][0]])
        return abs(round(scores[0][0] * 100, 3)), person_id

    def search_client(self, embedding):
        global client_index, client_indices
        query = np.array(embedding).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        with self.lock:
            scores, ids = client_index.search(query, 1)

        if len(scores) == 0 or len(ids) == 0 or len(ids[0]) == 0:
            return 0, 0
        person_id = int(client_indices[ids[0][0]])
        return abs(round(scores[0][0] * 100, 3)), person_id


class ImageHandler:
    @staticmethod
    def move_file(file_path, orig_image_path, destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
        os.rename(file_path, f'{destination_folder}/{os.path.basename(file_path)}')
        if os.path.exists(orig_image_path):
            os.remove(orig_image_path)

    @staticmethod
    def clean_files(file_path, orig_image_path):
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(orig_image_path):
            os.remove(orig_image_path)


class MainRunner:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.org_name = os.path.basename(images_folder)
        self.cameras_path_directories = [dir for dir in os.listdir(self.images_folder)]
        self.db = Database()
        self.face_processor = FaceProcessor.get_instance()
        self.index_manager = IndexManager(self.org_name)
        self.lock = threading.Lock()
        self.check_add_to_db = False

    def main_run(self):
        threads = []
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('cam'):
                continue
            camera_directory = f"{self.images_folder}/{camera_directory}"
            camera_id = 1
            time.sleep(1)
            Config.logger.warning(f'Camera start --> {camera_directory}')
            thread = threading.Thread(target=self.classify_images, args=(camera_directory, camera_id))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if self.check_add_to_db:
            update_database_to_db(Database().employees, app=self.face_processor.app)
            self.check_add_to_db = False

    def classify_images(self, folder_path, camera_id):
        list_files = [file for file in os.listdir(folder_path) if file.endswith('SNAP.jpg')]
        for file in list_files:
            file_path = os.path.join(folder_path, file)
            orig_image_path = file_path.replace('SNAP', 'BACKGROUND')
            if not os.path.exists(orig_image_path) or os.path.getsize(file_path) == 0:
                ImageHandler.clean_files(file_path, orig_image_path)
                continue

            date = extract_date_from_filename(file)
            try:
                faces = self.face_processor.process_image(file_path)
                if len(faces) == 0:
                    Config.logger.error("No faces found in the image")
                    ImageHandler.clean_files(file_path, orig_image_path)
                    continue
            except Exception as e:
                Config.logger.error(f'ERROR for app get: {e}')
                continue

            face_data = get_faces_data(faces)
            self.process_faces(face_data, file_path, orig_image_path, folder_path, camera_id, date)

    def process_faces(self, face_data, file_path, orig_image_path, folder_path, camera_id, date):
        score, person_id = self.index_manager.search_employee(face_data.embedding)
        Config.logger.info(f"Employee Score {score}")
        if score == 0:
            ImageHandler.move_file(file_path, orig_image_path, f"{folder_path}/error")
            return

        if score > Config.THRESHOLD_IS_DB:
            self.handle_recognized(file_path, orig_image_path, face_data, folder_path, person_id, date, camera_id)
        else:
            self.handle_regular_client(file_path, orig_image_path, face_data, folder_path, date, camera_id)

    def handle_recognized(self, file_path, orig_image_path, face_data, folder_path, person_id, date, camera_id):
        os.makedirs(f"{folder_path}/recognized", exist_ok=True)
        new_file_path = f'{folder_path}/recognized/{person_id}_{face_data.det_score}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
        os.rename(file_path, new_file_path)
        back_file_name = self.send_background(orig_image_path, face_data.embedding)
        if back_file_name:
            send_report(camera_id, person_id, back_file_name, date, face_data.det_score, Config.logger)
        else:
            os.remove(orig_image_path)

    def handle_regular_client(self, file_path, orig_image_path, face_data, folder_path, date, camera_id):
        score, person_id = self.index_manager.search_client(face_data.embedding)
        Config.logger.info(f"Client Score {score}, id {person_id}")
        if score == 0 and person_id == 0:
            ImageHandler.move_file(file_path, orig_image_path, f"{folder_path}/error")
        elif score > Config.THRESHOLD_IS_DB:
            self.add_regular_client_to_db(face_data, score, person_id, file_path, date, camera_id)
            ImageHandler.move_file(file_path, orig_image_path, f"{folder_path}/regular_clients")
        else:
            person_id = self.add_new_client_to_db(face_data, file_path, date, camera_id)
            if person_id:
                ImageHandler.move_file(file_path, orig_image_path, f"{folder_path}/new_clients")
            else:
                ImageHandler.move_file(file_path, orig_image_path, f"{folder_path}/no_good")

    def add_regular_client_to_db(self, face_data, score, person_id, file_path, date, camera_id):
        try:
            if (face_data.det_score >= Config.DET_SCORE_THRESH and
                    abs(face_data.pose[1]) < Config.POSE_THRESHOLD and abs(face_data.pose[0]) < Config.POSE_THRESHOLD):
                client_data = {
                    "type": "regular_client",
                    'score': float(score),
                    "person_id": int(person_id),
                    "embedding": face_data.embedding.tolist(),
                    "gender": int(face_data.gender),
                    "age": int(face_data.age),
                    "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': file_path,
                }
                self.db.clients.insert_one(client_data)
                Config.logger.info("Regular client checked and added to db.")
                send_report_client(client_data, camera_id, Config.logger)
            else:
                Config.logger.info("One of the conditions failed for regular client.")
        except Exception as e:
            Config.logger.error(f'Exception adding regular client: {e}')

    def add_new_client_to_db(self, face_data, file_path, date, camera_id):
        Config.logger.info("Attempting to add a new client.")
        try:
            if (face_data.det_score >= Config.DET_SCORE_THRESH and
                    abs(face_data.pose[1]) < Config.POSE_THRESHOLD and abs(face_data.pose[0]) < Config.POSE_THRESHOLD):
                person_id = self.db.increment_counter('client_id')

                client_data = {
                    "type": "new_client",
                    "person_id": int(person_id),
                    "embedding": face_data.embedding.tolist(),
                    "score": float(face_data.det_score),
                    "gender": int(face_data.gender),
                    "age": int(face_data.age),
                    "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': file_path,
                }
                with self.lock:
                    self.index_manager.update_client_index([client_data])
                self.db.clients.insert_one(client_data)
                send_report_client(client_data, camera_id, Config.logger)
                Config.logger.info(f"New client added with ID: {person_id}")
                return person_id
        except Exception as e:
            Config.logger.error(f'Exception adding new client: {e}')


    def send_background(self, file_path, embedding):
        image = cv2.imread(file_path)
        image_data = self.face_processor.get_faces(image)
        for data in image_data:
            if compute_sim(data.embedding, embedding) > 0.8:
                x1, y1, x2, y2 = map(int, data.bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(file_path, image)
                return file_path
        return False

    def add_employer_to_db(self, img_path, person_id):
        try:
            image_name = os.path.basename(img_path)
            folder = f"{os.getenv('USERS_FOLDER_PATH')}/{person_id}/images"
            os.makedirs(folder, exist_ok=True)
            os.rename(img_path, f"{folder}/{image_name}")
            url = f'{os.getenv("ADD_IMAGE_TO_USER")}/{person_id}'
            token = os.getenv("TOKEN_FOR_API")
            data = {'image': image_name}
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            }
            response = requests.post(url, data=data, headers=headers, timeout=10)
            Config.logger.info(f'Status code add to db: {response.status_code}')
            self.check_add_to_db = True
        except Exception as e:
            Config.logger.error(f'Exception adding employee image: {e}')

if __name__ == '__main__':
    runner = MainRunner(os.getenv('IMAGES_FOLDER'))
    while True:
        try:
            runner.main_run()
        except Exception as e:
            Config.logger.error(f'Exception main_run {e}')
        time.sleep(5)
