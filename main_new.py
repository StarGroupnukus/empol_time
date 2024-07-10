import os
import threading
import time
from datetime import datetime
import cv2
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from download_file import new_create_indexes, update_database
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger, send_report

load_dotenv()

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

    def initialize_counter(self, counter_id):
        if self.counters.find_one({'_id': counter_id}) is None:
            self.counters.insert_one({'_id': counter_id, 'seq': 0})

    def increment_counter(self, counter_id):
        return self.counters.find_one_and_update(
            {'_id': counter_id},
            {'$inc': {'seq': 1}},
            upsert=True,
            return_document=True
        )['seq']

class FaceProcessor:
    def __init__(self):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0)

    def get_faces(self, image):
        return self.app.get(image)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        return self.get_faces(image)

class IndexManager:
    def __init__(self, org_name):
        self.org_name = org_name
        self.client_index, self.client_indices = new_create_indexes(Database().clients, Config.logger)
        self.employee_index, self.employee_indices = update_database(org_name, FaceProcessor().app)

    def update_client_index(self, new_clients):
        embeddings = [np.array(client["embedding"]) for client in new_clients]
        client_ids = [client["person_id"] for client in new_clients]

        vectors = np.array(embeddings).astype('float32')
        faiss.normalize_L2(vectors)
        self.client_index.add(vectors)
        self.client_indices.extend(client_ids)

    def search_employee(self, embedding):
        query = np.array(embedding).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, ids = self.employee_index.search(query, 1)
        if len(scores) == 0 or len(ids) == 0 or len(ids[0]) == 0:
            return 0, 0
        person_id = int(self.employee_indices[ids[0][0]])
        return abs(round(scores[0][0] * 100, 3)), person_id

    def search_client(self, embedding):
        query = np.array(embedding).astype(np.float32).reshape(1, -1)
        scores, ids = self.client_index.search(query, 1)
        if len(scores) == 0 or len(ids) == 0 or len(ids[0]) == 0:
            return 0, 0
        person_id = int(self.client_indices[ids[0][0]])
        return abs(round(scores[0][0] * 100, 3)), person_id

class ImageHandler:
    @staticmethod
    def move_file(file_path, orig_image_path, destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
        os.rename(file_path, f'{destination_folder}/{os.path.basename(file_path)}')
        if orig_image_path:
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
        self.face_processor = FaceProcessor()
        self.index_manager = IndexManager(self.org_name)
        self.new_clients = []
        self.lock = threading.Lock()
        self.check_add_to_db = False

    def main_run(self):
        threads = []
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('test'):
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
        if self.new_clients:
            self.index_manager.update_client_index(self.new_clients)
            self.new_clients.clear()
        if self.check_add_to_db:
            update_database(self.org_name, app=self.face_processor.app)
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
            self.handle_regular_client(file_path, face_data, folder_path, date)

    def handle_recognized(self, file_path, orig_image_path, face_data, folder_path, person_id, date, camera_id):
        os.makedirs(f"{folder_path}/recognized", exist_ok=True)
        new_file_path = f'{folder_path}/recognized/{person_id}_{face_data.det_score}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
        os.rename(file_path, new_file_path)
        back_file_name = self.send_background(orig_image_path, face_data.embedding)
        if back_file_name:
            send_report(camera_id, person_id, back_file_name, date, face_data.det_score, Config.logger)
        else:
            os.remove(orig_image_path)

    def handle_regular_client(self, file_path, face_data, folder_path, date):
        score, person_id = self.index_manager.search_client(face_data.embedding)
        Config.logger.info(f"Client Score {score}, id {person_id}")
        if score == 0 and person_id == 0:
            ImageHandler.move_file(file_path, None, f"{folder_path}/error")
        elif score > Config.THRESHOLD_IS_DB:
            self.add_regular_client_to_db(face_data, score, person_id, file_path, date)
            ImageHandler.move_file(file_path, None, f"{folder_path}/regular_clients")
        else:
            person_id = self.add_new_client_to_db(face_data, file_path, date)
            if person_id:
                ImageHandler.move_file(file_path, None, f"{folder_path}/new_clients")
            else:
                ImageHandler.move_file(file_path, None, f"{folder_path}/no_good")

    def add_regular_client_to_db(self, face_data, score, person_id, file_path, date):
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
                    'image_path': os.path.basename(file_path),
                }
                self.db.clients.insert_one(client_data)
                Config.logger.info("Regular client checked and added to db.")
            else:
                Config.logger.info("One of the conditions failed for regular client.")
        except Exception as e:
            Config.logger.error(f'Exception adding regular client: {e}')

    def add_new_client_to_db(self, face_data, file_path, date):
        Config.logger.info("Attempting to add a new client.")
        try:
            if (face_data.det_score >= Config.DET_SCORE_THRESH and
                abs(face_data.pose[1]) < Config.POSE_THRESHOLD and abs(face_data.pose[0]) < Config.POSE_THRESHOLD):
                new_client_id = self.check_new_clients(face_data)
                if new_client_id == 0:
                    person_id = self.db.increment_counter('client_id')
                else:
                    person_id = new_client_id
                client_data = {
                    "type": "new_client",
                    "person_id": int(person_id),
                    "embedding": face_data.embedding.tolist(),
                    "gender": int(face_data.gender),
                    "age": int(face_data.age),
                    "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_path': os.path.basename(file_path),
                }
                with self.lock:
                    self.new_clients.append(client_data)
                Config.logger.info(f"New client added with ID: {person_id}")
                if len(self.new_clients) >= Config.INDEX_UPDATE_THRESHOLD:
                    threading.Thread(target=self.index_manager.update_client_index, args=(self.new_clients,)).start()
                    self.new_clients.clear()
                return person_id
        except Exception as e:
            Config.logger.error(f'Exception adding new client: {e}')

    def check_new_clients(self, face_data):
        new_embedding = np.array(face_data.embedding)
        for client_data in self.new_clients:
            existing_embedding = np.array(client_data['embedding'])
            similarity = compute_sim(new_embedding, existing_embedding)
            if similarity > Config.CHECK_NEW_CLIENT:
                Config.logger.info("Client with similar embedding already exists in new_clients.")
                return client_data['person_id']
        return 0

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