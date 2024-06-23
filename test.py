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

from download_file import update_database, create_indexes
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger, send_report

load_dotenv()

TRESHOLD_IS_DB = 14
POSE_TRESHOLD = 30
DET_SCORE_TRESH = 0.75
IMAGE_COUNT = 10
TRESHOLD_ADD_DB = 19
DIMENSIONS = 512
INDEX_UPDATE_TRESHOLD = 1
# ADD_TO_CLIENT_DB_THRESHOLD =20
logger = setup_logger('MainRunner', 'logs/main.log')


class MainRunner:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.org_name = os.path.basename(images_folder)
        self.cameras_path_directories = [dir for dir in os.listdir(self.images_folder)]
        self.db = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.mongodb = self.db.biz_count
        self.employees_db = self.mongodb.employees
        self.clients_db = self.mongodb.clients
        self.counter_db = self.mongodb.counters
        self.logger = setup_logger('MainRunner', 'logs/main.log')
        self.employee_data = list(self.clients_db.find())
        self.app = self.setup_app()
        self.client_index, self.client_indices = self.initialize_client_index()
        # self.initialize_counter('client_id')
        self.check_add_to_db = False
        self.employee_index = faiss.read_index(f'index_file{self.org_name}.index')
        self.employee_indices = np.load(f'indices{self.org_name}.npy', allow_pickle=True)
        self.new_clients = {}

    def setup_app(self):
        app = FaceAnalysis()
        app.prepare(ctx_id=0)
        update_database(self.org_name, app=app)
        return app

    def initialize_client_index(self):
        client_data = list(self.clients_db.find())
        if not client_data:
            self.logger.warning("Client index is not created due to empty database.")
            return faiss.IndexFlatIP(DIMENSIONS), []
        else:
            return create_indexes(self.clients_db, self.org_name, 'client')

    # def initialize_counter(self, counter_id):
    #     if self.counter_db.find_one({'_id': counter_id}) is None:
    #         self.counter_db.insert_one({'_id': counter_id, 'seq': 0})
    #         self.logger.info(f"Initialized counter for {counter_id}")

    def main_run(self):
        threads = []
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('Camera'):
                continue
            camera_directory = f"{self.images_folder}/{camera_directory}"
            camera_id = camera_directory.split(' ')[1]
            time.sleep(1)
            self.logger.warning(f'Camera start --> {camera_directory}')
            thread = threading.Thread(target=self.classify_images, args=(camera_directory, camera_id))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if self.check_add_to_db:
            update_database(self.org_name, app=self.app)
            self.check_add_to_db = False

    def classify_images(self, folder_path, camera_id):
        list_files = [file for file in os.listdir(folder_path) if file.endswith('SNAP.jpg')]
        for file in list_files:
            file_path = os.path.join(folder_path, file)
            orig_image_path = file_path.replace('SNAP', 'BACKGROUND')
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                os.remove(orig_image_path)
                continue
            image = cv2.imread(file_path)
            date = extract_date_from_filename(file)
            faces = self.app.get(image)

            if not faces:
                os.makedirs(f"{folder_path}/not_face", exist_ok=True)
                os.rename(file_path, f'{folder_path}/not_face/{file}')
                os.remove(orig_image_path)
                continue
            else:
                face_data = get_faces_data(faces)
                if self.clients_db.count_documents({}) == 0:
                    self.logger.warning("The clients database is empty. Skipping image classification.")
                    self.add_new_client_to_db(face_data)
                else:
                    score, person_id = self.is_employee(face_data, file_path)
                    if score == 0:
                        os.makedirs(f"{folder_path}/error", exist_ok=True)
                        os.rename(file_path, f'{folder_path}/error/{file}')
                        os.remove(orig_image_path)
                        continue
                    if score > TRESHOLD_IS_DB:
                        back_file_name = self.send_background(orig_image_path, face_data.embedding)
                        if back_file_name:
                            send_report(camera_id, person_id, back_file_name, date, score, logger)
                        else:
                            os.remove(orig_image_path)
                        # continue
                    else:
                        score, person_id = self.is_regular_client(face_data, file_path)
                        if score == 0 and person_id == 0:
                            os.makedirs(f"{folder_path}/error", exist_ok=True)
                            os.rename(file_path, f'{folder_path}/error/{file}')
                            continue
                        elif score > TRESHOLD_IS_DB:
                            if os.path.isfile(f'{folder_path}/{file}'):
                                os.makedirs(f"{folder_path}/regular_clients", exist_ok=True)
                                os.rename(f'{folder_path}/{file}',
                                          f'{folder_path}/regular_clients/{person_id}_{score}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                                # self.send_client_data(camera_id, person_id, date, file_path, face_data)
                        else:
                            person_id = self.add_new_client_to_db(face_data)
                            if person_id:
                                os.makedirs(f"{folder_path}/new_clients", exist_ok=True)
                                os.rename(f'{folder_path}/{file}',
                                          f'{folder_path}/new_clients/{person_id}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                                print(f'new client {person_id} ')
                            else:
                                os.makedirs(f"{folder_path}/no_good", exist_ok=True)
                                os.rename(f'{folder_path}/{file}', f'{folder_path}/no_good/{file}')
                                # self.send_client_data(camera_id, person_id, date, file_path, face_data)

    def is_regular_client(self, face_data, file_path):
        try:
            if np.all(face_data.embedding) == 0:
                return False, 0, 0

            query = np.array(face_data.embedding).astype(np.float32).reshape(1, -1)
            index = self.client_index
            scores, ids = [i[0].tolist() for i in index.search(query, 5)]
            indices = self.client_indices
            person_ids = [int(indices[id_empl]) for id_empl in ids]
            person_id, score = person_ids[0], scores[0]

            # добавление в базу и проверка
            self.add_regular_client_to_db(face_data, score, person_id, file_path)
            return person_id, score

        except Exception as e:
            logger.error(e)
            return 0, 0

    def is_employee(self, face_data, file_path):
        try:
            if np.all(face_data.embedding) == 0:
                return False, 0, 0
            query = np.array(face_data.embedding).astype(np.float32).reshape(1, -1)
            index = self.employee_index
            scores, ids = [i[0].tolist() for i in index.search(query, 5)]
            indices = self.employee_indices
            person_ids = [int(indices[id_empl]) for id_empl in ids]
            person_id, score = max(person_ids), scores[0]

            images_count = self.mongodb.count_documents({'person_id': person_id})

            if (images_count < 40 and score > TRESHOLD_ADD_DB and face_data.det_score >= DET_SCORE_TRESH and abs(
                    face_data.pose[1]) < POSE_TRESHOLD and
                    abs(face_data.pose[0]) < POSE_TRESHOLD):
                document = self.mongodb.find_one({"person_id": person_id}, sort=[("update_date", -1)])
                doc_upd_time = datetime.strptime(document['update_date'], '%Y-%m-%d %H:%M:%S')
                delta_time = (datetime.now() - doc_upd_time).total_seconds()
                if delta_time > 4000:
                    self.add_employer_to_db(file_path, person_id)
            return score, person_id
        except Exception as e:
            self.logger.error(e)
            return False, 0, 0

    def add_regular_client_to_db(self, face_data, score, person_id, file_path):
        try:
            if face_data.det_score >= DET_SCORE_TRESH and abs(face_data.pose[1]) < POSE_TRESHOLD and abs(
                    face_data.pose[0]) < POSE_TRESHOLD:
                client_data = {
                    'score': score,
                    "person_id": person_id,
                    "embedding": face_data.embedding.tolist(),
                    "gender": face_data.gender,
                    "age": face_data.age,
                    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.clients_db.insert_one(client_data)
        except Exception as e:
            logger.error(f'Exception add image: {e}')

    def add_new_client_to_db(self, face_data):
        try:
            if face_data.det_score >= DET_SCORE_TRESH and abs(face_data.pose[1]) < POSE_TRESHOLD and abs(
                    face_data.pose[0]) < POSE_TRESHOLD:
                counter = self.counter_db.find_one_and_update(
                    {'_id': 'client_id'},
                    {'$inc': {'seq': 1}},
                    upsert=True,
                    return_document=True
                )
                person_id = counter['seq']
                client_data = {
                    "person_id": person_id,
                    "embedding": face_data.embedding.tolist(),
                    "gender": face_data.gender,
                    "age": face_data.age,
                    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.new_clients[person_id] = client_data
                self.logger.info(f"New client added with ID: {person_id}")
                if len(self.new_clients) >= INDEX_UPDATE_TRESHOLD:
                    self.update_client_index()
                return person_id
        except Exception as e:
            logger.error(f'Exception add image: {e}')

    def update_client_index(self):
        try:
            for person_id, client_data in self.new_clients.items():
                embedding = np.array(client_data["embedding"]).astype('float32').reshape(1, -1)
                self.client_index.add(embedding)
                self.client_indices.append(person_id)
                self.clients_db.insert_one(client_data)
            self.new_clients.clear()
        except Exception as e:
            self.logger.error(f'Exception updating index: {e}')

    def send_client_data(self, camera_id, person_id, date, file_path, face_data):
        try:
            url = os.getenv("SEND_CLIENT_URL")
            token = os.getenv("TOKEN_CLIENT_API")
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            }
            gender = face_data.gender
            age = face_data.age
            data = {
                "camera_id": camera_id,
                "person_id": person_id,
                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "gender": gender,
                "age": age
            }

            files = {'images': open(file_path, 'rb')}
            response = requests.post(url, data=data, files=files, headers=headers, timeout=10)
            self.logger.info(f"Sent data: {response.status_code}")
            if response.status_code != 201:
                self.logger.error(f"Error: {response.status_code} for client {person_id}")
            else:
                self.logger.info(f"Report  sent successfully for client {person_id}")
        except Exception as e:
            self.logger.error(f'Exception sending client data: {e}')

    def send_background(self, file_path, embedding):
        image = cv2.imread(file_path)
        image_data = self.app.get(image)
        for data in image_data:
            if compute_sim(data.embedding, embedding) > 0.8:
                x1, y1, x2, y2 = [int(val) for val in data.bbox]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(file_path, image)
                return file_path
        return False

    def add_employer_to_db(self, img_path, person_id):
        try:
            image_name = img_path.split('/')[-1]
            folder = f"{os.getenv('USERS_FOLDER_PATH')}/{person_id}/images"
            os.makedirs(folder, exist_ok=True)
            os.rename(img_path, f"{folder}/{image_name}")
            url = f'{os.getenv("ADD_IMAGE_TO_USER")}/{person_id}'
            token = os.getenv("TOKEN_FOR_API")
            data = {
                'image': image_name,
            }

            try:
                with requests.post(
                        url, data=data, headers={
                            "Accept": "application/json",
                            "Authorization": f"Bearer {token}"
                        },
                        timeout=10
                ) as response:
                    logger.info(f'status code add to db : {response.status_code}')
                    self.check_add_to_db = True
            except Exception as e:
                logger.error(f'Exception to sent: {e}')
        except Exception as e:
            logger.error(f'Exception add image: {e}')


if __name__ == '__main__':
    test = MainRunner(os.getenv('IMAGES_FOLDER'))
    while True:
        try:
            test.main_run()
        except Exception as e:
            print(e)
        time.sleep(5)
