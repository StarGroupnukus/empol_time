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
from download_file import new_create_indexes, update_employees_database, update_database
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger, send_report

# Load environment variables
load_dotenv()

# Constants
CHECK_NEW_CLIENT = 0.5
TRESHOLD_IS_DB = 14
POSE_TRESHOLD = 30
DET_SCORE_TRESH = 0.75
IMAGE_COUNT = 10
TRESHOLD_ADD_DB = 19
DIMENSIONS = 512
INDEX_UPDATE_TRESHOLD = 5
INIT_IMAGE_PATH = './pavel.png'

# Setup logger
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
        self.client_index, self.client_indices = new_create_indexes(self.clients_db, self.org_name, 'client', logger)
        self.initialize_counter('client_id')
        self.check_add_to_db = False
        self.employee_index = faiss.read_index(f'index_file{self.org_name}.index')
        self.employee_indices = np.load(f'indices{self.org_name}.npy', allow_pickle=True)
        self.new_clients = []
        self.lock = threading.Lock()  # Thread locking to prevent race conditions

    def setup_app(self):
        """Initialize the FaceAnalysis app."""
        app = FaceAnalysis()
        app.prepare(ctx_id=0)
        update_employees_database(self.employees_db, self.org_name, app=app)
        return app

    def initialize_counter(self, counter_id):
        """Initialize a counter in the database."""
        if self.counter_db.find_one({'_id': counter_id}) is None:
            self.counter_db.insert_one({'_id': counter_id, 'seq': 0})
            self.logger.info(f"Initialized counter for {counter_id}")

    def init_clients_db(self):
        """Initialize the clients database with a default image."""
        image = cv2.imread(INIT_IMAGE_PATH)
        face_data = self.app.get(image)[0]
        client_data = {
            "person_id": 0,
            "embedding": face_data.embedding.tolist(),
        }
        self.clients_db.insert_one(client_data)

    def initialize_client_index(self):
        """Initialize the FAISS index for clients."""
        client_data = list(self.clients_db.find())
        if len(client_data) == 0:
            self.init_clients_db()
        clients_index = faiss.read_index(f'index_file{self.org_name}_client.index')
        clients_indices = np.load(f'indices{self.org_name}.npy', allow_pickle=True)
        return clients_index, clients_indices

    def main_run(self):
        """Main function to process images from cameras."""
        threads = []
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('test'):
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
        if self.new_clients:
            self.update_client_index()
        if self.check_add_to_db:
            update_database(self.org_name, app=self.app)
            self.check_add_to_db = False

    def classify_images(self, folder_path, camera_id):
        """Classify images from a given folder path and camera ID."""
        list_files = [file for file in os.listdir(folder_path) if file.endswith('SNAP.jpg')]
        for file in list_files:
            file_path = os.path.join(folder_path, file)
            orig_image_path = file_path.replace('SNAP', 'BACKGROUND')
            if not os.path.exists(orig_image_path):
                continue
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                os.remove(orig_image_path)
                continue
            image = cv2.imread(file_path)
            date = extract_date_from_filename(file)
            try:
                faces = self.app.get(image)
            except Exception as e:
                self.logger.error(f'ERROR for app get: {e}')
                continue
            if not faces:
                os.makedirs(f"{folder_path}/not_face", exist_ok=True)
                os.rename(file_path, f'{folder_path}/not_face/{file}')
                os.remove(orig_image_path)
                continue
            face_data = get_faces_data(faces)
            score, person_id = self.is_employee(face_data, file_path)
            self.logger.info(f"Employee Score {score}")
            if score == 0:
                os.makedirs(f"{folder_path}/error", exist_ok=True)
                os.rename(file_path, f'{folder_path}/error/{file}')
                os.remove(orig_image_path)
                continue
            if score > TRESHOLD_IS_DB:
                os.makedirs(f"{folder_path}/recognized", exist_ok=True)
                os.rename(f'{folder_path}/{file}',
                          f'{folder_path}/recognized/{person_id}_{score}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                back_file_name = self.send_background(orig_image_path, face_data.embedding)
                if back_file_name:
                    send_report(camera_id, person_id, back_file_name, date, score, self.logger)
                else:
                    os.remove(orig_image_path)
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
                        self.logger.info(
                            f"================is_regular_client score:{score} id:{person_id}================")
                        self.add_regular_client_to_db(face_data, score, person_id, file_path, date)
                else:
                    person_id = self.add_new_client_to_db(face_data, file_path, date)
                    if person_id:
                        os.makedirs(f"{folder_path}/new_clients", exist_ok=True)
                        os.rename(f'{folder_path}/{file}',
                                  f'{folder_path}/new_clients/{person_id}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                        self.logger.info(f'new client {person_id} ')
                    else:
                        os.makedirs(f"{folder_path}/no_good", exist_ok=True)
                        os.rename(f'{folder_path}/{file}', f'{folder_path}/no_good/{file}')

    def is_regular_client(self, face_data, file_path):
        """Check if the face data belongs to a regular client."""
        try:
            if np.all(face_data.embedding) == 0:
                return 0, 0
            query = np.array(face_data.embedding).astype(np.float32).reshape(1, -1)
            scores, ids = self.client_index.search(query, 1)
            indices = self.client_indices
            person_id = int(indices[ids[0][0]])
            score = scores[0][0]
            return score, person_id
        except Exception as e:
            self.logger.error(e)
            return 0, 0

    def is_employee(self, face_data, file_path):
        """Check if the face data belongs to an employee."""
        try:
            if np.all(face_data.embedding) == 0:
                return 0, 0
            query = np.array(face_data.embedding).astype(np.float32).reshape(1, -1)
            scores, ids = self.employee_index.search(query, 1)
            indices = self.employee_indices
            person_id = int(indices[ids[0][0]])
            score = scores[0][0]

            images_count = self.employees_db.count_documents({'person_id': person_id})

            if (images_count < 40 and score > TRESHOLD_ADD_DB and face_data.det_score >= DET_SCORE_TRESH and
                abs(face_data.pose[1]) < POSE_TRESHOLD and abs(face_data.pose[0]) < POSE_TRESHOLD):
                document = self.employees_db.find_one({"person_id": person_id}, sort=[("update_date", -1)])
                doc_upd_time = datetime.strptime(document['update_date'], '%Y-%m-%d %H:%M:%S')
                delta_time = (datetime.now() - doc_upd_time).total_seconds()
                if delta_time > 4000:
                    self.add_employer_to_db(file_path, person_id)
            return score, person_id
        except Exception as e:
            self.logger.error(e)
            return 0, 0

    def add_regular_client_to_db(self, face_data, score, person_id, file_path, date):
        """Add regular client data to the database."""
        try:
            if (face_data.det_score >= DET_SCORE_TRESH and
                abs(face_data.pose[1]) < POSE_TRESHOLD and abs(face_data.pose[0]) < POSE_TRESHOLD):
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
                self.clients_db.insert_one(client_data)
                self.logger.info("Regular client checked and added to db.")
            else:
                self.logger.info("One of the conditions failed for regular client.")
        except Exception as e:
            self.logger.error(f'Exception adding regular client: {e}')

    def add_new_client_to_db(self, face_data, file_path, date):
        """Add new client data to the database."""
        self.logger.info("Attempting to add a new client.")
        try:
            if (face_data.det_score >= DET_SCORE_TRESH and
                abs(face_data.pose[1]) < POSE_TRESHOLD and abs(face_data.pose[0]) < POSE_TRESHOLD):
                new_client_id = self.check_new_clients(face_data)
                if new_client_id == 0:
                    counter = self.counter_db.find_one_and_update(
                        {'_id': 'client_id'},
                        {'$inc': {'seq': 1}},
                        upsert=True,
                        return_document=True
                    )
                    person_id = counter['seq']
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
                with self.lock:  # Lock during new client addition
                    self.new_clients.append(client_data)
                self.logger.info(f"New client added with ID: {person_id}")
                if len(self.new_clients) >= INDEX_UPDATE_TRESHOLD:
                    threading.Thread(target=self.update_client_index).start()  # Async index update
                return person_id
        except Exception as e:
            self.logger.error(f'Exception adding new client: {e}')

    def check_new_clients(self, face_data):
        """Check if the new client data already exists in the new clients list."""
        new_embedding = np.array(face_data.embedding)
        for client_data in self.new_clients:
            existing_embedding = np.array(client_data['embedding'])
            similarity = compute_sim(new_embedding, existing_embedding)
            if similarity > CHECK_NEW_CLIENT:
                self.logger.info("Client with similar embedding already exists in new_clients.")
                return client_data['person_id']
        return 0

    def update_client_index(self):
        """Update the FAISS index with new client data."""
        try:
            embeddings = []
            client_ids = []
            client_data_list = []
            with self.lock:  # Lock during index update
                for client_data in self.new_clients:
                    embedding = np.array(client_data["embedding"])
                    embeddings.append(embedding)
                    client_ids.append(client_data["person_id"])
                    client_data_list.append(client_data)
                if embeddings:
                    vectors = np.array(embeddings).astype('float32')
                    faiss.normalize_L2(vectors)
                    self.client_index.add(vectors)
                    self.client_indices.extend(client_ids)
                    self.clients_db.insert_many(client_data_list)
                    np.save(f'indices{self.org_name}.npy', np.array(self.client_indices, dtype=object),
                            allow_pickle=True)
                    self.logger.info(f"Client index updated. Index length: {self.client_index.ntotal}")
                self.new_clients.clear()
        except Exception as e:
            self.logger.error(f'Exception updating client index: {e}')

    def send_client_data(self, camera_id, person_id, date, file_path, face_data):
        """Send client data to an external API."""
        try:
            url = os.getenv("SEND_CLIENT_URL")
            token = os.getenv("TOKEN_CLIENT_API")
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            }
            data = {
                "camera_id": camera_id,
                "person_id": person_id,
                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "gender": face_data.gender,
                "age": face_data.age
            }
            files = {'images': open(file_path, 'rb')}
            response = requests.post(url, data=data, files=files, headers=headers, timeout=10)
            self.logger.info(f"Sent data: {response.status_code}")
            if response.status_code != 201:
                self.logger.error(f"Error: {response.status_code} for client {person_id}")
            else:
                self.logger.info(f"Report sent successfully for client {person_id}")
        except Exception as e:
            self.logger.error(f'Exception sending client data: {e}')

    def send_background(self, file_path, embedding):
        """Send the background image with face highlighted."""
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
        """Add employee image to the database."""
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
            self.logger.info(f'Status code add to db: {response.status_code}')
            self.check_add_to_db = True
        except Exception as e:
            self.logger.error(f'Exception adding employee image: {e}')


if __name__ == '__main__':
    test = MainRunner(os.getenv('IMAGES_FOLDER'))
    while True:
        try:
            test.main_run()
        except Exception as e:
            logger.error(e)
        time.sleep(5)
