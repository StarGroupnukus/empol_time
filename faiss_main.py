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

from download_file import update_database
from funcs import extract_date_from_filename, send_report, get_faces_data, setup_logger, compute_sim

load_dotenv()

logger = setup_logger('Mainrunning', 'logs/Mainrunning.log')

TRESHOLD_IS_DB = 14
POSE_TRESHOLD = 30
DET_SCORE_TRESH = 0.75
IMAGE_COUNT = 10
TRESHOLD_ADD_DB = 19
DIMENSIONS = 512


class MainRunner:
    def __init__(self, images_folder, ):
        self.images_folder = images_folder
        self.org_name = images_folder.split('/')[-1]
        self.cameras_path_directories = [dir for dir in os.listdir(self.images_folder)]
        self.app = self.setup_app()
        self.check_add_to_db = False
        self.db = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.mongodb = self.db[os.getenv("DB_NAME")][self.org_name]
        self.fais_index = faiss.read_index(f'index_file{self.org_name}.index')
        with open(f'indices{self.org_name}.npy', 'rb') as f:
            self.indices = np.load(f, allow_pickle=True)

    def setup_app(self):
        app = FaceAnalysis()
        app.prepare(ctx_id=0)
        update_database(self.org_name)
        return app

    def main_run(self):
        threads = []
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('cam'):
                continue
            camera_directory = f"{self.images_folder}/{camera_directory}"
            camera_id = camera_directory.split(' ')[1]
            time.sleep(1)
            thread = threading.Thread(target=self.classify_images, args=(camera_directory, camera_id,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if self.check_add_to_db:
            update_database(self.org_name)

    def classify_images(self, folder_path, camera_id, ):
        list_files = [file for file in os.listdir(folder_path) if file.endswith('SNAP.jpg')]
        for file in list_files:
            file_path = os.path.join(folder_path, file)
            orig_image_path = file_path.replace('SNAP', 'BACKGROUND')
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                continue
            elif os.path.exists(orig_image_path):
                image = cv2.imread(file_path)
                date = extract_date_from_filename(file)
                face = self.app.get(image)

                if not face:
                    os.makedirs(f"{folder_path}/not_face", exist_ok=True)
                    os.rename(file_path, f'{folder_path}/not_face/{file}')
                    os.remove(orig_image_path)
                    continue
                else:
                    face_data = get_faces_data(face)

                score, person_id = self.who_is_this(face_data, file_path)
                logger.info(f'{score}, {person_id}')
                if score == 0:
                    os.makedirs(f"{folder_path}/error", exist_ok=True)
                    os.rename(file_path, f'{folder_path}/error/{file}')
                    os.remove(orig_image_path)
                    continue
                if score > TRESHOLD_IS_DB:
                    os.makedirs(f"{folder_path}/unknowns", exist_ok=True)
                    os.rename(file_path,
                              f'{folder_path}/unknowns/{score}_{date.strftime("%Y-%m-%d_%H-%M-%S_%f")[:23]}.jpg')
                    os.remove(orig_image_path)

                else:
                    if os.path.isfile(f'{folder_path}/{file}'):
                        os.makedirs(f"{folder_path}/recognized", exist_ok=True)
                        os.rename(f'{folder_path}/{file}',
                                  f'{folder_path}/recognized/{person_id}_{score}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                    back_file_name = self.send_background(orig_image_path, face_data.embedding)
                    if back_file_name:
                        send_report(camera_id,
                                    person_id,
                                    back_file_name,
                                    date,
                                    score,
                                    logger)
                    else:
                        os.remove(orig_image_path)

    def who_is_this(self, face_data, file_path):
        try:
            if np.all(face_data.embedding) == 0:
                return 0, 0
            query = np.array(face_data.embedding).astype(np.float32).reshape(1, -1)
            scores, ids = [i[0].tolist() for i in self.fais_index.search(query, 5)]
            person_ids = [int(self.indices[id]) for id in ids]
            person_id, score = person_ids[0], scores[0]

            images_count = self.mongodb.count_documents({'person_id': person_id})

            if (images_count < 40 and score < TRESHOLD_ADD_DB and face_data.det_score >= DET_SCORE_TRESH and abs(
                    face_data.pose[1]) < POSE_TRESHOLD and
                    abs(face_data.pose[0]) < POSE_TRESHOLD):
                document = self.mongodb.find_one({"person_id": person_id}, sort=[("update_date", -1)])
                doc_upd_time = datetime.strptime(document['update_date'], '%Y-%m-%d %H:%M:%S')
                delta_time = (datetime.now() - doc_upd_time).total_seconds()
                # print(delta_time)
                if delta_time > 4000:
                    self.add_to_db(file_path, person_id)
            return score, person_id

        except Exception as e:
            logger.error(e)
            return 0, 0

    def send_background(self, file_path, embedding, ):
        image = cv2.imread(file_path)
        image_data = self.app.get(image)
        for data in image_data:
            if compute_sim(data.embedding, embedding) > 0.8:
                x1, y1, x2, y2 = [int(val) for val in data.bbox]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(file_path, image)
                return file_path
        return False

    def add_to_db(self, img_path, person_id, ):
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
                    # print(response.text)
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