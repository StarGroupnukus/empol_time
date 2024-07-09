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
from funcs import extract_date_from_filename, send_report, get_faces_data, setup_logger, compute_sim, copy_files

load_dotenv()

logger = setup_logger('Mainrunning', 'logs/Mainrunning.log')

TRESHOLD_IS_DB = 13
POSE_TRESHOLD = 30
DET_SCORE_TRESH = 0.75
IMAGE_COUNT = 10
TRESHOLD_ADD_DB = 15
DIMENSIONS = 512


class MainRunner:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.org_name = os.path.basename(images_folder)
        self.cameras_path_directories = [dir for dir in os.listdir(self.images_folder)]
        self.check_add_to_db = False
        self.app = self.setup_face_analysis()
        self.db = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.mongodb = self.db[os.getenv("DB_NAME")][self.org_name]
        self.fais_index, self.indices = update_database(org_name=self.org_name, app=self.app)


    def setup_face_analysis(self):
        app = FaceAnalysis()
        app.prepare(ctx_id=0)

        return app

    def main_run(self):
        threads = []
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('cam'):
                continue
            camera_directory_path = os.path.join(self.images_folder, camera_directory)
            camera_id = camera_directory.split(' ')[1]
            thread = threading.Thread(target=self.classify_images, args=(camera_directory_path, camera_id,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if self.check_add_to_db:
            self.fais_index, self.indices = update_database(self.org_name, app=self.app)
            self.check_add_to_db = False

    def classify_images(self, folder_path, camera_id):
        list_files = [file for file in os.listdir(folder_path) if file.endswith('SNAP.jpg')]
        for file in list_files:
            time.sleep(1)
            file_path = os.path.join(folder_path, file)
            orig_image_path = file_path.replace('SNAP', 'BACKGROUND')
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                os.remove(orig_image_path)
                continue
            if os.path.exists(orig_image_path):
                copy_files(file_path, orig_image_path, f'{os.getenv("IMAGES_FOLDER")}/test')
                image = cv2.imread(file_path)
                date = extract_date_from_filename(file)
                faces = self.app.get(image)

                if not faces:
                    self.move_file(file_path, os.path.join(folder_path, "not_face", file))
                    os.remove(orig_image_path)
                    continue

                face_data = get_faces_data(faces)
                score, person_id = self.who_is_this(face_data, file_path)

                if score == 0:
                    self.move_file(file_path, os.path.join(folder_path, "error", file))
                    os.remove(orig_image_path)
                    continue
                if score < TRESHOLD_IS_DB:
                    self.move_file(file_path, os.path.join(folder_path, "unknowns", f'{round(score, 2)}_{date.strftime("%Y-%m-%d_%H-%M-%S_%f")[:23]}.jpg'))
                    os.remove(orig_image_path)
                else:
                    if os.path.exists(file_path):
                        recognized_path = os.path.join(folder_path, "recognized", f'{person_id}_{round(score, 2)}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                        self.move_file(file_path, recognized_path)
                    back_file_name = self.send_background(orig_image_path, face_data.embedding)
                    if back_file_name:
                        send_report(camera_id, person_id, back_file_name, date, score, logger)
                    else:
                        os.remove(orig_image_path)

    def who_is_this(self, face_data, file_path):
        try:
            if np.all(face_data.embedding) == 0:
                return 0, 0
            query = np.array(face_data.embedding.tolist(), dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query)
            scores, ids = self.fais_index.search(query, 5)
            scores, ids = scores[0], ids[0]
            print(scores)
            person_ids = [int(self.indices[id]) for id in ids]
            person_id, score = person_ids[0], scores[0]

            images_count = self.mongodb.count_documents({'person_id': person_id})

            if images_count < 40 and score > TRESHOLD_ADD_DB and face_data.det_score >= DET_SCORE_TRESH \
                    and abs(face_data.pose[1]) < POSE_TRESHOLD and abs(face_data.pose[0]) < POSE_TRESHOLD:
                document = self.mongodb.find_one({"person_id": person_id}, sort=[("update_date", -1)])
                if document:
                    doc_upd_time = datetime.strptime(document['update_date'], '%Y-%m-%d %H:%M:%S')
                    delta_time = (datetime.now() - doc_upd_time).total_seconds()
                    if delta_time > 4000:
                        self.add_to_db(file_path, person_id)
            return score, person_id

        except Exception as e:
            logger.error(f'Exeption WHO IS THIS {e}')
            return 0, 0

    def send_background(self, file_path, embedding):
        image = cv2.imread(file_path)
        image_data = self.app.get(image)
        for data in image_data:
            if compute_sim(data.embedding, embedding) > 0.8:
                x1, y1, x2, y2 = map(int, data.bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(file_path, image)
                return file_path
        return False

    def add_to_db(self, img_path, person_id):
        try:
            image_name = os.path.basename(img_path)
            folder = os.path.join(os.getenv('USERS_FOLDER_PATH'), str(person_id), "images")
            os.makedirs(folder, exist_ok=True)
            os.rename(img_path, os.path.join(folder, image_name))
            url = f'{os.getenv("ADD_IMAGE_TO_USER")}/{person_id}'
            token = os.getenv("TOKEN_FOR_API")
            data = {'image': image_name}

            response = requests.post(url, data=data, headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}"
            }, timeout=10)
            logger.info(f'status code add to db : {response.status_code}')
            self.check_add_to_db = True

        except Exception as e:
            logger.error(f'Exception add image: {e}')

    def move_file(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)


if __name__ == '__main__':
    test = MainRunner(os.getenv('IMAGES_FOLDER'))
    while True:
        try:
            test.main_run()
        except Exception as e:
            print(e)
        time.sleep(5)
