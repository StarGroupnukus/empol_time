import os
import threading
import time
import cv2
import numpy as np
import requests
from annoy import AnnoyIndex
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from download_file import update_database
from funcs import extract_date_from_filename, send_report, get_faces_data, setup_logger, compute_sim
load_dotenv()

logger = setup_logger('Mainrunning', 'logs/Mainrunning.log')

TRESHOLD_IS_DB = 20
POSE_TRESHOLD = 30
DET_SCORE_TRESH = 0.7
IMAGE_COUNT = 10
TRESHOLD_ADD_DB = 19
DIMENSIONS = 512


class MainRunner:
    def __init__(self, images_folder,):
        self.images_folder = images_folder
        self.org_name = images_folder.split('/')[-1]
        self.cameras_path_directories = [dir for dir in os.listdir(self.images_folder)]
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0)
        self.app_detection = FaceAnalysis(allowed_modules='detection')
        self.app_detection.prepare(ctx_id=0)
        self.db = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.mongodb = self.db[os.getenv("DB_NAME")][self.org_name]
        update_database(self.org_name)
        self.annoy = AnnoyIndex(DIMENSIONS, metric='euclidean')
        self.annoy.load(f'embeddings/{self.org_name}.ann')

    def main_run(self):
        threads = []
        print(self.cameras_path_directories)
        for camera_directory in self.cameras_path_directories:
            if not camera_directory.startswith('cam'):
                continue
            camera_directory = f"{self.images_folder}/{camera_directory}"
            camera_id = camera_directory.split(' ')[1]
            time.sleep(1)
            print(camera_id)
            thread = threading.Thread(target=self.classify_images, args=(camera_directory, camera_id,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

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

                score, image_id = self.who_is_this(face_data, file_path)
                print(score, image_id)
                if score == 0:
                    os.makedirs(f"{folder_path}/error", exist_ok=True)
                    os.rename(file_path, f'{folder_path}/error/{file}')
                    os.remove(orig_image_path)
                    continue
                if score > TRESHOLD_IS_DB:
                    os.makedirs(f"{folder_path}/unknowns", exist_ok=True)
                    os.rename(file_path,
                              f'{folder_path}/unknowns/{date.strftime("%Y-%m-%d_%H-%M-%S_%f")[:23]}.jpg')
                    os.remove(orig_image_path)

                else:

                    back_file_name = self.send_background(orig_image_path, face_data.embedding)
                    os.makedirs(f"{folder_path}/recognized", exist_ok=True)
                    os.rename(f'{folder_path}/{file}',
                              f'{folder_path}/recognized/{image_id}_{score}_{date.strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
                    if back_file_name:
                        person_id = self.mongodb.find_one({'_id': image_id})['person_id']
                        send_report(camera_id,
                                    person_id,
                                    image_id,
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

            image_ids, scores = self.annoy.get_nns_by_vector(face_data.embedding, 2, include_distances=True)

            image_id, score = image_ids[0], scores[0]
            if score < TRESHOLD_ADD_DB and face_data.det_score >= DET_SCORE_TRESH and abs(
                        face_data.pose[1]) < POSE_TRESHOLD and abs(
                        face_data.pose[0]) < POSE_TRESHOLD:
                self.add_to_db(file_path, image_id)
            return score, image_id

        except Exception as e:
            logger.error(e)
            return 0, 0

    def send_background(self, file_path, embedding,):
        image = cv2.imread(file_path)
        image_data = self.app.get(image)
        for data in image_data:
            if compute_sim(data.embedding, embedding) > 0.8:
                x1, y1, x2, y2 = [int(val) for val in data.bbox]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(file_path, image)
                return file_path
        return False

    def add_to_db(self, img_path, image_id, ):
        try:
            person_id = self.mongodb.find_one({'_id': image_id})['person_id']
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