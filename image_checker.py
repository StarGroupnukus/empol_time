import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from funcs import extract_date_from_filename, get_faces_data, setup_logger, copy_files
from main import logger

load_dotenv()

DET_SCORE_THRESH = 0.75
POSE_THRESHOLD = 30


class ImageProcessor:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.org_name = os.path.basename(images_folder)
        self.cameras_path_directories = [dir for dir in os.listdir(self.images_folder) if dir.startswith('cam')]
        self.app = self.setup_face_analysis()

    def setup_face_analysis(self):
        app = FaceAnalysis()
        app.prepare(ctx_id=0)
        return app

    def main_run(self):
        threads = []
        for camera_directory in self.cameras_path_directories:
            camera_directory_path = os.path.join(self.images_folder, camera_directory)
            camera_id = camera_directory.split(' ')[1]
            thread = threading.Thread(target=self.process_images, args=(camera_directory_path, camera_id,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def process_images(self, folder_path, camera_id):
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

                if face_data.det_score < DET_SCORE_THRESH or abs(face_data.pose[1]) > POSE_THRESHOLD or abs(
                        face_data.pose[0]) > POSE_THRESHOLD:
                    self.move_file(file_path, os.path.join(folder_path, "low_quality", file))
                    os.remove(orig_image_path)
                    continue

                background_image_path = self.process_background(orig_image_path, face_data.embedding)
                if background_image_path:
                    self.send_data_for_identification(camera_id, background_image_path, face_data.embedding, date)
                else:
                    os.remove(orig_image_path)

    def process_background(self, file_path, embedding):
        image = cv2.imread(file_path)
        image_data = self.app.get(image)
        for data in image_data:
            if np.dot(data.embedding, embedding) / (np.linalg.norm(data.embedding) * np.linalg.norm(embedding)) > 0.8:
                x1, y1, x2, y2 = map(int, data.bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(file_path, image)
                return file_path
        return None

    def send_data_for_identification(self, camera_id, background_image_path, embedding, date):
        #send with API

    def move_file(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)


if __name__ == '__main__':
    processor = ImageProcessor(os.getenv('IMAGES_FOLDER'))
    while True:
        try:
            processor.main_run()
        except Exception as e:
            logger.error(f"Error in main run: {e}")
        time.sleep(5)
