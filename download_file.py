import json
import os
import time
import urllib.request

import cv2
import numpy as np
import requests
from annoy import AnnoyIndex
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from pymongo import MongoClient

from funcs import get_faces_data

load_dotenv()

mongo_url = os.getenv("MONGODB_LOCAL")
client = MongoClient(mongo_url)


def download_file(filename):
    url = os.getenv('SEND_REPORT_API')
    token = os.getenv('TOKEN_FOR_API')

    headers = {
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Это вызовет исключение, если запрос завершится с ошибкой.

        with open(filename, 'wb') as f:
            f.write(response.content)

        print(f"File downloaded successfully as '{filename}'")
    except requests.exceptions.RequestException as e:
        print("Failed to download file->", e)


def process_json(json_data, db, app):
    update_count = 0
    active_ids = []
    for item in json_data['data']:
        id = item['id']
        for image in item['images']:
            img_url = image['url']
            img_id = image['id']
            active_ids.append(img_id)
            if not db.find_one({'_id': img_id}):
                try:
                    print(img_url)
                    print(img_id)
                    req = urllib.request.urlopen(img_url)
                    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    face = app.get(img)
                    face_data = get_faces_data(face)
                    embedding = face_data.embedding.tolist()
                    processed_item = {
                        '_id': img_id,
                        'person_id': id,
                        'image_url': img_url,
                        'embedding': embedding,
                        'det_score': round((float(face_data.det_score) * 100), 3),
                        'pose': face_data.pose.tolist(),
                        "update_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    db.insert_one(processed_item)
                    update_count += 1
                except Exception as e:
                    print("Failed to process image->", e)

    print(f'Updated {update_count}')
    db.delete_many({'_id': {'$nin': active_ids}})


def get_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("Файл не найден.")
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON. Убедитесь, что файл содержит корректный JSON.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def to_build(collection, ann_file, tree_n=40):
    num_dimensions = 512

    t = AnnoyIndex(num_dimensions, 'euclidean')
    for doc in collection.find({}):
        t.add_item(doc['_id'], doc['embedding'])

    t.build(tree_n)
    os.makedirs('embeddings', exist_ok=True)
    t.save(f'embeddings/{ann_file}.ann')


def update_database(org_name):
    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    file_name = f'{org_name}.json'
    download_file(file_name)

    data = get_data(file_name)
    collection = org_name
    db = client[os.getenv("DB_NAME")][collection]
    start_time = time.time()
    process_json(data, db, app)
    print(time.time() - start_time)
    os.remove(file_name)

    to_build(db, org_name, tree_n=40)


if __name__ == '__main__':
    app = FaceAnalysis()
    app.prepare(ctx_id=0)
    update_database(org_name='ecosun')
