import json
import os
import time
import urllib.request

import cv2
import faiss
import numpy as np
import requests
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

    headers = {'Authorization': f'Bearer {token}'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

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
                    if face:
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
            return json.load(file)
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("JSON decode error. Ensure the file contains valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_indexes(db, org_id):
    docs = db.find({})
    embeddings = []
    indices = []
    for doc in docs:
        embeddings.append(doc['embedding'])
        indices.append(doc['person_id'])
    vectors = np.array(embeddings).astype('float32')
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, f'index_file{org_id}.index')
    with open(f'indices{org_id}.npy', 'wb') as f:
        np.save(f, indices)

def new_create_indexes(db, org_id, role):
    docs = db.find()
    embeddings = []
    indices = []
    for doc in docs:
        embeddings.append(doc['embedding'])
        indices.append(doc['person_id'])

    if len(embeddings) == 0:
        return None, None
    vectors = np.array(embeddings).astype('float32')
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if role != 'client':
        faiss.write_index(index, f'index_file{org_id}.index')
        with open(f'indices{org_id}.npy', 'wb') as f:
            np.save(f, indices)
    else:
        if len(vectors) == 0:
            return faiss.IndexFlatIP(512), []
        return index, indices


def update_database(org_name, app):
    file_name = f'{org_name}.json'
    download_file(file_name)

    data = get_data(file_name)
    collection = org_name
    db = client[os.getenv("DB_NAME")][collection]
    start_time = time.time()
    process_json(data, db, app)
    print(f"Time taken: {time.time() - start_time} seconds")
    os.remove(file_name)

    # Обновить new_create_indexes
    create_indexes(db, org_name)


if __name__ == '__main__':
    app = FaceAnalysis()
    app.prepare(ctx_id=0)
    update_database(org_name='ecosun', app=app)
