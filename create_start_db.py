import os
from datetime import datetime

from insightface.app import FaceAnalysis
from pymongo import MongoClient


def add_face_data_to_db(app,db, image_path):
    face_data = app.get(image_path)[0]

    client_data = {
        'score': 0,
        "person_id": 0,
        "embedding": face_data.embedding.tolist(),
        "gender": face_data.gender,
        "age": face_data.age,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    db.insert_one(client_data)
