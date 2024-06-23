import os
from datetime import datetime
import cv2


def add_face_data_to_db(app, db, image_path):
    image = cv2.imread(image_path)
    face_data = app.get(image)[0]

    client_data = {
        'score': 0,
        "person_id": 0,
        "embedding": face_data.embedding.tolist(),
        "gender": face_data.gender,
        "age": face_data.age,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    db.insert_one(client_data)
