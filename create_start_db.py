import os
from datetime import datetime

from insightface.app import FaceAnalysis
from pymongo import MongoClient

db = MongoClient(os.getenv('MONGODB_LOCAL'))
mongodb = db.biz_count
employees_db = mongodb.employees
clients_db = mongodb.clients

app = FaceAnalysis()
app.prepare(ctx_id=0)

face_data = app.get('/home/stargroup/photo_2024-05-03_18-46-43.jpg')[0]
client_data = {
    'score': 0,
    "person_id": 0,
    "embedding": face_data.embedding.tolist(),
    "gender": face_data.gender,
    "age": face_data.age,
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
clients_db.insert_one(client_data)
