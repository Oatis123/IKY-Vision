import pickle
from pathlib import Path
import cv2
import numpy as np
from core import FaceDetector, FaceRecognizer

dataset_dir = Path("dataset")
detector = FaceDetector(conf=0.6)
recognizer = FaceRecognizer(db_path=None)

names = []
vectors = []

for person_dir in dataset_dir.iterdir():
    if not person_dir.is_dir():
        continue

    person_name = person_dir.name
    for img_path in person_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = detector.detect(img)
        if not faces:
            print(f"[Пропуск] Лицо не найдено на {img_path}")
            continue

        x1, y1, x2, y2, _ = faces[0]
        h, w = img.shape[:2]
        face_crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        emb = recognizer.get_embedding(face_crop)
        names.append(person_name)
        vectors.append(emb)
        print(f"[OK] Добавлен: {person_name} ({img_path.name})")

with open("faces.pkl", "wb") as f:
    pickle.dump({"names": names, "vectors": np.array(vectors, dtype=np.float32)}, f)

print(f"Готово! Сохранено векторов: {len(names)}")