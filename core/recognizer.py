import os
import pickle
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1


class FaceRecognizer:
    def __init__(self, db_path: str | None = "faces.pkl", threshold: float = 0.6):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self.names: list[str] = []
        self.vectors: np.ndarray = np.empty((0, 512), dtype=np.float32)

        if db_path is not None:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"База лиц не найдена: {db_path}. Сначала запусти enroll.py")
            with open(db_path, "rb") as f:
                data = pickle.load(f)
                self.names = data["names"]
                self.vectors = data["vectors"]

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160, 160))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy()[0]

        return emb

    def recognize(self, face_crop: np.ndarray) -> tuple[str, float]:
        if len(self.names) == 0:
            return "Unknown", 0.0

        emb = self.get_embedding(face_crop)

        similarities = self.vectors @ emb
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self.threshold:
            return "Unknown", best_score

        return self.names[best_idx], best_score