from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import numpy as np

class FaceDetector:
    def __init__(self, model_path: str | None = None, conf: float = 0.5):
        if model_path is None:
            model_path = hf_hub_download(
                repo_id="arnabdhar/YOLOv8-Face-Detection",
                filename="model.pt",
                local_dir="models",
            )
            
        self.model = YOLO(model_path)
        self.conf = conf
    
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        
        faces = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            faces.append((x1, y1, x2, y2, score))
        
        return faces