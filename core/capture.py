import cv2

class VideoCapture:
    def __init__(self):
        self.camera_id: int = 0
    
    def start(self):
        cap = cv2.VideoCapture(self.camera_id)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Поток прерван или кадр не получен")
                yield frame
            
        finally:
            cap.release()
        