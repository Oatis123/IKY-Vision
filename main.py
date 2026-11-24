import cv2
from core import VideoCapture, FaceDetector, FaceRecognizer, PersonTracker

stream = VideoCapture()
detector = FaceDetector(conf=0.5)
recognizer = FaceRecognizer("faces.pkl", threshold=0.6)
tracker = PersonTracker()

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", 1280, 720)

for frame in stream.start():
    h, w = frame.shape[:2]
    faces = detector.detect(frame)
    names = []

    for x1, y1, x2, y2, _ in faces:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        name, sim = recognizer.recognize(face_crop)
        names.append(name)
        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} ({sim:.2f})",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    
    missing = tracker.track(names)
    
    print(f"Камера потеряла: {', '.join(names)}")

    cv2.imshow("Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()