import cv2
import os


class FaceRecognizer:
    def __init__(self, db_cursor):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        trainer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trainer', 'trainer.yml')
        if os.path.exists(trainer_path):
            self.recognizer.read(trainer_path)
        else:
            print("Warning: trainer.yml not found. Face recognition may not work properly.")

        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        self.faceCascade = cv2.CascadeClassifier(cascade_path)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Lấy danh sách tên từ cơ sở dữ liệua
        db_cursor.execute("SELECT DISTINCT msv, name FROM students")
        self.id_name_map = {i: row[1] for i, row in enumerate(db_cursor.fetchall())}

    def recognize_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))

        recognized_faces = []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 85:
                person_name = self.id_name_map.get(id, "Unknown")
                confidence_value = round(100 - confidence, 2)
            else:
                person_name = "Unknown"
                confidence_value = 0

            # Hiển thị tên và độ tin cậy lên ảnh
            label = f"{person_name} ({confidence_value:.2f}%)"

            # Calculate position for text
            label_size, _ = cv2.getTextSize(label, self.font, 0.7, 2)
            text_x = x + (w - label_size[0]) // 2  # Center text horizontally
            text_y = y - 10 if y - 10 > 10 else y + h + 30

            # Draw a filled rectangle as background for text
            cv2.rectangle(img, (text_x - 5, text_y - label_size[1] - 5),
                          (text_x + label_size[0] + 5, text_y + 5),
                          (0, 255, 0), cv2.FILLED)

            # Put text on the image
            cv2.putText(img, label, (text_x, text_y), self.font, 0.7, (0, 0, 0), 2)

            recognized_faces.append((person_name, confidence_value))

        return img, recognized_faces


def get_face_recognizer(db_cursor):
    return FaceRecognizer(db_cursor)
