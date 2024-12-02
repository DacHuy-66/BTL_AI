import cv2
import numpy as np
from PIL import Image
import io
import os
import sqlite3


# lấy dữ liệu từ dataset
def get_images_from_dataset(dataset_path):
    image_data = []
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                with open(image_path, 'rb') as file:
                    image_data.append((person_folder, file.read()))
    return image_data


# lấy dữ liệu ảnh từ cơ sở dữ liệu
def get_images_from_database(db_cursor):
    db_cursor.execute("SELECT msv, image FROM face_images")
    return db_cursor.fetchall()


# huấn luyện bộ nhận diện khuôn mặt
def train_face_recognizer(image_data):
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=5, grid_x=5, grid_y=5)
    # tệp XML chứa mô hình Haar Cascade cho việc phát hiện khuôn mặt.
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    detector = cv2.CascadeClassifier(cascade_path)

    faces = []
    ids = []

    # Tiền xử lý ảnh
    for msv, img_data in image_data:
        pil_img = Image.open(io.BytesIO(img_data)).convert('L')
        img_numpy = np.array(pil_img, 'uint8')
        face = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in face:
            faces.append(img_numpy[y:y + h, x:x + w])
            ids.append(msv)

    # Huấn luyện mô hình
    print("\n[INFO] Training data...")

    if not faces:
        print("[ERROR] No faces found in the images. Check your data.")
        return 0

    unique_ids = list(set(ids))
    id_to_index = {id: index for index, id in enumerate(unique_ids)}
    numeric_ids = [id_to_index[id] for id in ids]

    recognizer.train(faces, np.array(numeric_ids))

    # Lưu mô hình
    trainer_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trainer')
    if not os.path.exists(trainer_dir):
        os.makedirs(trainer_dir)
    recognizer.write(os.path.join(trainer_dir, 'trainer.yml'))

    print(f"\n[INFO] {len(unique_ids)} faces trained. Exiting.")
    return len(unique_ids)


def main():
    # Path to the dataset folder
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

    # Connect to the database
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()

    # Get images from dataset
    dataset_images = get_images_from_dataset(dataset_path)

    # Get images from database
    db_images = get_images_from_database(cursor)

    # Combine images from both sources
    all_images = dataset_images + db_images

    # Train the face recognizer
    num_faces_trained = train_face_recognizer(all_images)

    print(f"Total faces trained: {num_faces_trained}")

    # Close the database connection
    conn.close()
