import cv2
import os
import numpy as np
from PIL import Image

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for imagePath in image_paths:
        gray_img = Image.open(imagePath).convert('L')
        img_np = np.array(gray_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])

        face_samples.append(img_np)
        ids.append(id)

    return face_samples, ids

faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

if not os.path.exists('trainer'):
    os.makedirs('trainer')

recognizer.write('trainer/trainer.yml')

print("Training Complete")
