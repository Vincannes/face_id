import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('datas/haarcascade_frontalface_default.xml')

imgs_path = os.path.join(os.path.dirname(__file__), "dataset", 'imgs')
xml_path = os.path.join(os.path.dirname(__file__), 'dataset', 'xml')

# function to get the images and label data
def get_images_labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        print(imagePath)
        pil_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(pil_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].rpartition('.')[0].split('_')[-1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


def run():
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_labels(imgs_path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write(os.path.join(xml_path, 'trainner.yml'))

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

if __name__ == '__main__':
    run()