import os
import cv2
import dlib
import pickle
import json
import numpy as np
# from . import training_model
from pprint import pprint

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
face_detector = dlib.get_frontal_face_detector()
pose_predictor_68_point = dlib.shape_predictor("datas/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("datas/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("datas/dlib_face_recognition_resnet_model_v1.dat")
cascadeClassifierPath = 'datas/haarcascade_frontalface_alt.xml' # Chemin du Classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
print('[INFO] Importing pretrained model..')

cap = cv2.VideoCapture(0) # On récupère la vidéo
# recognizer.read('trainner.yml')

face_sample = []
face_id = []

with open("dataset/labels.json", 'rb') as f:
    dataset_ids = json.load(f)

img_path = os.path.join(os.path.dirname(__file__), 'dataset', 'imgs')
xml_path = os.path.join(os.path.dirname(__file__), 'dataset', 'xml')

curr_id = 1
name = "vincent"
surname = "trolard"

find_user = False
# initialize the list of known encodings and known names

while True:
    rect, frame = cap.read()

    # Conversion GRay
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get Array
    image_array = np.array(grayImage, 'uint8')
    # Get Visage Area
    detected_faces = cascadeClassifier.detectMultiScale(grayImage,
                                                        scaleFactor=1.1,
                                                        minNeighbors=10,
                                                        minSize=(20, 20)
                                                        )

    data = {"id": curr_id, "name": name, "surname": surname}
    user_ids = [user['id'] for user in dataset_ids]
    if curr_id not in user_ids:
        dataset_ids.append(data)

    # Get Position Square Visage
    for(x, y, width, height) in detected_faces:
        face_area = image_array[y:y+height, x:x+width]
        face_sample.append(face_area)
        face_id.append(curr_id)
        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(img_path, 'user_{}.jpeg').format(curr_id), face_area)
        find_user = True

    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord('q') or find_user:
        break

# try:
#     training_model.run()
# except:
#     print('Image cannot been generate !')

with open("dataset/labels.json", 'w') as f:
    json.dump(dataset_ids, f, indent=4)

print('END')

