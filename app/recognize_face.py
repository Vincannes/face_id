import os
import cv2
import dlib
import json
import pickle
import face_recognition
import numpy as np

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("datas/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("datas/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("datas/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
cascadeClassifierPath = 'datas/haarcascade_frontalface_alt.xml'  # Chemin du Classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
print('[INFO] Importing pretrained model..')


# https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
cap = cv2.VideoCapture(0)  # On récupère la vidéo
recognizer.read('dataset/xml/trainner.yml')

with open("dataset/labels.json", 'r') as f:
    datas = json.load(f)

while True:
    rect, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion N/B
    detected_faces = cascadeClassifier.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=10,
                                                        minSize=(20, 20))  # Détection

    for (x, y, width, height) in detected_faces:
        face_resize = grayImage[y:y + height, x:x + width]
        id_, conf = recognizer.predict(face_resize)
        print(conf)
        if conf < 60:
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            for user in datas:
                if id_ == user['id']:
                    cv2.putText(frame, str(user['id']), (x + width, y), font, 1, color, stroke, cv2.LINE_AA)
                    cv2.putText(frame, user['name'], (x + width, y-30), font, 1, color, stroke, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)  # Dessin d'un rectangle

    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
