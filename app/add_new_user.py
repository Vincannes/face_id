import os
import cv2
import dlib
import pickle
import numpy as np
import pickle
import MySQLdb

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("datas/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("datas/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("datas/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
cascadeClassifierPath = 'datas/haarcascade_frontalface_alt.xml' # Chemin du Classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
print('[INFO] Importing pretrained model..')

cap = cv2.VideoCapture(0) # On récupère la vidéo
recognizer.read('trainner.yml')

data = dict()
known_names = []
known_faces = []

curr_id = 1
label = "vincent"

while(True):
    rect, frame = cap.read()

    # Conversion GRay
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Get Array
    image_array = np.array(grayImage)
    # Get Visage Area
    detected_faces = cascadeClassifier.detectMultiScale(grayImage,  
                                                        scaleFactor=1.1, 
                                                        minNeighbors=10, 
                                                        minSize=(20, 20)
                                                        ) # Détection

    if not label in data:	
        data[label] = curr_id
    user_id = data[label]

    # Get Position Square Visage
    for(x,y, width, height) in detected_faces:
        roi = image_array[y:y+height, x:x+width]
        known_names.append(roi)
        known_faces.append(user_id)
        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,255,0), 3)
        

    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord('q'):
        break


with open("labels.picle", 'wb') as f:
    pickle.dump(user_id, f)

recognizer.train(known_names, np.array(known_faces))
recognizer.save('datas.yml')



def save_to_DB(face_encoding):
    ## Pickle the list into a string
    face_pickled_data = pickle.dumps(face_encoding)

    ## Connect to the database
    connection = MySQLdb.connect('localhost','user','pass','myDatabase')

    ## Create a cursor for interacting
    cursor = connection.cursor()

    ## Add the information to the database table
    cursor.execute("""INSERT INTO faces VALUES (NULL, 'faceName', %s)""", (face_pickled_data, ))

    ## Select what we just added
    cursor.execute("""SELECT data FROM faces WHERE name = 'faceName'""")

    ## Dump the results to a string
    rows = cursor.fetchall()

    ## Get the results
    for each in rows:
        ## The result is also in a tuple
        for face_stored_pickled_data in each:
            face_data = pickle.loads(face_stored_pickled_data)

