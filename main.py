import face_recognition
import processing_face_data
import pickle
import time
from hikvisionapi import Client
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import processing_face_data

# Constant
CAM_IP = 'http://192.168.1.29'
CAM_USER = 'admin'
CAM_PASS = 'cam12345'


# Encoding images -----------------------------------------------------
print("Start processing faces...")
processing_face_data.crop_faces()
print("Start encoding faces...")
processing_face_data.encode_faces()
print("Prepare data done")
# Loading database ----------------------------------------------------

with open("encode_face.pkl", "rb") as f:
    data = pickle.load(f)
    x, y = data["x"], data["y"]
print("Loaded database")
# Start stream
cam = Client(CAM_IP, CAM_USER, CAM_PASS , timeout=30)
while(True):
    response = cam.Streaming.channels[102].picture(method='get', type='opaque_data')

    raw = response.raw.read()
    stream = BytesIO(raw)
    image = Image.open(stream).convert("RGB")
    open_cv_image = np.array(image)
    name, acc, x_pos, y_pos, w, h = face_recognition.recogition(x, y, open_cv_image)
    #name, acc = None, 1
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    # Put name
    if name and acc >= 0.9:
        open_cv_image = cv2.putText(
            open_cv_image, 
            "name: " + name + " | " + str(acc), 
            (20, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 225), 
            2,
            cv2.LINE_AA)  
        open_cv_image = cv2.rectangle(open_cv_image, (x_pos, y_pos), (x_pos + w, y_pos + h), (255, 0, 0), 2) 
    # Display the resulting frame
    cv2.imshow('face-stream', open_cv_image)
    print(x_pos, y_pos, w, h)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()

