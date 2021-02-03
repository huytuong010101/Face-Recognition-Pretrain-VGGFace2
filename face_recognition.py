from keras_vggface.utils import preprocess_input
import numpy as np
from keras.preprocessing import image as processing_image
import os
import cv2
from mtcnn.mtcnn import MTCNN
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from keras_vggface.vggface import VGGFace

detector = MTCNN()
model = VGGFace(model='resnet50')

def recogition(x, y, image_array=None, path=None):
    '''
    Detect and recognition the face in image
    '''
    # Load image
    if path:
        img = processing_image.load_img(path)
        img = processing_image.img_to_array(img)
    else:
        img = image_array

    # Detect face in image
    faces = detector.detect_faces(img)
    if not faces:
        #print("Cannot find any face in image")
        return None, 1, 0, 0, 0, 0
    
    # Crop face - preprocessing
    x_pos, y_pos, w, h = 0, 0, 0, 0
    for face in faces:
        if face["box"][2]*face["box"][3] > w*h:
            x_pos, y_pos, w, h = face["box"]
    img = img[y_pos:y_pos + h, x_pos:x_pos + w]
    img = cv2.resize(img, (224, 224))
    img = img.astype('float64')
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2) # or version=2
    
    # Encode face
    preds = model.predict(img)
    result = euclidean_distances(preds, x)
    index = np.argmin(result[0])
    return y[index], 1-result[0][index], x_pos, y_pos, w, h