from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
from keras.preprocessing import image as processing_image
import os
import cv2
from mtcnn.mtcnn import MTCNN
import pickle
import shutil


detector = MTCNN()
model = VGGFace(model='resnet50')

def crop_faces(FACES_DIR="faces", SAVED_FACES_LIST="faces.pkl"):
    '''
    This function use to crop face in raw image in FACES_DIR/raw,save list crop image to SAVED_FACES_PKL
    return dictionary contain all path to face image after crop
    '''
    # Delete old faces
    if os.path.exists(FACES_DIR + "/crop"):
        shutil.rmtree(FACES_DIR + "/crop")
        print("Delete old faces")

    # Create crop folder -  List all face name
    os.mkdir(FACES_DIR + "/crop")
    face_names = os.listdir(FACES_DIR + "/raw")
    face_images = {}

    # Loop all iamges and crop the face
    
    for face_name in face_names:

        # list image of face_name - create new folder to save crop image
        images = os.listdir(FACES_DIR + "/raw/" + face_name)
        os.mkdir(FACES_DIR + "/crop/" + face_name)
        face_images[face_name] = images
        for image in images:

            # Load and setect face in image
            image_path = FACES_DIR + "/raw/" + face_name + "/" + image
            img = processing_image.load_img(image_path)
            img = processing_image.img_to_array(img)
            faces = detector.detect_faces(img)
            if not faces:
                print("Cannot detect face in", image_path)
                continue

            # Find the biggest face
            x, y, w, h = 0, 0, 0, 0
            for face in faces:
                if face["box"][2]*face["box"][3] > w*h:
                    x, y, w, h = face["box"]
            
            # Crop face
            img = img[y:y+h, x:x+w]

            # Save the crop image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            path = FACES_DIR + "/crop/" + face_name + "/" + image
            cv2.imwrite(path, img)
            print("Saved image to", path)

    # Save list of faces
    with open(SAVED_FACES_LIST, "wb") as f:
        pickle.dump(face_images, f)
        print("Saved face list to", SAVED_FACES_LIST)
    return face_images

def encode_faces(FACES_DIR="faces", SAVED_FACES_LIST="faces.pkl", SAVED_FACES_ENCODE="encode_face.pkl"):
    x, y = [], []

    # Load list of faces
    with open(SAVED_FACES_LIST, "rb") as f:
        face_images = pickle.load(f)
        print("Load face list from", SAVED_FACES_LIST)
    for face in face_images:
        for image in face_images[face]:
            path = FACES_DIR + "/crop/" + face + "/" + image
            img = processing_image.load_img(path, target_size=(224, 224))
            img = processing_image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img, version=2)
            preds = model.predict(img)
            x.append(preds[0])
            y.append(face)
            print("Encoded", path)
    x = np.array(x)
    y = np.array(y)
    with open(SAVED_FACES_ENCODE, "wb") as f:
        pickle.dump({"x": x, "y": y}, f)
        print("Sace encoded face to", SAVED_FACES_ENCODE)
    return x, y

if __name__ == "__main__":
    crop_faces()
    encode_faces()