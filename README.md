# Face-Recognition-Pretrain-VGGFace2
## How to run?
### Setup environment
- Recommend Python 3
- cd to project directory -> python -m pip install -r requirements.txt
### Prepare data:
- In the faces/raw directory, create some new folder whose name is name of person
- Put in that folder the images which contain face of corresponding person
### Run code
- Crop face and encode face: python processing_face_data.py
- Change the constant in main.py, then: python main.py
