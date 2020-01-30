import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import pandas as pd


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def crop_rect(image, boxes):
    (x, y, w, h) = boxes
 #   cropped = image[y:h, x:w]
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 255), 2)
    return image


def landmark_def(image, landmarks_arr):
    i = 0
    while i < len(landmarks_arr):
        print(i)
        cv2.circle(img, (landmarks_arr[i], landmarks_arr[i + 1]), 2, thickness=-1, color=(0, 255, 255))
        i = i + 2
    return image


###################################################################################
# upload the image and convert BGR to RGB
url = 'https://i.pinimg.com/originals/5e/6f/c1/5e6fc1b854408c51b5655e0ed00d55f8.jpg'
img = url_to_image(url)
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

################################################################
# Init FaceAnalysis module by its default models

model = insightface.app.FaceAnalysis()

################################################################
# Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs

ctx_id = -1

################################################################
# Prepare the environment
# The nms threshold is set to 0.4 in this example.

model.prepare(ctx_id=ctx_id, nms=0.4)

################################################################
# Analyse faces in this image

results = pd.DataFrame(columns=['Face_Id', 'Age', 'Gender', 'Embedding shape', 'Bbox', 'Landmark'])
boxes = []
landmarks = []

faces = model.get(img)
for idx, face in enumerate(faces):
    gender = 'Male'
    if face.gender == 0:
        gender = 'Female'
    bbox = face.bbox.astype(np.int).flatten()
    boxes.append(bbox)
    landmark = face.landmark.astype(np.int).flatten()
    landmarks.append(landmark)
    age = face.age
    shape = face.embedding.shape

    ################################################################
    # save results of analysis to csv file

    results = results.append(
        {'Face_Id': idx + 1, 'Age': age, 'Gender': gender, 'Embedding shape': shape, 'Bbox': bbox,
         'Landmark': landmark},
        ignore_index=True)
    results.to_csv("RESULTS.csv", index=False, mode='w', sep=';')

################################################################
# crop faces and save them,add rectangles

crop_rect(img, boxes[0])

################################################################
# add landmarks

for array in landmarks:
    landmark_def(img, array)
cv2.imshow("IMAGE", img)
print(type(img))
cv2.waitKey(0)

################################################################
