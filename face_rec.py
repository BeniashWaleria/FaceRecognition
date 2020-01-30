import insightface
import requests
import os
from io import BytesIO
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
from numpy.linalg import norm


def url_to_image(url_):
    if re.match('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*(), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url_):
        resp = requests.get(url_)
        image_ = Image.open(BytesIO(resp.content))
    else:
        image_ = Image.open(url_)
    return image_


def get_label(filename):
    basename = os.path.basename(filename)
    name = '.'.join(basename.split('.')[:-1])
    return name


def compute_sim(db_matr, test_vect=None):
    distances = norm(db_matr - test_vect, 2, 1)
    angles = np.arccos(db_matr @ test_vect.T / (norm(db_matr, 2, 1) * norm(test_vect, 2))) * 180 / np.pi
    return angles, distances


def get_embeddings(image_, model_):
    faces_ = model_.get(np.array(image_))
    embeddings_ = []
    for i in faces_:
        embeddings_.append(i.embedding)
    return np.asarray(embeddings_)


def draw_boxes(im, box_array, label_list):
    for label_, box in zip(label_list, box_array):
        x, y, w, h = box
        draw = ImageDraw.Draw(im)
        draw.rectangle((x, y, w, h), outline='yellow', width=2)
        draw.text((x, h), label_, font=unicode_font, fill=font_color,back_ground_color='white')
    im.show()

###################################################################################
# upload the image

url = r'https://i.pinimg.com/originals/5e/6f/c1/5e6fc1b854408c51b5655e0ed00d55f8.jpg'


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
# Make inscriptions for photos
img = url_to_image(url)
font_size = 14
width = 500
height = 100
back_ground_color = (0, 0, 255)
font_color = (0, 250, 250)
unicode_font = ImageFont.truetype("arial.ttf", font_size)
directory = 'Samples'

labels = []
embeddings_arr = []
for filename_ in os.listdir(directory):
    image = url_to_image(directory + "\\" + filename_)
    embeddings_arr.append(get_embeddings(np.asarray(image), model))
    labels.append(get_label(filename_))

embeddings_arr = np.asarray(embeddings_arr)
embeddings_main = get_embeddings(np.asarray(img), model)
threshold = 75
predicted_labs = []
######################################################################
# perform face  recognition

print(labels)
for label, emb in zip(labels, embeddings_arr):
    angles_, distances_ = compute_sim(emb, embeddings_main)
    min_angle = angles_.min()
    index = angles_.argmin()
    if min_angle < threshold:
        predicted_labs.append(labels[index])
    else:
        print(min_angle)
print(predicted_labs)
boxes = []
faces = model.get(np.asarray(img))
for idx, face in enumerate(faces):
    bbox = face.bbox.astype(np.int)
    boxes.append(bbox)
boxes = np.asarray(boxes)
draw_boxes(img, boxes, predicted_labs)
img.show()