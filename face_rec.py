import insightface
import urllib
import urllib.request
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont


def url_to_image(url_):
    resp = urllib.request.urlopen(url_)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def crop_rect(image, bboxes):
    for i, box in enumerate(bboxes):
        (x, y, w, h) = (box[0], box[1], box[2], box[3])
        cropped = image[y:h, x:w]
        cv2.imwrite("cropped_face" + str(i) + ".jpg", cropped)
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 255), 2)
    return image


def landmark_def(image, landmarks_arr):
    i = 0
    while i < len(landmarks_arr):
        print(i)
        cv2.circle(img, (landmarks_arr[i], landmarks_arr[i + 1]), 2, thickness=-1, color=(0, 255, 255))
        i = i + 2
    return image

def draw_boxes(im, box_array, label_list):
    for idx, box in enumerate(box_array):
        x, y, w, h = box_array
        cv2.rectangle(im, (x, y), (w, h), (0, 255, 255), 2)
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), label_list, font=unicode_font, fill=font_color)
###################################################################################
# upload the image and convert BGR to RGB

url = 'https://i.pinimg.com/originals/5e/6f/c1/5e6fc1b854408c51b5655e0ed00d55f8.jpg'
img = url_to_image(url)

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

font_size = 36
width = 500
height = 100
back_ground_color = (0, 255, 255)
font_color = (0, 0, 0)
unicode_font = ImageFont.truetype("arial.ttf", font_size)

faces = model.get(img)
boxes = []
for idx, face in enumerate(faces):
    bbox = face.bbox.astype(np.int)
    boxes.append(bbox)
print(boxes)






