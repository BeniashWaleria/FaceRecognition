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


url = 'https://i.pinimg.com/originals/5e/6f/c1/5e6fc1b854408c51b5655e0ed00d55f8.jpg'
img = url_to_image(url)
cv2.imshow('image',img)
################################################################
# Init FaceAnalysis module by its default models


model = insightface.app.FaceAnalysis()

################################################################
# Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs
#

ctx_id = -1


################################################################
# Prepare the enviorment
# The nms threshold is set to 0.4 in this example.
#

model.prepare(ctx_id = ctx_id, nms=0.4)

################################################################
# Analysis faces in this image
#
faces = model.get(img)
results = pd.DataFrame(columns=['Face_Id','Age','Gender','Embedding shape','Bbox','Landmark'])
for idx, face in enumerate(faces):
  print("Face [%d]:"%idx)
  print("\tage:%d"%(face.age))
  gender = 'Male'
  if face.gender==0:
    gender = 'Female'
  print("\tgender:%s"%(gender))
  print("\tembedding shape:%s"%face.embedding.shape)
  print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
  print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
  print("")
  results=results.append({'Face_Id':idx+1, 'Age': face.age,'Gender':gender,'Embedding shape':face.embedding.shape,
                          'Bbox': str(face.bbox.astype(np.int)),'Landmark': str(face.landmark.astype(np.int))},
                 ignore_index=True)

  results.to_csv(r'C:\Users\User\Desktop\Face_Recognition.csv',index=False,mode='w',sep=';')