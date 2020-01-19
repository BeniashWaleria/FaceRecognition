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
cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

################################################################
# Init FaceAnalysis module by its default models


model = insightface.app.FaceAnalysis()

################################################################
# Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs

ctx_id = -1

################################################################
# Prepare the environment
# The nms threshold is set to 0.4 in this example.
#

model.prepare(ctx_id = ctx_id, nms=0.4)

################################################################
# Analyse faces in this image
#
faces = model.get(img)
results = pd.DataFrame(columns=['Face_Id','Age','Gender','Embedding shape','Bbox','Landmark'])
boxes=[]
landmarks=[]
for idx, face in enumerate(faces):
 # print("Face [%d]:"%idx)
  #print("\tage:%d"%(face.age))
  gender = 'Male'
  if face.gender==0:
    gender = 'Female'
 # print("\tgender:%s"%(gender))
 # print("\tembedding shape:%s"%face.embedding.shape)
  bbox=face.bbox.astype(np.int).flatten()
  boxes.append(bbox)
  landmark= face.landmark.astype(np.int).flatten()
  landmarks.append(landmark)
  print("\tbbox:{}".format(bbox))
  print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))

  #print("")
  results=results.append({'Face_Id':idx+1, 'Age': face.age,'Gender':gender,'Embedding shape':face.embedding.shape,
                          'Bbox': (face.bbox.astype(np.int)),'Landmark': (face.landmark.astype(np.int))},
                 ignore_index=True)
  results.to_csv(r'C:\Users\User\Desktop\Face_Recognition.csv',index=False,mode='w',sep=';')

for i,box in enumerate(boxes):
    (x,y,w,h)=(box[0],box[1],box[2],box[3])
    cropped = img[y: h, x: w]
    cv2.rectangle(img,(x,y),(w,h),(0,255,255),2)
    cv2.imwrite("cropped_face" + str(i) + ".jpg", cropped)
for i,lmk in enumerate(landmarks):
    cv2.circle(img,(lmk[0],lmk[1]),2,thickness=-1,color=(0,255,255))
    cv2.circle(img,(lmk[2],lmk[3]),2,thickness=-1,color=(0,255,255))
    cv2.circle(img,(lmk[4],lmk[5]),2,thickness=-1,color=(255,255,0))
    cv2.circle(img,(lmk[6],lmk[7]),2,thickness=-1,color=(255,255,0))
    cv2.circle(img,(lmk[8],lmk[9]),2,thickness=-1,color=(255,255,0))

#cv2.imshow("IMAGE",img)
#cv2.waitKey(0)