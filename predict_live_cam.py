'''
predict ID based on input image from haarcascade cropped frames, grabbed from face-detection.py
'''


import numpy as np
import cv2
from numpy import load
import os


# model selection
model_list = ['24', '120', 'CFD_1207']
model = model_list[2]

# model with 24 eigenfaces, but trained 25 images only
if model == '24':
    mn = load('./allFaces_mean.npy')
    eigenvecs = load("./24_reduce_eigvecs_col.npy")
    save_dir = '24_id_mean'

# model with 120 eigenfaces, but trained 25 images only
if model == '120':
    mn = load('./allFaces_mean.npy')
    eigenvecs = load("./24_reduce_eigvecs_col.npy")
    save_dir = '24_id_mean'

# model with 1206 eigenfaces, trained on 1207 images
if model == 'CFD_1207':
    mn = load('./CFDFaces_mean.npy')
    eigenvecs = load("./cfd__reduced_eigenvec.npy")
    save_dir = './cfd_id_mean'


# load mean face templates
ids = os.listdir(save_dir)
num_id = len(ids)
num_dim = load(os.path.join(save_dir, ids[0])).shape[0]
mean_faces = np.ndarray((num_dim, num_id))
index_key = []
for i, id in enumerate(ids):
    name = id.strip('.npy')
    index = i
    index_key.append((i,name))
    # mean = load(os.path.join(folder, id))
    mean = load(os.path.join(save_dir, id))
    mean_faces[:, i] = mean[:, 0]

# load face detector
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    i = 0
    for (x,y,w,h) in faces:
        # time.sleep(3)
        i+=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        input = cv2.resize(roi_gray, (112,112)).reshape(-1,1)
        input = input - mn
        test_eig = np.matmul(eigenvecs.T, input).flatten()
        # cv2.imshow('video', recog_input)
        ind = 0
        least = 1000000000
        for i, template in enumerate(mean_faces.T):
            # take eucl dist
            dist = np.linalg.norm(template - test_eig)
            if dist <= least:
                least = dist
                ind = i
        # print('recognised: {}'.format(index_key[ind][1]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, index_key[ind][1], (x,y), font, 0.5, (225,225,225), 1)  # test print to bound box
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()

