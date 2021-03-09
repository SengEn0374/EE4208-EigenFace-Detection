'''
Calculate each person's mean face in eigen space using pretrained principal eigen vectors
Save each classes mean for comparison (prediction) later
'''
import numpy as np
from numpy import save
from numpy import load
import os
import cv2

# save dir
save_dir = './cfd_id_mean'
if os.path.exists(save_dir) != True:
    os.makedirs(save_dir)

# load eigen vectors
eigenvecs = load("./cfd__reduced_eigenvec.npy")

# load mean dataset
mn = load('./CFDFaces_mean.npy')

data_dir = './frontal_face_aligned'
IDs = os.listdir(data_dir)
# step through each class (ID)
for ID in IDs:
    ID_dir = os.path.join(data_dir, ID)
    imgs = os.listdir(ID_dir)
    flg = 1
    for img in imgs:
        im_dir = os.path.join(ID_dir, img)
        if flg == 1:
            im_vec = cv2.imread(im_dir, 0).reshape(-1,1)
            im_vec = im_vec - mn
            eig_face_sum = np.matmul(eigenvecs.T, im_vec)
            # print(eig_face.shape)
            flg = 0
        else:
            im_vec = cv2.imread(im_dir, 0).reshape(-1,1)
            im_vec = im_vec - mn
            eig_face = np.matmul(eigenvecs.T, im_vec)
            eig_face_sum = eig_face_sum + eig_face
    avg = eig_face_sum/len(imgs)
    # print(avg.shape)
    save(os.path.join(save_dir,"{}.npy".format(ID)), avg)
    # save mean of each person into a vector shape = (num_reduced_vecs, 1) in .npy file to compare one by one later



# test ###############################################################################################################
# folder = './24_id_mean'
# ids = os.listdir(folder)
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

# print(index_key[1][0], index_key[1][1])
# print(mean_faces)


input_img = '../test64.jpg'
testimg = cv2.imread(input_img, 0).reshape(-1,1)
# mean adjust input img
mn = load('./CFDFaces_mean.npy')
testimg = testimg - mn
test_eig = np.matmul(eigenvecs.T, testimg).flatten()

largest = 0
ind = 0
least = 1000000000
for i, template in enumerate(mean_faces.T):
    '''
    dot_prod = np.dot(template, test_eig)
    if dot_prod >= largest:
        largest = dot_prod
        ind = i
    print('%s: %d, dot_prod: %f' % (index_key[i][1], i, dot_prod))
    '''
    # take eucl dist
    dist = np.linalg.norm(template - test_eig)
    if dist <= least:
        least = dist
        ind = i
    print('%s: %d, dot_prod: %f' % (index_key[i][1], i, dist))
print('recognised: {}'.format(index_key[ind][1]))



# find threshold for rejection

