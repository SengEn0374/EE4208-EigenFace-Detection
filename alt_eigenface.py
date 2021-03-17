'''
logic: C = k * matmul(X, X.T), ignoring constant k,
C = matmul(X, X.t) = (10000, 10000) array, if num pixel =100,  C is HUGE for eigen decomposition cal.
then, CV = wV, where
    (XX.T)V = wV   ---- (*)
HOWEVER, if we find cov of X.TX instead
    (XT.X)(Vi) = l(Vi)
    X(X.TX)Vi  = l(XVi)
    (XX.T)XVi  = l(XVi)  ---- (1)

        comparing  (1) and (*)

    (XX.T)V = wV   --- (*)

        we can see eigen vectors, V = XVi
        and eigen values l = w

X.TX is size (num sample, num samples), much smaller than (10000, 10000), much faster, less exp eig vec and val computes
'''
import numpy as np
from numpy import save
from numpy import load
import cv2
import os


# load data
data_dir = './cfd_crop'
imgs = os.listdir(data_dir)

# get data dimensions
img_dir = os.path.join(data_dir, imgs[0])
img = cv2.imread(img_dir, 0)
num_img = len(imgs)
num_dim = img.shape[0] * img.shape[1]

# create image tensor
image_tensor = np.ndarray((num_img, num_dim))

# load flattened images to img tensor
for i, img in enumerate(imgs):
    img_dir = os.path.join(data_dir, img)
    gray = cv2.imread(img_dir, 0).flatten()  # read b/w, flatten
    image_tensor[i, :] = gray

# sanity check
print('image tensor shape:', image_tensor.shape)

# get mean face
avg = image_tensor.mean(axis=0)
print(avg)

# zero-mean the faces in tensor
X_T = image_tensor - avg

# print(avg)
# print(image_tensor)
# print(X)

# find covariance matrix -- between each faces .. instead of between each pixels
C = np.matmul(X_T, X_T.T) / num_img
print('covariance matrix shape:', C.shape)

# find eigenvectors, self covariance matrix is always symmetric, use 'eigh' for faster compute
eigvals, eigvecs = np.linalg.eigh(C)   # output in ascending eigen value
print("number of eigen vectors:", eigvecs.shape[1])
print("eigenvals matrix shape:", eigvals.shape)
# re-order in largest to smallest eigen values
eigvals_dsc = np.flip(eigvals, axis=0)
eigvecs_dsc = np.flip(eigvecs, axis=1)
# print(eigvals)  # ensure order is correct

# dimension reduction
count=0
for eigval in eigvals_dsc:
    if eigval >= 1.0:
       count+=1
print("number of useful dimensions:", count)
reduced_eigvecs = eigvecs_dsc[:, 0:count]

# convert back to X's eigenvectors (reduced)
v = np.matmul(X_T.T, reduced_eigvecs)
print("eigenvectors shape:", v.shape)
print(v)

v_loaded = load("cfd__reduced_eigenvec.npy")
print(v_loaded.shape)


# save
# save('CFD_alt_eigenvecs.npy', v)