import numpy as np
from numpy import save
from numpy import load
from PIL import Image
import cv2
import os
import time
from matplotlib import pyplot as plt




def main():
    # load dataset
    list_dir = './cfd_crop.lst'
    imgs_dir = open(list_dir, 'r')
    imgs = imgs_dir.read()
    img_list = imgs.split(',')      # list of img directories

    # get data counts and data pixels
    img_vec = cv2.imread(img_list[0], 0).reshape(-1,1)
    dim = len(img_vec)  # num_pixels
    N = len(img_list)   # num_samples

    # create empty data array
    X = np.ndarray((dim, N))

    # create a matrix of all image column vectors
    for i, img_dir in enumerate(img_list):
        img_vec = cv2.imread(img_dir, 0).reshape(-1,1)
        X[:, i] = img_vec[:, 0]

    # sanity check:   should see (num_pixels, num_samples)
    print(X.shape)

    # train

    # mean adjust all dimensions (pixels)
    mn = X.mean(axis=1).reshape(-1, 1)
    print(mn.shape)
    # save('CFDFaces_mean.npy', mn)
    print("mean matrix shape:", mn.shape)    # check : (num_pixels, one)
    X = X - mn    # X = RowDataAdjust in lec slides = mean adjusted pixel row matrix

    # covariance matrix
    _C = np.matmul(X.T, X) / N  # X(X.T)

    print("Covar mat shape:", _C.shape)      # check

    print("Calculating eigenvectors...")
    _w, _v = np.linalg.eigh(_C)  # (X)(X.T)[(X)(_v)] = _w[(X)(_v)]
    print("\tdone")

    # transform _v back to v and sort according to eigenvalue large to small
    v = np.matmul(X, _v)
    eig_pairs = [(_w[index], v[:, index] / np.linalg.norm(v[:, index])) for index in range(len(_w))]
    eig_pairs.sort(reverse=True)
    # find dimensions that has meaning full data ---> eigval > 1
    count = 0
    for pair in eig_pairs:
        if pair[0] >= 1:
            count+=1
    print("useful dimensions:", count)

    # dimension reduction
    v_reduc = np.ndarray((dim, count))
    for i in range(count):
        v_reduc[:, i] =  eig_pairs[i][1]
    # v_reduc = v[:, 0:count]
    save('fast_cfd_reduced_eigenvec.npy', v_reduc)  # save in binary for faster access
    print(v_reduc.shape)

    # get faces in reduced eigen space
    # final_data = np.matmul(v_reduc.T, X)
    # print(final_data.shape)  # (num_dim_left, num_samples)


    # check with slow method results  -- pass
    # old = load("cfd__reduced_eigenvec.npy")
    # print(old)


    return






if __name__ == "__main__":
    main()
