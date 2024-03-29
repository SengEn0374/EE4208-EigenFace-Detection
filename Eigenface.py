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
    # list_dir = './56x56.lst'
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
    '''
    # create test vectorised faces
    x = np.linspace(0, 10000, 100)
    x = x.reshape(-1, 1)
    # print(x)
    y = np.linspace(1, 1000, 100)
    y = y.reshape(-1, 1)
    # print(y)
    z = np.linspace(5,50123, 100)
    z = z.reshape(-1, 1)

    faces_list = [x,y,z]
    '''

    # mean adjust all dimensions (pixels)
    mn = X.mean(axis=1).reshape(-1, 1)
    print(mn.shape)
    save('CFDFaces_mean.npy', mn)
    print("mean matrix shape:", mn.shape)    # check : (num_pixels, one)
    X = X - mn    # X = RowDataAdjust in lec slides = mean adjusted pixel row matrix

    # covariance matrix
    C = np.matmul(X, X.T) / N  # X(X.T)

    print("Covar mat shape:", C.shape)      # check

    # Covar matrix symmetric, (k,k), num_eigvals eigvects = k
    # w = eigen val row vector  [w1, w2, w3]      v = eigen vev col vector matrix   [v1 v2 v3 ..]
    # np.linalg.eigh()  # returns w, v in sorted low to high.     w[i] -> v[ : , i]

    print("Calculating eigenvectors...")
    w, v = np.linalg.eigh(C)  # bottle neck here

    # change to descending order
    w = np.flip(w)
    v = np.flip(v, axis=1)

    # find dimensions that has meaning full data ---> eigval > 1
    count = 0
    for val in w:
        if val >= 1:
            count+=1
    print("useful dimensions:", count)

    # dimension reduction
    v_reduc = v[:, 0:count]
    save('150_cfd__reduced_eigenvec.npy', v_reduc)  # save in binary for faster access
    # print(v_reduc.shape)

    # get faces in reduced eigen space
    final_data = np.matmul(v_reduc.T, X)
    print(final_data.shape)  # (num_dim_left, num_samples)

    # plot data for visualising training
    # plt.scatter(final_data[0, :], final_data[1, :], final_data[2, :])   # plot top 2 principle components
    # plt.show()
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_data[0, 0:4], final_data[1, 0:4], final_data[2, 0:4], marker="o")
    ax.scatter(final_data[0, 5:9], final_data[1, 5:9], final_data[2, 5:9], marker="^")
    ax.scatter(final_data[0, 10:14], final_data[1, 10:14], final_data[2, 10:14], marker="1")
    ax.scatter(final_data[0, 15:19], final_data[1, 15:19], final_data[2, 15:19], marker="s")
    ax.scatter(final_data[0, 20:24], final_data[1, 20:24], final_data[2, 20:24], marker="_")
    plt.show()

    # test code, load reduced eigen vecs.npy data
    eigenvecs = load("cfd__reduced_eigenvec.npy")
    # check loaded is same as original saved
    print(eigenvecs)
    print(v_reduc)
    '''
    return






if __name__ == "__main__":
    main()
