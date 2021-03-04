import numpy as np
from numpy import savetxt
from PIL import Image
import cv2
import torch
import os
import time
from matplotlib import pyplot as plt

device = torch.device("cpu")



def main():
    # load dataset
    list_dir = './frontal_face_aligned.lst'
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
    # print(mn)
    print("mean matrix shape:", mn.shape)    # check : (num_pixels, one)
    X = X - mn    # X = RowDataAdjust in lec slides = mean adjusted pixel row matrix

    # covariance matrix
    C = np.matmul(X, X.T) / N

    # why is np.cov(X)    and    formula (1/N)(X)(X.T) differing by 2x ?   2 x np.cov(x) = formula
    # _C = np.cov(X) / N  # numpy.cov expects an num_dimensions x num_samples array, X is (num_dim, num_samples)
    # print(C)
    # print(_C)
    print("Covar mat shape:", C.shape)      # check

    # Covar matrix symmetric, (k,k), num_eigvals eigvects = k
    # w = eigen val row vector  [w1, w2, w3]      v = eigen vev col vector matrix   [v1 v2 v3 ..]
    # np.linalg.eigh()  # returns w, v in sorted low to high.     w[i] -> v[ : , i]

    # numpy
    # start = time.time()
    w, v = np.linalg.eigh(C)  # alternatively
    # elapsed = time.time() - start
    # print("np", elapsed)

    # change to descending order
    w = np.flip(w)
    v = np.flip(v, axis=1)
    # print(w)

    # save eig col vectors, and eig val array to CSV
    # savetxt('eigvec_col.csv', v, delimiter=',')
    # savetxt('eigval_arr.csv', w, delimiter=',')


    # find dimensions that has meaning full data ---> eigval > 1
    count = 0
    for val in w:
        if val >= 1:
            count+=1
    print("useful dimensions:", count)

    # dimension reduction
    v_reduc = v[:, 0:count]
    # savetxt('reduce_eigvecs_col.csv', v, delimiter=',')
    # print(v_reduc.shape)

    # get faces in reduced eigen space
    final_data = np.matmul(v_reduc.T, X)
    print(final_data.shape)  # (num_dim_left, num_samples)

    # plot data
    # plt.scatter(final_data[0, :], final_data[1, :], final_data[2, :])   # plot top 2 principle components
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_data[0, 0:4], final_data[1, 0:4], final_data[2, 0:4], marker="o")
    ax.scatter(final_data[0, 5:9], final_data[1, 5:9], final_data[2, 5:9], marker="^")
    ax.scatter(final_data[0, 10:14], final_data[1, 10:14], final_data[2, 10:14], marker="1")
    ax.scatter(final_data[0, 15:19], final_data[1, 15:19], final_data[2, 15:19], marker="s")
    ax.scatter(final_data[0, 20:24], final_data[1, 20:24], final_data[2, 20:24], marker="_")
    plt.show()

    #

    return


def make_pixel_row_mat(vec_faces_list):
    '''
    IE [pixel_1 of all faces,
        pixel_2 of all faces,
        .                               ie shape: (num_dim, num_sample)     in this case is (10000, 3)
        .
        .
        pixel_K of all faces]
    '''
    for i, face in enumerate(vec_faces_list):
        if i == 0:
            X = vec_faces_list[i]
        else:
            X = np.append(X, vec_faces_list[i], axis=1)
    return X


# zero mean adjustment
def mean_adjust(pixel):
    avg = np.average(pixel)
    return pixel - avg




'''
X = np.append(x,y, axis=1)
X = np.append(X,z, axis=1)
print(X)
print(X[0, 1])

X_T = X.T
print(X_T)

N = len(x)

# Covariance matrix of every pixel's covariances
C = (1/N) * np.matmul(X, X_T)
# print(C.shape)
'''



if __name__ == "__main__":
    main()
