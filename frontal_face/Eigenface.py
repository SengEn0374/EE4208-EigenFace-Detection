import numpy as np
import os

def main():
    # img = place holder for input image
    img = np.ones((100,100))
    k = img.shape[1] * img.shape[0]  # k = m*n, image size = (m,n)
    print("num pixels:", k)


    # create test vectorised faces
    x = np.linspace(0, 10000, 10000)
    x = x.reshape(-1, 1)
    # print(x)
    y = np.linspace(1, 1000, 10000)
    y = y.reshape(-1, 1)
    # print(y)
    z = np.linspace(5,50123, 10000)
    z = z.reshape(-1, 1)

    faces_list = [x,y,z]

    # number of samples (images) = N
    N = len(faces_list)
    # N = len(os.list.dir("dataset_path"))  # use this in real scenario

    # concat all face col vectors into a matrix
    X = make_pixel_row_mat(faces_list)
    print("pixel row matrix shape:", X.shape)  # check : (num_pixels, num_face_samples)

    # mean adjust all pixels
    mean = X.sum(axis=1) / N
    mean = mean.reshape(-1,1)
    print("mean matrix shape:", mean.shape)    # check : (num_pixels, one)
    # print(mean)
    X = X - mean    # X = RowDataAdjust in lec slides = mean adjusted pixel row matrix
    # print(X)

    # covariance matrix
    C = np.matmul(X, X.T) / N
    print("Covar mat shape:", C.shape)      # check

    # stopped at calculating eigvec and eigval manually
    return


def make_pixel_row_mat(vec_faces_list):
    '''
    IE [pixel_1 of all faces,
        pixel_2 of all faces,
        .
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
