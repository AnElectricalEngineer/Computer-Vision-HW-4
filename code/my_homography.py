import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt


# Add imports if needed:

# end imports

# Add extra functions here:

def projectPoints(im1, im2, N):
    p1, p2 = getPoints(im1, im2, N)
    H = computeH(p1, p2)
    fig1.suptitle('Choose an arbitrary point from the left image')
    plt.draw()
    arbitraryPoint = np.asarray(fig1.ginput(n=1, timeout=-1)[0])
    axes1[0].scatter(arbitraryPoint[0], arbitraryPoint[1], marker='x',
                     color='green')
    plt.draw()
    arbitraryPoint = np.array([arbitraryPoint[0], arbitraryPoint[1], 1])
    projectedPoint = np.matmul(H, arbitraryPoint)
    projectedPoint = projectedPoint / projectedPoint[-1]
    axes1[1].scatter(projectedPoint[0], projectedPoint[1], marker='o',
                     color='green')
    fig1.suptitle('The green circle on the right image is the corresponding '
                  'point')
    plt.draw()
    plt.savefig("../output/Section 1.2 - Transformation Correctness "
                "Example.png")
# Extra functions end

# HW functions:
fig1, axes1 = plt.subplots(1, 2)

def getPoints(im1, im2, N):
    axes1[0].imshow(im1)
    axes1[1].imshow(im2)
    axes1[0].set_xticks([])
    axes1[0].set_yticks([])
    axes1[1].set_xticks([])
    axes1[1].set_yticks([])
    fig1.suptitle(f'Click on {N} corresponding points on each image, '
                  f'in alternating order, starting from the left image')

    p1 = []
    p2 = []

    while len(p1) + len(p2) < 2 * N:
        p1.append(np.asarray(fig1.ginput(n=1, timeout=-1)[0]))
        axes1[0].scatter(p1[-1][0], p1[-1][1], marker='x', color='red')
        plt.draw()
        p2.append(np.asarray(fig1.ginput(n=1, timeout=-1)[0]))
        axes1[1].scatter(p2[-1][0], p2[-1][1], marker='x', color='blue')
        plt.draw()
    p1 = np.array(p1)
    p2 = np.array(p2)
    return p1.T, p2.T


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    N = p1.shape[1]
    A = np.zeros((2*N, 9))

    p1 = np.vstack([p1, np.ones(N)])
    A[range(0, 2*N, 2), 0:3] = p1.T
    A[range(1, 2*N, 2), 3:6] = p1.T
    A[range(0, 2*N, 2), 6:8] = -p1[0:2].T * np.array([p2[0], p2[0]]).T
    A[range(1, 2*N, 2), 6:8] = -p1[0:2].T * np.array([p2[1], p2[1]]).T
    A[range(0, 2 * N, 2), 8] = -p2[0].T
    A[range(1, 2 * N, 2), 8] = -p2[1].T

    _lambda, V = np.linalg.eig(np.matmul(A.T, A))
    indexOfSmallestLambda = np.abs(_lambda).argmin()
    H2to1 = V[:, indexOfSmallestLambda].reshape(3, 3)

    return H2to1


def warpH(im1, H, out_size):
    """
    Your code here
    """
    return warp_im1


def imageStitching(img1, wrap_img2):
    """
    Your code here
    """
    return panoImg


def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH


def getPoints_SIFT(im1, im2):
    """
    Your code here
    """
    return p1, p2


if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread('data/incline_R.png')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # p1, p2 = getPoints(im1, im2, 4)
    # computeH(p1, p1)

    projectPoints(im1, im2, 6)
