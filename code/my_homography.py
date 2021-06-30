import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy.ma as ma


# Add imports if needed:

# end imports

# Add extra functions here:

def projectPoints(im1, im2, N):
    p1, p2 = getPoints(im1, im2, N)
    H2to1 = computeH(p1, p2)

    fig1.suptitle('Choose an arbitrary point from the left image')
    plt.draw()
    arbitrary_point = np.asarray(fig1.ginput(n=1, timeout=-1)[0])
    axes1[0].scatter(arbitrary_point[0], arbitrary_point[1], marker='x',
                     color='green')
    plt.draw()
    arbitrary_point = np.array([arbitrary_point[0], arbitrary_point[1], 1])
    H1to2 = np.linalg.inv(H2to1)
    projectedPoint = np.matmul(H1to2, arbitrary_point)
    projectedPoint = projectedPoint / projectedPoint[-1]
    axes1[1].scatter(projectedPoint[0], projectedPoint[1], marker='o',
                     color='green')
    fig1.suptitle('The green circle on the right image is the corresponding '
                  'point')
    plt.draw()
    plt.savefig("../output/Section 1.2 - Transformation Correctness "
                "Example.png")


# Computes the size of an image im1 after a warp H1to2 is applied to it
def computeOutSize(im1, H2to1):
    H1to2 = np.linalg.inv(H2to1)

    # height and width of image to be warped
    height_im1, width_im1 = im1.shape[:2]

    old_upper_left_corner = np.float32([0, 0, 1])
    old_upper_right_corner = np.float32([width_im1, 0, 1])
    old_lower_left_corner = np.float32([0, height_im1, 1])
    old_lower_right_corner = np.float32([width_im1, height_im1, 1])

    # The corners of the image im1 before warping
    old_corners = np.float32([old_upper_left_corner, old_upper_right_corner,
                              old_lower_left_corner,
                              old_lower_right_corner]).T

    # The corners of the image im1 after warping (after H1to2 is applied)
    new_corners = H1to2 @ old_corners
    # Adjust for scale
    new_corners[:, 0] /= new_corners[2, 0]
    new_corners[:, 1] /= new_corners[2, 1]
    new_corners[:, 2] /= new_corners[2, 2]
    new_corners[:, 3] /= new_corners[2, 3]

    # Calculate necessary size of new warped image

    # Find min X, Y coordinates
    x_min = np.min((np.floor(new_corners[0, :])))
    y_min = np.min((np.floor(new_corners[1, :])))

    # Find max X, Y coordinates
    x_max = np.max((np.ceil(new_corners[0, :])))
    y_max = np.max((np.ceil(new_corners[1, :])))

    new_width = int(x_max - x_min)
    new_height = int(y_max - y_min)
    out_size = (new_height, new_width, 3)
    return out_size


# Computes the size of the bounding box needed to warp an image im1 (left
# image) and stitch it to im2 (unwarped image - right side)
def computeOutSizeForAxisAlignment(im1, im2, H2to1):
    H1to2 = np.linalg.inv(H2to1)

    # height and width of image to be warped
    height_im1, width_im1 = im1.shape[:2]
    height_im2, width_im2 = im2.shape[:2]

    old_upper_left_corner = np.float32([0, 0, 1])
    old_upper_right_corner = np.float32([width_im1, 0, 1])
    old_lower_left_corner = np.float32([0, height_im1, 1])
    old_lower_right_corner = np.float32([width_im1, height_im1, 1])

    # The corners of the image im1 before warping
    old_corners = np.float32([old_upper_left_corner, old_upper_right_corner,
                              old_lower_left_corner,
                              old_lower_right_corner]).T

    # The corners of the image im1 after warping (after H1to2 is applied)
    new_corners = H1to2 @ old_corners

    # Adjust for scale
    new_corners[:, 0] /= new_corners[2, 0]
    new_corners[:, 1] /= new_corners[2, 1]
    new_corners[:, 2] /= new_corners[2, 2]
    new_corners[:, 3] /= new_corners[2, 3]

    # Calculate necessary size of new warped image
    x_min = np.min((np.floor(new_corners[0, :])))
    x_min = min(x_min, 0)
    y_min = np.min((np.floor(new_corners[1, :])))
    y_min = min(y_min, 0)

    x_max = np.max((np.ceil(new_corners[0, :])))
    x_max = max(x_max, width_im2)
    y_max = np.max((np.ceil(new_corners[1, :])))
    y_max = max(y_max, height_im2)

    # Calculate amount of translation necessary
    x_right_amount = int(x_min)
    y_down_amount = int(y_min)

    warped_im_width = int(x_max - x_min)
    warped_im_height = int(y_max - y_min)

    return (warped_im_height, warped_im_width, 3), y_down_amount, x_right_amount


def alignImage(im1, im2, warpedIm1, H2to1, align_warped_im=True):
    H1to2 = np.linalg.inv(H2to1)
    total_out_size, y_down_amount, x_right_amount = \
        computeOutSizeForAxisAlignment(im1, im2, H2to1)
    # TODO maybe put warped_im1 here instead of im1

    if align_warped_im == True:
        warped_im1_aligned = np.zeros(total_out_size)
        warped_im1_mask = np.where(warpedIm1 != [0, 0, 0])
        warped_im1_aligned[warped_im1_mask] = warpedIm1[warped_im1_mask]
        return warped_im1_aligned.astype(np.uint8)
    # else:
    #     # calculate how much to move down - y axis
    #     upper_left_corner = H1to2 @ np.array([0, 0, 1]).T
    #     upper_left_corner = upper_left_corner / upper_left_corner[2]
    #     y = abs(int(np.ceil(upper_left_corner[1])))
    #
    #     # calculate how much to move right - x axis
    #     x = out_size[1] - im2.shape[1]
    #
    #     im2_aligned = np.zeros(out_size, dtype='uint8')
    #     im2_mask = np.where(im2 != [0, 0, 0])
    #     #         im2_aligned[im2_mask[0] + y, im2_mask[1] + x, im2_mask[2]] = im2[im2_mask]
    #     im2_aligned[im2_mask[0] - y2, im2_mask[1] - x2, im2_mask[2]] = im2[
    #         im2_mask]
    #     return im2_aligned


# Extra functions end

# HW functions:

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


################################################################################

# Gets a set of matching points between two images - p1, p2, and calculates
# the transformation between them.
# continue later
def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    N = p1.shape[1]
    A = np.zeros((2 * N, 9))

    x_i = p2[0, :].T  # column vector of x_is (Nx1)
    y_i = p2[1, :].T  # column vector of y_is (Nx1)
    u_i = p1[0, :].T  # column vector of u_is (Nx1)
    v_i = p1[1, :].T  # column vector of v_is (Nx1)

    # Build A column by column
    A[::2, 0] = x_i  # put x_is in first column of all even rows of A
    A[::2, 1] = y_i  # put y_is in second column of all even rows of A
    A[::2, 2] = np.ones(N)  # put 1s in third column of all even rows of A

    A[1::2, 3] = x_i  # put x_is in fourth column of all odd rows of A
    A[1::2, 4] = y_i  # put y_is in fifth column of all odd rows of A
    A[1::2, 5] = np.ones(N)  # put 1s in sixth column of all odd rows of A

    A[::2,
    6] = -x_i * u_i  # put -x_i*u_i in the seventh column of all even rows of A
    A[1::2,
    6] = -x_i * v_i  # put -x_i*v_i in the seventh column of all odd rows of A

    A[::2,
    7] = -y_i * u_i  # put -y_i*u_i in the eighth column of all even rows of A
    A[1::2,
    7] = -y_i * v_i  # put -y_i*v_i in the eighth column of all odd rows of A

    A[::2, 8] = -u_i  # put -u_i in the 9th column of all the even rows of A
    A[1::2, 8] = -v_i  # put -v_i in the 9th column of all the odd rows of A

    _lambda, V = np.linalg.eig(A.T @ A)
    indexOfSmallestLambda = np.argmin(_lambda)
    H2to1 = V[:, indexOfSmallestLambda].reshape((3, 3))

    return H2to1


################################################################################

def warpH(im1, H2to1, out_size, interpolation_type='linear'):
    epsilon = 10e-15

    # Split im1 into channels
    red_channel_im1 = im1[:, :, 0]
    green_channel_im1 = im1[:, :, 1]
    blue_channel_im1 = im1[:, :, 2]

    # Create blank output image canvas
    warp_im1 = np.zeros((out_size[0], out_size[1], 3))

    # Create mappings for interpolation
    upper_left_corner = H2to1 @ np.array([0, 0, 1]).T
    upper_left_corner = upper_left_corner / (upper_left_corner[2] + epsilon)

    bottom_left_corner = H2to1 @ np.array([0, out_size[0], 1]).T
    bottom_left_corner = bottom_left_corner / (bottom_left_corner[2] + epsilon)

    upper_right_corner = H2to1 @ np.array([out_size[1], 0, 1]).T
    upper_right_corner = upper_right_corner / (upper_right_corner[2] + epsilon)

    x_grid = np.linspace(upper_left_corner[0], upper_right_corner[0],
                         im1.shape[1])
    y_grid = np.linspace(upper_left_corner[1], bottom_left_corner[1],
                         im1.shape[0])

    # Interpolate by channel
    f_red = interpolate.interp2d(x_grid, y_grid, red_channel_im1,
                                 kind=interpolation_type,
                                 fill_value=0)

    f_green = interpolate.interp2d(x_grid, y_grid, green_channel_im1,
                                   kind=interpolation_type,
                                   fill_value=0)

    f_blue = interpolate.interp2d(x_grid, y_grid, blue_channel_im1,
                                  kind=interpolation_type,
                                  fill_value=0)

    for y in range(out_size[0]):
        for x in range(out_size[1]):
            new_coords = H2to1 @ np.array([x, y, 1]).T
            new_coords = new_coords / (new_coords[2] + epsilon)
            red_value = f_red(new_coords[0], new_coords[1])
            green_value = f_green(new_coords[0], new_coords[1])
            blue_value = f_blue(new_coords[0], new_coords[1])
            warp_im1[y, x, 0] = red_value
            warp_im1[y, x, 1] = green_value
            warp_im1[y, x, 2] = blue_value

    warp_im1 = warp_im1.astype(np.uint8)
    return warp_im1


def imageStitching(img1, warp_img2):
    img1_mask = np.where(img1 != [0, 0, 0])
    print(img1_mask)
    panoImg = img1
    panoImg[img1_mask] = warp_img2
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


################################################################################
# Question 1.2
def Q1Two():
    projectPoints(im1, im2, 6)


# TODO test func - delete at end
def outSizeTestFunc():
    return computeOutSize(im1, H2to1)


# Question 1.3
def Q1Three():
    warped_im1 = warpH(im1, H2to1, out_size, interpolation_type='linear')
    return warped_im1


################################################################################
if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread('data/incline_R.png')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # fig1, axes1 = plt.subplots(1, 2)
    # Q1Two()

    p1 = np.array([[454.5436828, 511.94919355, 602.95793011, 623.95994624,
                    640.76155914, 612.75887097], [110.43200202, 111.83213642,
                                                  484.26788911, 481.4676203,
                                                  482.8677547, 197.24033535]])
    p2 = np.array([[118.05591398, 185.7, 294.24516129, 316.2688172,
                    335.14623656, 294.24516129],
                   [151.48032796, 154.62656452, 536.89430645, 535.32118817,
                    532.17495161, 244.29430645]])

    H2to1 = computeH(p1, p2)

    out_size = outSizeTestFunc()
    print(f"{out_size}")

    warped_im1 = Q1Three()
    # plt.imshow(warped_im1)

    out_size_for_alignment1 = computeOutSizeForAxisAlignment(im1, im2,
                                                             H2to1)
    print(out_size_for_alignment1)

    warped_im1_aligned = alignImage(im1, im2, warped_im1, H2to1, True)
    plt.imshow(warped_im1_aligned)
    print("end")
