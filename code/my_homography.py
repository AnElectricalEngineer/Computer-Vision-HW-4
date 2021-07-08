import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy.ma as ma

# Add imports if needed:

import random


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
    else:
        im2_aligned = np.zeros(total_out_size)
        im2_mask = np.where(im2 != [0, 0, 0])
        im2_aligned[im2_mask[0] - y_down_amount, im2_mask[1] - x_right_amount,
                    im2_mask[2]] = im2[im2_mask]
        return im2_aligned.astype(np.uint8)


# Here we took 10 best features...make sure correct
# TODO look at mergeSeveralImages() method from reference 1 -
#  I think we did this part wrong...best I think to start from center
#  outward, AND compute warps based on (possibly warped) images only,
#  not entire panorama
def createBigPanorama(images, mode='SIFT'):
    current_result = images[0]
    if mode == 'SIFT':
        for i, image in enumerate(images[1:]):
            p1_SIFT, p2_SIFT = getPoints_SIFT(current_result, image)
            H_SIFT_2to1 = computeH(p1_SIFT[:, :10], p2_SIFT[:, :10])
            out_size = computeOutSize(current_result, H_SIFT_2to1)
            warped_im1_SIFT = warpH(current_result, H_SIFT_2to1, out_size,
                                    interpolation_type='linear')
            warped_im1_SIFT_aligned = alignImage(current_result, image,
                                                 warped_im1_SIFT, H_SIFT_2to1,
                                                 True)
            image_aligned = alignImage(current_result, image, warped_im1_SIFT,
                                       H_SIFT_2to1, False)
            current_result = imageStitching(image_aligned,
                                            warped_im1_SIFT_aligned)
        return current_result


# Extra functions end

# HW functions:

def getPoints(im1, im2, N):
    fig1, axes1 = plt.subplots(1, 2)
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
    # TODO maybe change to np.abs(_lambda).argmin()
    indexOfSmallestLambda = np.argmin(_lambda)
    H2to1 = V[:, indexOfSmallestLambda].reshape((3, 3))

    return H2to1


################################################################################

# TODO change all the transposeds
def warpH(im1, H2to1, out_size, y_down, x_right, interpolation_type='linear'):
    epsilon = 10e-15

    # Split im1 into channels
    red_channel_im1 = im1[:, :, 0]
    green_channel_im1 = im1[:, :, 1]
    blue_channel_im1 = im1[:, :, 2]

    # Create blank output image canvas
    # warp_im1 = np.zeros((out_size[0], out_size[1], 3))
    warp_im1 = np.zeros((out_size[1], out_size[0], 3))

    # Create mappings for interpolation
    # TODO maybe remove epsilons?
    # upper_left_corner = H2to1 @ np.array([0, 0, 1]).T
    # upper_left_corner = upper_left_corner / (upper_left_corner[2] + epsilon)
    #
    # bottom_left_corner = H2to1 @ np.array([0, out_size[0], 1]).T
    # bottom_left_corner = bottom_left_corner / (bottom_left_corner[2] + epsilon)
    #
    # upper_right_corner = H2to1 @ np.array([out_size[1], 0, 1]).T
    # upper_right_corner = upper_right_corner / (upper_right_corner[2] + epsilon)
    #
    # x_grid = np.linspace(upper_left_corner[0], upper_right_corner[0],
    #                      im1.shape[1])
    # y_grid = np.linspace(upper_left_corner[1], bottom_left_corner[1],
    #                      im1.shape[0])

    x_grid = np.arange(im1.shape[0]).astype(float)
    y_grid = np.arange(im1.shape[1]).astype(float)

    # Interpolate by channel
    f_red = interpolate.interp2d(x_grid, y_grid, red_channel_im1.T,
                                 kind=interpolation_type,
                                 fill_value=0)

    f_green = interpolate.interp2d(x_grid, y_grid, green_channel_im1.T,
                                   kind=interpolation_type,
                                   fill_value=0)

    f_blue = interpolate.interp2d(x_grid, y_grid, blue_channel_im1.T,
                                  kind=interpolation_type,
                                  fill_value=0)

    # for y in range(out_size[0]):
    #     for x in range(out_size[1]):
    for x in range(out_size[1]):
        for y in range(out_size[0]):
            # new_coords = H2to1 @ np.array([x + x_right, y + y_down, 1]).T
            new_coords = H2to1 @ np.array([x + x_right, y + y_down, 1])
            # TODO maybe remove epsilon?
            new_coords = new_coords / (new_coords[2] + epsilon)
            if 0 <= new_coords[0] < len(y_grid) and 0 <= new_coords[1] < len(
                    x_grid):
                # red_value = f_red(new_coords[0], new_coords[1])
                # green_value = f_green(new_coords[0], new_coords[1])
                # blue_value = f_blue(new_coords[0], new_coords[1])
                # warp_im1[y, x, 0] = red_value
                # warp_im1[y, x, 1] = green_value
                # warp_im1[y, x, 2] = blue_value

                red_value = f_red(new_coords[1], new_coords[0])
                green_value = f_green(new_coords[1], new_coords[0])
                blue_value = f_blue(new_coords[1], new_coords[0])
                warp_im1[x, y, 0] = red_value
                warp_im1[x, y, 1] = green_value
                warp_im1[x, y, 2] = blue_value

    warp_im1 = warp_im1.astype(np.uint8)
    warp_im1 = np.transpose(warp_im1, (1, 0, 2))
    return warp_im1

# TODO review this
# def imageStitching(im1, im2):
#     im1_mask = np.where(im1 == [0, 0, 0])
#     panoImg = im1
#     panoImg[im1_mask] = im2[im1_mask]
#     return panoImg

def imageStitching(im1, im2, y_down_amount, x_right_amount):
    h1, w1, _ = im2.shape
    im2_aligned = np.zeros(im1.shape)
    im2_aligned[-y_down_amount:h1 - y_down_amount, -x_right_amount:w1 -
                                                                   x_right_amount, :] = np.copy(im2)
    im2_mask = np.where(im1 == [0, 0, 0])
    panoImg = im1
    panoImg[im2_mask] = im2_aligned[im2_mask]
    panoImg = panoImg.astype('uint8')
    return panoImg


def ransacH(p1, p2, nIter=250, tol=3):
    num_points = p1.shape[1]
    p2_homogeneous = np.vstack((p2, np.ones((1, num_points))))
    max_num_inliers = 0

    for i in range(nIter):
        # choose 4 random points
        num_points = p1.shape[1]
        random_indexes = random.sample(range(0, num_points), 4)
        chosen_points_1 = p1[:, random_indexes]
        chosen_points_2 = p2[:, random_indexes]

        # compute H based on 4 random points
        H2to1 = computeH(chosen_points_1, chosen_points_2)

        # check norma
        projected_points = H2to1 @ p2_homogeneous
        projected_points[:, :] /= projected_points[2, :]
        norms = np.sum((projected_points[:2] - p1) ** 2, axis=0)
        num_inliers = np.count_nonzero(norms < tol ** 2)
        if (num_inliers >= max_num_inliers):
            max_num_inliers = num_inliers
            bestH = H2to1

    return bestH


def getPoints_SIFT(im1, im2):
    # Initiate the SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors
    kp_im1, desc_im1 = sift.detectAndCompute(im1, None)
    kp_im2, desc_im2 = sift.detectAndCompute(im2, None)

    # Match keypoints using L2 norm - documentation said that it is good for SIFT
    # Use crossCheck=true which ensures that the matcher returns only those matches
    # with value (i,j) such that i-th descriptor in set A has j-th descriptor in
    # set B as the best match and vice-versa.
    brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Get best matches of the two images
    matches = brute_force.match(desc_im1, desc_im2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    p1 = np.array([kp_im1[m.queryIdx].pt for m in matches]).T
    p2 = np.array([kp_im2[m.trainIdx].pt for m in matches]).T

    return p1, p2


################################################################################
# Question 1.2
def Q1Two():
    projectPoints(im1, im2, 6)


# Question 1.3
def Q1Three():
    warped_im1 = warpH(im1, H2to1, out_size, y_down_amount, x_right_amount,
                       interpolation_type='linear')
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

    # p1 = np.array([[454.5436828, 511.94919355, 602.95793011, 623.95994624,
    #                 640.76155914, 612.75887097], [110.43200202, 111.83213642,
    #                                               484.26788911, 481.4676203,
    #                                               482.8677547, 197.24033535]])
    # p2 = np.array([[118.05591398, 185.7, 294.24516129, 316.2688172,
    #                 335.14623656, 294.24516129],
    #                [151.48032796, 154.62656452, 536.89430645, 535.32118817,
    #                 532.17495161, 244.29430645]])

    p1 = np.array(
        [[517.60420168, 703.15042017, 603.11680672, 520.83109244, 541.80588235,
          627.31848739, 383.68823529],
         [450.61932773, 516.77058824, 502.24957983, 171.49327731, 126.31680672,
          255.39243697, 182.78739496]])
    p2 = np.array(
        [[202.98235294, 394.98235294, 296.56218487, 198.14201681, 220.7302521,
          311.08319328, 40.02436975],
         [508.05798319, 559.68823529, 558.07478992, 220.86470588, 172.46134454,
          306.37731092, 228.93193277]])

    H2to1 = computeH(p1, p2)
    # out_size = computeOutSize(im1, H2to1)
    # print(f"{out_size}")

    out_size, y_down_amount, x_right_amount = computeOutSizeForAxisAlignment(
        im1, im2, H2to1)

    # warped_im1 = Q1Three()
    # np.save('./../temp/warped_im1', warped_im1)
    # plt.imshow(warped_im1)

    warped_im1_aligned = np.load('./../temp/warped_im1.npy')
    # out_size_for_alignment1 = computeOutSizeForAxisAlignment(im1, im2,
    #                                                          H2to1)
    # print(out_size_for_alignment1)

    # warped_im1_aligned = alignImage(im1, im2, warped_im1, H2to1, True)
    # plt.imshow(warped_im1_aligned)

    # im2_aligned = alignImage(im1, im2, warped_im1_aligned, H2to1, False)
    # plt.imshow(im2_aligned)

    stitched_image = imageStitching(warped_im1_aligned, im2, y_down_amount,
                                    x_right_amount)
    plt.imshow(stitched_image)

    # p1_SIFT, p2_SIFT = getPoints_SIFT(im1, im2)
    # H_SIFT_2to1 = computeH(p1_SIFT[:, :10], p2_SIFT[:, :10])
    # out_size_im_1_sift_warped = computeOutSize(im1, H_SIFT_2to1)
    # warped_im1_SIFT = warpH(im1, H_SIFT_2to1, out_size_im_1_sift_warped,
    #                         interpolation_type='linear')
    # np.save('./../temp/warped_im1_SIFT', warped_im1_SIFT)

    # warped_im1_SIFT = np.load('./../temp/warped_im1_SIFT.npy')
    # plt.imshow(warped_im1_SIFT)

    # warped_im1_SIFT_aligned = alignImage(im1, im2, warped_im1_SIFT,
    #                                      H_SIFT_2to1, True)
    # im2_aligned = alignImage(im1, im2, warped_im1_SIFT, H_SIFT_2to1, False)

    # plt.imshow(warped_im1_SIFT_aligned)
    # stitched_image_SIFT = imageStitching(im2_aligned, warped_im1_SIFT_aligned)
    # plt.imshow(stitched_image_SIFT)

    # beach_images = [cv2.imread('./data/beach1.jpg'),
    #                 cv2.imread('./data/beach2.jpg'),
    #                 cv2.imread('./data/beach3.jpg'),
    #                 cv2.imread('./data/beach4.jpg')]
    #
    # scale_percent = 30  # percent of original size
    # width = int(beach_images[0].shape[1] * scale_percent / 100)
    # height = int(beach_images[0].shape[0] * scale_percent / 100)
    # dim = (width, height)
    #
    # beach_images = [cv2.resize(img, dim) for img in beach_images]
    # beach_images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in
    #                 beach_images]
    #
    # SIFT_result = createBigPanorama(beach_images)
    # SIFT_result = cv2.cvtColor(SIFT_result, cv2.COLOR_BGR2RGB)
    # plt.imshow(SIFT_result)

    # palace_images = [cv2.imread('./data/sintra5.JPG'),
    #                  cv2.imread('./data/sintra4.JPG')]
    #
    # scale_percent = 20  # percent of original size
    # width = int(palace_images[0].shape[1] * scale_percent / 100)
    # height = int(palace_images[0].shape[0] * scale_percent / 100)
    # dim = (width, height)
    #
    # palace_images = [cv2.resize(img, dim) for img in palace_images]
    #
    # SIFT_result_palace = createBigPanorama(palace_images)
    # SIFT_result_palace = cv2.cvtColor(SIFT_result_palace, cv2.COLOR_BGR2RGB)
    # plt.imshow(SIFT_result_palace)

    # H2to1_Ransac_SIFT = ransacH(p1_SIFT, p1_SIFT)
    # print(H2to1_Ransac_SIFT)

    print("end")
