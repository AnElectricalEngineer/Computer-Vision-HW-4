import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy.ma as ma
import time

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


# Computes the size of the bounding box needed to warp an image im1 (left
# image) and stitch it to im2 (unwarped image - right side). Returns the
# size, and the amount by which to shift the right (static) image down and
# right by.
def computeOutSize(im1, im2, H2to1):
    H1to2 = np.linalg.inv(H2to1)

    # height and width of image to be warped
    height_im1, width_im1 = im1.shape[:2]

    # height and width of static image
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


def createBigPanorama(images, mode='SIFT', use_ransac=True):
    left_side = images[:1 + (len(images) // 2)]
    right_side = images[(len(images) // 2):]
    current_result_left = left_side[0]
    current_result_right = right_side[-1]

    # Stitch left side
    for i, image in enumerate(left_side[1:]):
        if mode == 'SIFT':
            p1, p2 = getPoints_SIFT(current_result_left, image)
        else:
            p1, p2 = getPoints(current_result_left, image, 6, True)
        if use_ransac == True:
            H_SIFT_2to1 = ransacH(p1, p2)
        else:
            H_SIFT_2to1 = computeH(p1[:, :], p2[:, :])
        out_size, y_down_amount, x_right_amount = \
            computeOutSize(current_result_left, image, H_SIFT_2to1)
        warped_im1_SIFT = warpH(current_result_left, H_SIFT_2to1, out_size,
                                y_down_amount, x_right_amount,
                                interpolation_type='linear')
        current_result_left = imageStitching(warped_im1_SIFT, image,
                                             y_down_amount, x_right_amount)
    # Stitch right side
    for i, image in enumerate(reversed(right_side[:-1])):
        if mode == 'SIFT':
            p1, p2 = getPoints_SIFT(current_result_right, image)
        else:
            p1, p2 = getPoints(current_result_right, image, 6, True)
        if use_ransac == True:
            H_SIFT_2to1 = ransacH(p1, p2)
        else:
            H_SIFT_2to1 = computeH(p1[:, :], p2[:, :])
        out_size, y_down_amount, x_right_amount = \
            computeOutSize(current_result_right, image, H_SIFT_2to1)
        warped_im1_SIFT = warpH(current_result_right, H_SIFT_2to1, out_size,
                                y_down_amount, x_right_amount,
                                interpolation_type='linear')
        current_result_right = imageStitching(warped_im1_SIFT, image,
                                              y_down_amount, x_right_amount)

    # Stitch left and right results to one large panorama
    if mode == 'SIFT':
        p1, p2 = getPoints_SIFT(current_result_left, current_result_right)
    else:
        p1, p2 = getPoints(current_result_left, current_result_right, 6, True)
    if use_ransac == True:
        H_SIFT_2to1 = ransacH(p1, p2)
    else:
        H_SIFT_2to1 = computeH(p1[:, :], p2[:, :])
    out_size, y_down_amount, x_right_amount = \
        computeOutSize(current_result_left, current_result_right,
                       H_SIFT_2to1)
    warped_im1_SIFT = warpH(current_result_left, H_SIFT_2to1, out_size,
                            y_down_amount, x_right_amount,
                            interpolation_type='linear')
    final_result = imageStitching(warped_im1_SIFT, current_result_right,
                                  y_down_amount, x_right_amount)
    return final_result


# Extra functions end

# HW functions:

def getPoints(im1, im2, N, create_new_fig=False):
    if create_new_fig:
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

def warpH(im1, H2to1, out_size, y_down, x_right, interpolation_type='linear'):
    epsilon = 10e-15

    # Split im1 into channels
    red_channel_im1 = im1[:, :, 0]
    green_channel_im1 = im1[:, :, 1]
    blue_channel_im1 = im1[:, :, 2]

    # Create blank output image canvas
    warp_im1 = np.zeros((out_size[0], out_size[1], 3))

    # Create mappings for interpolation
    x_grid = np.arange(im1.shape[1]).astype(float)
    y_grid = np.arange(im1.shape[0]).astype(float)

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

    for x in range(out_size[1]):
        for y in range(out_size[0]):
            new_coords = H2to1 @ np.array([x + x_right, y + y_down, 1]).T
            new_coords = new_coords / (new_coords[2] + epsilon)

            # Skip coordinates that are out of bounds
            if 0 <= new_coords[1] < len(y_grid) and 0 <= new_coords[0] < len(
                    x_grid):
                # Do inverse warping
                red_value = f_red(new_coords[0], new_coords[1])
                green_value = f_green(new_coords[0], new_coords[1])
                blue_value = f_blue(new_coords[0], new_coords[1])
                warp_im1[y, x, 0] = red_value
                warp_im1[y, x, 1] = green_value
                warp_im1[y, x, 2] = blue_value

    warp_im1 = warp_im1.astype(np.uint8)
    return warp_im1


# Receives two images, im1, and im2. Im1 is the left image, which is warped.
# Im2 is the right image, which is not warped. Stitches the two images
# together and returns the panorama image.
def imageStitching(im1, im2, y_down_amount, x_right_amount):
    im2_aligned = np.zeros(im1.shape)
    im2_aligned[-y_down_amount:im2.shape[0] - y_down_amount,
    - x_right_amount:im2.shape[1] - x_right_amount, :] = np.copy(im2)

    im1_mask = np.where(im1 == [0, 0, 0])
    panoImg = im1
    panoImg[im1_mask] = im2_aligned[im1_mask]
    panoImg = panoImg.astype('uint8')
    return panoImg


def ransacH(p1, p2, nIter=250, tol=3):
    num_points = p1.shape[1]
    p2_homogeneous = np.vstack((p2, np.ones((1, num_points))))
    max_num_inliers = 0
    best_inliers_indices = np.array([])

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
        if num_inliers >= max_num_inliers:
            max_num_inliers = num_inliers
            best_inliers_indices = np.where(norms < tol ** 2)

    bestH = computeH(p1[:, best_inliers_indices].squeeze(), p2[:,
                                                            best_inliers_indices].squeeze())
    return bestH


def getPoints_SIFT(im1, im2):
    # Initiate the SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors
    kp_im1, desc_im1 = sift.detectAndCompute(im1, None)
    kp_im2, desc_im2 = sift.detectAndCompute(im2, None)

    # Match keypoints using L2 norm - documentation said that it is good for
    # SIFT Use crossCheck=true which ensures that the matcher returns only
    # those matches with value (i,j) such that i-th descriptor in set A has
    # j-th descriptor in set B as the best match and vice-versa.
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
    # These points are good pairs
    p1 = np.array([[454.5436828, 511.94919355, 602.95793011, 623.95994624,
                    640.76155914, 612.75887097], [110.43200202, 111.83213642,
                                                  484.26788911, 481.4676203,
                                                  482.8677547, 197.24033535]])
    p2 = np.array([[118.05591398, 185.7, 294.24516129, 316.2688172,
                    335.14623656, 294.24516129],
                   [151.48032796, 154.62656452, 536.89430645, 535.32118817,
                    532.17495161, 244.29430645]])

    H2to1 = computeH(p1, p2)
    out_size, y_down_amount, x_right_amount = computeOutSize(im1, im2, H2to1)

    # Linear interpolation
    warped_im1_linear = warpH(im1, H2to1, out_size, y_down_amount,
                              x_right_amount,
                              interpolation_type='linear')
    fig2, axes2 = plt.subplots(1, 1)
    axes2.imshow(warped_im1_linear)
    axes2.set_xticks([])
    axes2.set_yticks([])
    fig2.suptitle('Section 1.3 - Incline_L image warped using linear '
                  'interpolation')
    plt.savefig("./../output/Section 1.3 - Incline_L image warped using "
                "linear interpolation.png")

    # Cubic interpolation
    warped_im1_cubic = warpH(im1, H2to1, out_size, y_down_amount,
                             x_right_amount,
                             interpolation_type='cubic')
    fig3, axes3 = plt.subplots(1, 1)
    axes3.imshow(warped_im1_cubic)
    axes3.set_xticks([])
    axes3.set_yticks([])
    fig3.suptitle('Section 1.3 - Incline_L image warped using cubic '
                  'interpolation')
    plt.savefig("./../output/Section 1.3 - Incline_L image warped using "
                "cubic interpolation.png")


def Q1Four():
    p1 = np.array([[454.5436828, 511.94919355, 602.95793011, 623.95994624,
                    640.76155914, 612.75887097], [110.43200202, 111.83213642,
                                                  484.26788911, 481.4676203,
                                                  482.8677547, 197.24033535]])
    p2 = np.array([[118.05591398, 185.7, 294.24516129, 316.2688172,
                    335.14623656, 294.24516129],
                   [151.48032796, 154.62656452, 536.89430645, 535.32118817,
                    532.17495161, 244.29430645]])

    H2to1 = computeH(p1, p2)
    out_size, y_down_amount, x_right_amount = computeOutSize(im1, im2, H2to1)
    warped_im1_linear = warpH(im1, H2to1, out_size, y_down_amount,
                              x_right_amount,
                              interpolation_type='linear')
    stitched_image = imageStitching(warped_im1_linear, im2, y_down_amount,
                                    x_right_amount)

    fig, axes = plt.subplots(1, 1)
    axes.imshow(stitched_image)
    axes.set_xticks([])
    axes.set_yticks([])
    fig.suptitle('Section 1.4 - Stitched Incline_L and Incline_R images')
    plt.savefig("./../output/Section 1.4 - Stitched Incline_L and Incline_R "
                "images.png")


def Q1Five():
    p1_SIFT, p2_SIFT = getPoints_SIFT(im1, im2)

    # Compute H based on 10 best SIFT features
    H_SIFT_2to1 = computeH(p1_SIFT[:, :10], p2_SIFT[:, :10])
    out_size_im_1_sift_warped, y_down_amount, x_right_amount = computeOutSize(
        im1, im2, H_SIFT_2to1)
    warped_im1_SIFT = warpH(im1, H_SIFT_2to1, out_size_im_1_sift_warped,
                            y_down_amount, x_right_amount,
                            interpolation_type='linear')
    stitched_image_SIFT = imageStitching(warped_im1_SIFT, im2, y_down_amount,
                                         x_right_amount)
    fig, axes = plt.subplots(1, 1)
    axes.imshow(stitched_image_SIFT)
    axes.set_xticks([])
    axes.set_yticks([])
    fig.suptitle('Section 1.5 - Stitched Incline Images using SIFT')
    plt.savefig("./../output/Section 1.5 - Stitched Incline Images using "
                "SIFT.png")


def Q1Six():
    # Load beach images
    beach_images = [
        cv2.cvtColor(cv2.imread('./data/beach1.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach2.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach3.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach4.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach5.jpg'), cv2.COLOR_BGR2RGB)]

    scale_percent = 40  # percent of original size
    width = int(beach_images[0].shape[1] * scale_percent / 100)
    height = int(beach_images[0].shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the images due to time constraints
    beach_images = [cv2.resize(img, dim) for img in beach_images]
    # beach_images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in
    #                 beach_images]
    # manual_stitch_beach = createBigPanorama(beach_images, mode='MANUAL',
    #                                         use_ransac=False)
    # manual_stitch_beach = cv2.rotate(manual_stitch_beach,
    #                                  cv2.ROTATE_90_COUNTERCLOCKWISE)

    # fig1, axes1 = plt.subplots(1, 1)
    # axes1.imshow(manual_stitch_beach)
    # axes1.set_xticks([])
    # axes1.set_yticks([])
    # fig1.suptitle('Section 1.6 - Stitched Beach Images Using Manually '
    #               'Selected Points')
    # plt.savefig("./../output/Section 1.6 - Stitched Beach Images Using "
    #             "Manually Selected Points.png")

    start_SIFT_stitch_beach_time = time.time()
    print(f"Q1.6 - Starting panorama creation of Beach images using SIFT and "
          f"no RANSAC, with scale factor of {scale_percent}")
    SIFT_stitch_beach = createBigPanorama(beach_images, mode='SIFT',
                                          use_ransac=False)
    end_SIFT_stitch_beach_time = time.time()
    print(f"Q1.6 - Panorama creation of Beach images using SIFT and no RANSAC "
          f"with with scale factor of {scale_percent} took "
          f"{end_SIFT_stitch_beach_time - start_SIFT_stitch_beach_time} "
          f"seconds to complete")
    # SIFT_stitch_beach = cv2.rotate(SIFT_stitch_beach,
    #                                cv2.ROTATE_90_COUNTERCLOCKWISE)
    fig2, axes2 = plt.subplots(1, 1)
    axes2.imshow(SIFT_stitch_beach)
    axes2.set_xticks([])
    axes2.set_yticks([])
    fig2.suptitle('Section 1.6 - Stitched Beach Images Using SIFT')
    plt.savefig("./../output/Section 1.6 - Stitched Beach Images Using "
                "SIFT.png")

    # Load palace images in opposite order
    palace_images = [
        cv2.cvtColor(cv2.imread('./data/sintra5.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra4.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra3.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra2.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra1.JPG'), cv2.COLOR_BGR2RGB)]

    scale_percent = 40  # percent of original size
    width = int(palace_images[0].shape[1] * scale_percent / 100)
    height = int(palace_images[0].shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the images due to time constraints
    palace_images = [cv2.resize(img, dim) for img in palace_images]

    # manual_stitch_palace = createBigPanorama(palace_images, mode='MANUAL',
    #                                          use_ransac=False)

    # fig3, axes3 = plt.subplots(1, 1)
    # axes3.imshow(manual_stitch_palace)
    # axes3.set_xticks([])
    # axes3.set_yticks([])
    # fig3.suptitle('Section 1.6 - Stitched Palace Images Using Manually '
    #               'Selected Points')
    # plt.savefig("./../output/Section 1.6 - Stitched Palace Images Using "
    #             "Manually Selected Points.png")

    start_SIFT_stitch_palace_time = time.time()
    print(f"Q1.6 - Starting panorama creation of Palace images using SIFT and "
          f"no RANSAC, with scale factor of {scale_percent}")
    SIFT_stitch_palace = createBigPanorama(palace_images, mode='SIFT',
                                           use_ransac=False)
    end_SIFT_stitch_palace_time = time.time()
    print(f"Q1.6 - Panorama creation of Palace images using SIFT and no RANSAC "
          f"with with scale factor of {scale_percent} took "
          f"{end_SIFT_stitch_palace_time - start_SIFT_stitch_palace_time} "
          f"seconds to complete")

    fig4, axes4 = plt.subplots(1, 1)
    axes4.imshow(SIFT_stitch_palace)
    axes4.set_xticks([])
    axes4.set_yticks([])
    fig4.suptitle('Section 1.6 - Stitched Palace Images Using SIFT')
    plt.savefig("./../output/Section 1.6 - Stitched Palace Images Using "
                "SIFT.png")

def Q1Seven():
    # Load beach images
    beach_images = [
        cv2.cvtColor(cv2.imread('./data/beach1.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach2.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach3.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach4.jpg'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/beach5.jpg'), cv2.COLOR_BGR2RGB)]

    scale_percent = 50  # percent of original size
    width = int(beach_images[0].shape[1] * scale_percent / 100)
    height = int(beach_images[0].shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the images due to time constraints
    beach_images = [cv2.resize(img, dim) for img in beach_images]
    # beach_images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in
    #                 beach_images]
    # manual_stitch_beach = createBigPanorama(beach_images, mode='MANUAL',
    #                                         use_ransac=True)
    # manual_stitch_beach = cv2.rotate(manual_stitch_beach,
    #                                  cv2.ROTATE_90_COUNTERCLOCKWISE)
    #
    # fig1, axes1 = plt.subplots(1, 1)
    # axes1.imshow(manual_stitch_beach)
    # axes1.set_xticks([])
    # axes1.set_yticks([])
    # fig1.suptitle('Section 1.7 - Stitched Beach Images Using Manually '
    #               'Selected Points and RANSAC')
    # plt.savefig("./../output/Section 1.7 - Stitched Beach Images Using "
    #             "Manually Selected Points and RANSAC.png")
    #
    start_SIFT_stitch_beach_time = time.time()
    print(f"Q1.7 - Starting panorama creation of Beach images using SIFT and "
          f"RANSAC, with scale factor of {scale_percent}")
    SIFT_stitch_beach = createBigPanorama(beach_images, mode='SIFT',
                                          use_ransac=True)
    end_SIFT_stitch_beach_time = time.time()
    print(f"Q1.7 - Panorama creation of Beach images using SIFT and RANSAC "
          f"with with scale factor of {scale_percent} took "
          f"{end_SIFT_stitch_beach_time - start_SIFT_stitch_beach_time} "
          f"seconds to complete")
    # SIFT_stitch_beach = cv2.rotate(SIFT_stitch_beach,
    #                                cv2.ROTATE_90_COUNTERCLOCKWISE)

    fig2, axes2 = plt.subplots(1, 1)
    axes2.imshow(SIFT_stitch_beach)
    axes2.set_xticks([])
    axes2.set_yticks([])
    fig2.suptitle('Section 1.7 - Stitched Beach Images Using SIFT and RANSAC')
    plt.savefig("./../output/Section 1.7 - Stitched Beach Images Using "
                "SIFT and RANSAC.png")

    # # Load palace images in opposite order
    palace_images = [
        cv2.cvtColor(cv2.imread('./data/sintra5.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra4.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra3.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra2.JPG'), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread('./data/sintra1.JPG'), cv2.COLOR_BGR2RGB)]

    scale_percent = 15  # percent of original size
    width = int(palace_images[0].shape[1] * scale_percent / 100)
    height = int(palace_images[0].shape[0] * scale_percent / 100)
    dim = (width, height)

    # # Resize the images due to time constraints
    palace_images = [cv2.resize(img, dim) for img in palace_images]

    # manual_stitch_palace = createBigPanorama(palace_images, mode='MANUAL',
    #                                          use_ransac=True)

    # fig3, axes3 = plt.subplots(1, 1)
    # axes3.imshow(manual_stitch_palace)
    # axes3.set_xticks([])
    # axes3.set_yticks([])
    # fig3.suptitle('Section 1.7 - Stitched Palace Images Using Manually '
    #               'Selected Points and RANSAC')
    # plt.savefig("./../output/Section 1.7 - Stitched Palace Images Using "
    #             "Manually Selected Points and RANSAC.png")

    # start_SIFT_stitch_palace_time = time.time()
    # print(f"Q1.7 - Starting panorama creation of Palace images using SIFT and "
    #       f"RANSAC, with scale factor of {scale_percent}")
    # SIFT_stitch_palace = createBigPanorama(palace_images, mode='SIFT',
    #                                        use_ransac=True)
    # end_SIFT_stitch_palace_time = time.time()
    # print(f"Q1.7 - Panorama creation of Palace images using SIFT and RANSAC "
    #       f"with with scale factor of {scale_percent} took "
    #       f"{end_SIFT_stitch_palace_time - start_SIFT_stitch_palace_time} "
    #       f"seconds to complete")

    # TODO DO NOT Uncomment the block below (until turn in)
    # fig4, axes4 = plt.subplots(1, 1)
    # axes4.imshow(SIFT_stitch_palace)
    # axes4.set_xticks([])
    # axes4.set_yticks([])
    # fig4.suptitle('Section 1.7 - Stitched Palace Images Using SIFT and RANSAC')
    # plt.savefig("./../output/Section 1.7 - Stitched Palace Images Using "
    #             "SIFT and RANSAC.png")


################################################################################
if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread('data/incline_R.png')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    fig1, axes1 = plt.subplots(1, 2)  # Figure for Q1.2
    # Q1Two()
    # Q1Three()
    # Q1Four()
    # Q1Five()
    # Q1Six()
    Q1Seven()


    print("end")
