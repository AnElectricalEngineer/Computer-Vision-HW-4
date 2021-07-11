import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
import my_homography as mh

# Add functions here:
"""
   Your code here
"""


# Functions end

# HW functions:
def create_ref(im_path, do_new_warp=False):
    book_im = cv2.imread(im_path)
    book_im = cv2.cvtColor(book_im, cv2.COLOR_BGR2RGB)

    fig1, axes1 = plt.subplots(1, 1)
    axes1.imshow(book_im)
    axes1.set_xticks([])
    axes1.set_yticks([])
    fig1.suptitle('Choose the 4 corners of the book, starting from the upper '
                  'left-hand corner, and going in clockwise order')
    p1 = []

    while len(p1) < 4:
        p1.append(np.asarray(fig1.ginput(n=1, timeout=-1)[0]))
        axes1.scatter(p1[-1][0], p1[-1][1], marker='x', color='red')
        plt.draw()
    p1 = np.array(p1)
    p1 = p1.T

    width = 350
    height = 440
    rectangle_points = np.array([[0, width - 1, width - 1, 0], [0, 0, height
                                                                - 1,
                                                                height - 1]])
    im2 = np.zeros((height, width, 3))

    H2to1 = mh.computeH(p1, rectangle_points)
    out_size, y_down_amount, x_right_amount = mh.computeOutSize(book_im, im2,
                                                                H2to1)

    p1_homogeneous = np.vstack((p1, np.ones((1, 4))))
    new_coords = np.linalg.inv(H2to1) @ p1_homogeneous
    new_coords[:, 0] /= new_coords[2, 0]
    new_coords[:, 1] /= new_coords[2, 1]
    new_coords[:, 2] /= new_coords[2, 2]
    new_coords[:, 3] /= new_coords[2, 3]

    for points in range(4):
        new_coords[0, points] -= x_right_amount
        new_coords[1, points] -= y_down_amount

    if do_new_warp == True:
        ref_image = mh.warpH(book_im, H2to1, out_size, y_down_amount,
                             x_right_amount, interpolation_type='linear')
        x_min = int(new_coords[0, 0])
        y_min = int(new_coords[1, 0])
        x_max = int(new_coords[0, 1])
        y_max = int(new_coords[1, 2])

        ref_image = ref_image[y_min:y_max, x_min:x_max]
        np.save('./../temp/ref_image', ref_image)
    else:
        # x_min = int(new_coords[0, 0])
        # y_min = int(new_coords[1, 0])
        # x_max = int(new_coords[0, 1])
        # y_max = int(new_coords[1, 2])

        ref_image = np.load('./../temp/ref_image.npy')
        # ref_image = ref_image[y_min:y_max, x_min:x_max]
    return ref_image


def im2im(ref_im, scene_im_path, do_new_warp=False):
    scene_im = cv2.imread(scene_im_path)
    scene_im = cv2.cvtColor(scene_im, cv2.COLOR_BGR2RGB)

    fig1, axes1 = plt.subplots(1, 1)
    axes1.imshow(scene_im)
    axes1.set_xticks([])
    axes1.set_yticks([])
    fig1.suptitle('Choose the 4 corners of the book in the scene, starting '
                  'from the upper left-hand corner, and going in clockwise '
                  'order')
    p2 = []

    while len(p2) < 4:
        p2.append(np.asarray(fig1.ginput(n=1, timeout=-1)[0]))
        axes1.scatter(p2[-1][0], p2[-1][1], marker='x', color='red')
        plt.draw()
    p2 = np.array(p2)
    p2 = p2.T

    x_size = int(p2[0, 2] - p2[0, 3])
    y_size = int(p2[1, 3] - p2[1, 0])

    width = ref_im.shape[1]
    height = ref_im.shape[0]
    rectangle_points = np.array([[0, width - 1, width - 1, 0], [0, 0, height
                                                                - 1,
                                                                height - 1]])

    H2to1 = mh.computeH(rectangle_points, p2)
    y_down_amount, x_right_amount = -int(p2[0, 1]), -int(p2[0, 0])
    if do_new_warp == True:
        ref_im_warped = mh.warpH(ref_im, H2to1, scene_im.shape,
                                 0, 0)
        np.save('./../temp/ref_image_warped', ref_im_warped)
    else:
        ref_im_warped = np.load('./../temp/ref_image_warped.npy')

    ref_im_mask = np.where(ref_im_warped == [0, 0, 0])
    ref_im_warped[ref_im_mask] = scene_im[ref_im_mask]

    fig2, axes2 = plt.subplots(1, 1)
    axes2.imshow(ref_im_warped)
    axes2.set_xticks([])
    axes2.set_yticks([])


if __name__ == '__main__':
    print('my_ar')

    ref_im = create_ref('data/pf_desk.jpg', False)
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(test_image)

    im2im(ref_im, 'my_data/scene_im_2.jpeg', False)

    print('end')
