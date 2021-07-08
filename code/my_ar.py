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
def create_ref(im_path):
    book_im = cv2.imread(im_path)
    book_im = cv2.cvtColor(book_im, cv2.COLOR_BGR2RGB)

    im2 = cv2.imread('data/pf_scan_scaled.jpg')
    p1, p2 = mh.getPoints(book_im, im2, 4)
    # fig1, axes1 = plt.subplots(1, 1)
    # axes1.imshow(book_im)
    # axes1.set_xticks([])
    # axes1.set_yticks([])
    # fig1.suptitle('Choose the 4 corners of the book, starting from the upper '
    #               'left-hand corner, and going in clockwise order')
    # p1 = []
    #
    # while len(p1) < 4:
    #     p1.append(np.asarray(fig1.ginput(n=1, timeout=-1)[0]))
    #     axes1.scatter(p1[-1][0], p1[-1][1], marker='x', color='red')
    #     plt.draw()
    # p1 = np.array(p1)
    # p1 = p1.T

    # x_min = int(np.floor(np.min(p1[0])))
    # x_max = int(np.ceil(np.max(p1[0])))
    # y_min = int(np.floor(np.min(p1[1])))
    # y_max = int(np.ceil(np.max(p1[1])))
    # book_im_cropped = book_im[y_min:y_max, x_min:x_max]
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(book_im_cropped)

    # p1 = np.stack(p1).astype("float32")

    # width = 350
    # height = 440
    # width = 672
    # height = 874
    # rectangle_points = np.array([[0, width - 1, 0, width - 1], [0, 0, height
    #                                                             - 1,
    #                                                             height - 1]])
    # im2 = np.zeros((height, width, 3))

    # H2to1 = mh.computeH(p1, rectangle_points)
    H2to1 = mh.computeH(p1, p2)
    out_size, y_down_amount, x_right_amount = mh.computeOutSizeForAxisAlignment(
        book_im, im2, H2to1)
    ref_image = mh.warpH(book_im, H2to1, out_size, y_down_amount,
                         x_right_amount, interpolation_type='linear')
    # H2to1 = mh.computeH(p1, rectangle_points)
    # out_size = mh.computeOutSize(book_im_cropped, H2to1)
    # ref_image = mh.warpH(book_im_cropped, H2to1, out_size,
    #                      interpolation_type='linear')

    # dst = np.array([
    #     [0, 0],
    #     [width - 1, 0],
    #     [width - 1, height - 1],
    #     [0, height - 1]], dtype="float32")
    # H = cv2.getPerspectiveTransform(p1, dst)
    # ref_image = cv2.warpPerspective(book_im, H, (width, height))
    return ref_image


if __name__ == '__main__':
    print('my_ar')
    #fig1, axes1 = plt.subplots(1, 2)
    test_image = create_ref('data/pf_desk.jpg')
    fig, axes = plt.subplots(1, 1)
    axes.imshow(test_image)

    print('end')
