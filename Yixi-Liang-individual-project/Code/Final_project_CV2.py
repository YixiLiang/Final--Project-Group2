"""
from kaggle.com/code/tyrionlannisterlzy/traditional-method-to-do-interpolations
"""

import cv2
import numpy as np
# import ffmpeg
import os

from tqdm import tqdm

cur_file = os.getcwd()
base_directory = cur_file + '/Image Super Resolution - Unsplash'
hires_folder = os.path.join(base_directory, 'high res')
lowres_folder = os.path.join(base_directory, 'low res')

# ------------------------------------------------------------------------------------------------------------------
def base_function(x, a=-0.5):
    # describe the base function sin(x)/x
    Wx = 0
    if np.abs(x) <= 1:
        Wx = (a + 2) * (np.abs(x) ** 3) - (a + 3) * x ** 2 + 1
    elif 1 <= np.abs(x) <= 2:
        Wx = a * (np.abs(x) ** 3) - 5 * a * (np.abs(x) ** 2) + 8 * a * np.abs(x) - 4 * a
    return Wx

def padding(img):
    h, w, c = img.shape
    print(img.shape)
    pad_image = np.zeros((h + 4, w + 4, c))
    pad_image[2:h + 2, 2:w + 2] = img
    return pad_image

def bicubic(img, sacle, a=-0.5):
    print("Doing bicubic")
    h, w, color = img.shape
    img = padding(img)
    nh = h*sacle
    nw = h*sacle
    new_img = np.zeros((nh, nw, color))

    for c in tqdm(range(color)):
        for i in range(nw):
            for j in range(nh):

                px = i/sacle + 2
                py = j/sacle + 2
                px_int = int(px)
                py_int = int(py)
                u = px - px_int
                v = py - py_int

                A = np.matrix([[base_function(u+1, a)], [base_function(u, a)], [base_function(u-1, a)], [base_function(u-2, a)]])
                C = np.matrix([base_function(v+1, a), base_function(v, a), base_function(v-1, a), base_function(v-2, a)])
                B = np.matrix([[img[py_int-1, px_int-1][c], img[py_int-1, px_int][c], img[py_int-1, px_int+1][c], img[py_int-1, px_int+2][c]],
                               [img[py_int, px_int-1][c], img[py_int, px_int][c], img[py_int, px_int+1][c], img[py_int, px_int+2][c]],
                               [img[py_int+1, px_int-1][c], img[py_int+1, px_int][c], img[py_int+1, px_int+1][c], img[py_int+1, px_int+2][c]],
                               [img[py_int+2, px_int-1][c], img[py_int+2, px_int][c], img[py_int+2, px_int+1][c], img[py_int+2, px_int+2][c]]])
                new_img[j, i][c] = np.dot(np.dot(C, B), A)
    return new_img
# ------------------------------------------------------------------------------------------------------------------

for i in range(1, 4):
    img_l = cv2.imread(lowres_folder + f'/{i}_6.jpg')
    img_h = cv2.imread(hires_folder + f'/{i}.jpg')
    size = (1200, 800)

    new_img = cv2.resize(img_l, size, interpolation=cv2.INTER_NEAREST)
    new_img_path = base_directory + f'/test_cv/{i}_cv_INTER_NEAREST.jpg'
    cv2.imwrite(new_img_path, new_img)

    new_img = cv2.resize(img_l, size, interpolation=cv2.INTER_LINEAR)
    new_img_path = base_directory + f'/test_cv/{i}_cv_INTER_LINEAR.jpg'
    cv2.imwrite(new_img_path, new_img)

    new_img = cv2.resize(img_l, size, interpolation=cv2.INTER_AREA)
    new_img_path = base_directory + f'/test_cv/{i}_cv_INTER_AREA.jpg'
    cv2.imwrite(new_img_path, new_img)

    new_img = cv2.resize(img_l, size, interpolation=cv2.INTER_CUBIC)
    new_img_path = base_directory + f'/test_cv/{i}_cv_INTER_CUBIC.jpg'
    cv2.imwrite(new_img_path, new_img)

    new_img = cv2.resize(img_l, size, interpolation=cv2.INTER_LANCZOS4)
    new_img_path = base_directory + f'/test_cv/{i}_cv_INTER_LANCZOS4.jpg'
    cv2.imwrite(new_img_path, new_img)

    # sacle = 4
    # new_img = bicubic(img_l, sacle)
    # new_img_path = base_directory + f'/test_cv/{i}_cv_bicubic.jpg'
    # cv2.imwrite(new_img_path, new_img)

print('Finish')
