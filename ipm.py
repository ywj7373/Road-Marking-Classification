import numpy as np
import cv2


# Calculate Source and Destination points
def calc_warp_points():
    src = np.float32([
        [371, 289],
        [420, 289],
        [506, 410],
        [200, 410]
    ])

    dst = np.float32([
        [200, 0],
        [506, 0],
        [506, 600],
        [200, 600]
    ])
    return src, dst


# Calculate Transform
def calc_transform(src_, dst_):
    M_ = cv2.getPerspectiveTransform(src_, dst_)
    M_inv = cv2.getPerspectiveTransform(dst_, src_)
    return M_, M_inv


# Get perspective transform 
def perspective_transform(M_, img_):
    img_size = (img_.shape[1], img_.shape[0])
    warped_img = cv2.warpPerspective(img_, M_, img_size, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return warped_img


def reverse_rectification(img):
    src_, dst_ = calc_warp_points()
    _, m_inv = calc_transform(src_, dst_)
    warped = perspective_transform(m_inv, img)
    return warped


def rectification(img):
    src_, dst_ = calc_warp_points()
    m, _ = calc_transform(src_, dst_)
    warped = perspective_transform(m, img)
    return warped




