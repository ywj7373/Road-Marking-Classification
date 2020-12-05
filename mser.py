import cv2
import numpy as np


def is_vertically_aligned(b1, b2):
    max_delta = 40.0
    (x1, y1, w1, h1) = b1
    (x2, y2, w2, h2) = b2

    if abs(y2 - y1) <= max_delta and abs(h2 - h1) <= max_delta:
        return True

    return False


def is_horizontally_close(b1, b2):
    max_delta = 100.0
    (x1, y1, w1, h1) = b1
    (x2, y2, w2, h2) = b2

    if abs(x1 - x2) <= max_delta:
        return True

    return False


def merge_box(boxes):
    X = []
    Y = []
    for box in boxes:
        (x, y, w, h) = box
        X.append(x)
        X.append(x + w)
        Y.append(y)
        Y.append(y + h)

    return min(X), min(Y), max(X) - min(X), max(Y) - min(Y)


def merge_similar_box(boxes):
    # Combine boxes that are similar
    _boxes = set()
    merged = False

    for b1 in boxes:
        similar_boxes = []
        for b2 in boxes:
            if is_vertically_aligned(b1, b2) and is_horizontally_close(b1, b2):
                similar_boxes.append(b2)

        # Combine boxes
        if len(similar_boxes) > 1:
            merged = True
            combined_box = merge_box(similar_boxes)
            _boxes.add(combined_box)
            # Remove merged boxes
            for box in similar_boxes:
                boxes.remove(box)

    if merged:
        return boxes + merge_similar_box(list(_boxes))
    else:
        return boxes


def mser(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _mser = cv2.MSER_create(_delta=5, _min_area=1000)
    regions, _ = _mser.detectRegions(gray)

    boxes = []
    for region in regions:
        (x, y, w, h) = cv2.boundingRect(np.reshape(region, (-1, 1, 2)))
        boxes.append((x, y, w, h))

    # Merge similar boxes
    return merge_similar_box(boxes)
