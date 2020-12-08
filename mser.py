import cv2
import numpy as np


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


def is_similar_box_by_iou(b1, b2):
    (x1, y1, w1, h1) = b1
    (x2, y2, w2, h2) = b2

    if min(x1+w1, x2+w2) - max(x1, x2) < 0 or min(y1+h1, y2+h2) - max(y1, y2) < 0:
        return False

    inter = (min(x1+w1, x2+w2) - max(x1, x2)) * (min(y1+h1, y2+h2) - max(y1, y2))
    union = (w1*h1 + w2*h2 - inter + 1e-8)
    iou = inter / union
    print(iou)

    if iou > 0.2:
        return True
    else:
        return False


def is_close(b1, b2):
    (x1, y1, w1, h1) = b1
    (x2, y2, w2, h2) = b2
    xy1s = [(x1, y1), (x1+w1, y1), (x1, y1+h1), (x1+w1, y1+h1)]
    xy2s = [(x2, y2), (x2+w2, y2), (x2, y2+h2), (x2+w2, y2+h2)]
    distance = 100000000000
    for xy1 in xy1s:
        for xy2 in xy2s:
            distance = min(distance, abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1]))

    return distance < 30


# Combine boxes that are similar
def merge_similar_box(boxes):
    _boxes = set()

    index = 0
    while index < len(boxes):
        if len(boxes) <= 1:
            break

        b1 = boxes[index]
        to_merge = None
        for b2 in boxes[index+1:]:
            # print(b1, b2)
            if is_close(b1, b2):
                to_merge = b2
                break
            # if is_similar_box_by_iou(b1, b2):
            #     print(f'is similar: {b1}, {b2}')
            #     to_merge = b2
            #     break

        if to_merge is None:
            index += 1
            continue

        combined_box = merge_box([b1, to_merge])
        boxes = boxes[:index] + [combined_box] + boxes[index:]
        boxes.remove(b1)
        boxes.remove(to_merge)

    _boxes = []
    for box in boxes:
        (x, y, w, h) = box

        if (w*h) > 50000: # Remove big box
            continue

        if x < 200 or x+w > 600: # Remove box on left and right side
            continue

        if h/w > 2:
            continue

        if w/h > 4:
            continue
        _boxes.append(box)

    sorted(_boxes, key=lambda x: x[1] + x[3])

    return _boxes


def mser(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _mser = cv2.MSER_create(_delta=5, _min_area=1000, _max_area=10000)
    regions, _ = _mser.detectRegions(gray)

    boxes = []
    for region in regions:
        (x, y, w, h) = cv2.boundingRect(np.reshape(region, (-1, 1, 2)))
        boxes.append((x, y, w, h))

    # Merge similar boxes
    return merge_similar_box(boxes)
