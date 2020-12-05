import cv2
import os
import numpy as np
import time


def hog(img, box):
    _hog = cv2.HOGDescriptor()

    (x, y, w, h) = box
    crop_img = img[y:y+h, x:x+w]
    resize_img = cv2.resize(crop_img, (64, 128))
    return _hog.compute(resize_img)  # (3780, 1)


def train_svm(labels, train_hog):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.01)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 10, 1.0))

    train_hog = np.array(train_hog).astype(np.float32)
    labels = np.array(labels).reshape((-1, 1))

    start_time = time.time()
    if svm.train(train_hog, cv2.ml.ROW_SAMPLE, labels):
        train_preds = svm.predict(train_hog)[1]
        print("Elasped time: {:6}s".format(time.time() - start_time))
        print('Training Accuracy: %.6f' % np.average(train_preds == labels))
        svm.save('svm.xml')
        print("SVM training completed!")


def run_svm(h):
    if not os.path.isfile('svm.xml'):
        print("No trained svm available!")
        exit(1)

    svm = cv2.ml.SVM_load('svm.xml')
    h = np.array(h).reshape(1, h.shape[0]).astype(np.float32)

    return svm.predict(h)


