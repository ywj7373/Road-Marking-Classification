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


def train_svm(labels, train_hog, kernel):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)

    if kernel == "linear":
        svm.setKernel(cv2.ml.SVM_LINEAR)
    elif kernel == "rbf":
        svm.setKernel(cv2.ml.SVM_RBF)
    else:
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


def train_rf(labels, train_hog):
    rf = cv2.ml.RTrees_create()
    rf.setMaxDepth(10)
    rf.setMinSampleCount(5)
    rf.setMaxCategories(15)

    train_hog = np.array(train_hog).astype(np.float32)
    labels = np.array(labels).reshape((-1, 1))

    start_time = time.time()
    if rf.train(train_hog, cv2.ml.ROW_SAMPLE, labels):
        train_preds = rf.predict(train_hog)[1]
        print("Elasped time: {:6}s".format(time.time() - start_time))
        print('Training Accuracy: %.6f' % np.average(train_preds == labels))
        rf.save('trees.xml')
        print("Random Forest training completed!")


def run_rf(h):
    if not os.path.isfile('trees.xml'):
        print("No trained random forest available!")
        exit(1)

    rf = cv2.ml.RTrees_load('trees.xml')
    h = np.array(h).reshape(1, h.shape[0]).astype(np.float32)

    return rf.predict(h)


def train_mlp(labels, train_hog):
    _, n = np.array(train_hog).shape
    mlp = cv2.ml.ANN_MLP_create()
    layer_sizes = np.int32([n, 100, 100, 13])

    mlp.setLayerSizes(layer_sizes)
    mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    mlp.setBackpropMomentumScale(0.0)
    mlp.setBackpropWeightScale(0.001)
    mlp.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

    train_hog = np.array(train_hog).astype(np.float32)
    labels = np.array(labels).reshape((-1, 1)).astype(np.float32)
    print(train_hog.shape)
    start_time = time.time()
    if mlp.train(train_hog, cv2.ml.ROW_SAMPLE, labels):
        train_preds = mlp.predict(train_hog)[1]
        print("Elasped time: {:6}s".format(time.time() - start_time))
        print('Training Accuracy: %.6f' % np.average(train_preds == labels))
        mlp.save('mlp.xml')
        print("MLP training completed!")


def run_mlp(h):
    if not os.path.isfile('mlp.xml'):
        print("No trained MLP available!")
        exit(1)

    mlp = cv2.ml.ANN_MLP_load('mlp.xml')
    h = np.array(h).reshape(1, h.shape[0]).astype(np.float32)

    return mlp.predict(h)
