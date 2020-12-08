import glob
import numpy as np
from PIL import Image
from ipm import reverse_rectification, rectification
from mser import mser
from svm import hog, train_svm, run_svm, train_rf, run_rf, train_mlp, run_mlp
import cv2
import argparse
import time

label_dict = {}


# Need to split train and test data for each class
def train_test_split(images, labels, test_size=0.1):
    # Group by class
    imap = {}
    for idx, label in enumerate(labels):
        image = images[idx]
        key = label[0]
        if key in imap:
            imap[key].append(image)
        else:
            imap[key] = [image]

    # Split data in class to train and test data
    train_img = []
    test_img = []
    train_labels = []
    test_labels = []
    for k, v in imap.items():
        n = len(v)  # Number of data in the class
        test_num = int(n * test_size)
        train_num = n - test_num
        train_img = train_img + (v[:train_num])
        test_img = test_img + (v[train_num:])
        train_labels = train_labels + ([k] * train_num)
        test_labels = test_labels + ([k] * test_num)

    return train_img, test_img, train_labels, test_labels


def get_candidates(img):
    # Inverse Perspective Transform: Rectification
    rectified_img = rectification(img)

    # Maximally Stable Extremal Regions: Get candidate regions
    boxes = mser(rectified_img)

    return rectified_img, boxes


def train_images(labels, images, classifier):
    # images = images[:20] # Comment this for full training
    # labels = labels[:20] # Comment this for full training

    # Create trainable hog features
    H = np.zeros((len(images), 3780))
    skipped = []
    for idx, image in enumerate(images):
        img = cv2.imread(image, 3)
        rectified_img, boxes = get_candidates(img)

        if len(boxes) <= 0:
            skipped.append(idx)
            continue

        best_box = boxes[-1]
        (x, y, w, h) = best_box
        cv2.rectangle(rectified_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Image.fromarray(rectified_img).show()
        # print(rectified_img.shape)
        # time.sleep(0.2)

        # Histogram of Gradients
        if best_box is not None:
            h = hog(rectified_img, best_box)
            H[idx] = np.array(h).flatten()

    # Skip data with no candidates
    _labels = []
    _H = []
    for idx, (label, h_element) in enumerate(zip(labels, H)):
        if idx in skipped:
            continue
        _labels.append(label)
        _H.append(h_element)

    # Train SVM
    if classifier == "svm-linear":
        print("Training SVM linear")
        train_svm(_labels, _H, "linear")
    elif classifier == "svm-rbf":
        print("Training SVM RBF")
        train_svm(_labels, _H, "rbf")
    elif classifier == "rf":
        print("Training Random Forest")
        train_rf(_labels, _H)
    elif classifier == "mlp":
        print("Training MLP")
        train_mlp(_labels, _H, len(label_dict))
    else:
        ValueError("Wrong classifier")
        exit(1)


def test_images(labels, images, classifier):
    global label_dict
    sum = 0

    for idx, image in enumerate(images):
        img = cv2.imread(image)
        correct_label = labels[idx]
        rectified_img, boxes = get_candidates(img)

        # Draw boxes and labels around ROI
        img_clone = rectified_img.copy()
        answers = []
        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(img_clone, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Run classifier
            h = hog(rectified_img, box)
            output = 0
            if classifier == "svm-linear" or classifier == "svm-rbf":
                output = int(run_svm(h)[1][0][0])
            elif classifier == "rf":
                output = int(run_rf(h)[1][0][0])
            elif classifier == "mlp":
                output = int(run_mlp(h))
            else:
                ValueError("Wrong classifier")
                exit(1)
            output_str = label_dict[output]
            answers.append(output)
            cv2.putText(img_clone, output_str, (x, y - 3), 3, 1, (0, 255, 0), 2, cv2.LINE_AA)

        Image.fromarray(img_clone).show()

        if correct_label in answers:
            sum += 1

        # Reverse Rectification with labels
        # img_clone = reverse_rectification(img_clone)
        # Image.fromarray(img_clone).show()

    # Evaluation
    print("Test Accuracy: {:6}".format(sum / len(labels)))


def main(args):
    # Load multiple images
    image_path = "RoadMarkingDataset2/*.jpg"
    images = sorted(glob.glob(image_path))

    # Load labels
    labels = np.zeros((len(images), 1)).astype(np.int32)
    global label_dict
    with open("dataset_annotations.txt", 'r') as f:
        imgIdx = 0
        for text in f.readlines():
            data_label = text.split(',')[8]  # ex) 'left_turn', '40', ...
            file_name = "RoadMarkingDataset2/{}".format(text.split(',')[9]).replace('.png', '.jpg').rstrip()

            if imgIdx >= len(images):
                break

            if file_name != images[imgIdx]:
                continue

            value_exists = False
            for k, v in label_dict.items():
                if v == data_label:
                    labels[imgIdx] = k
                    value_exists = True
            if not value_exists:
                key = len(label_dict)
                label_dict[key] = data_label
                labels[imgIdx] = key

            imgIdx += 1

    # Split data into train and test data (90, 10)
    train_img, test_img, train_labels, test_labels = train_test_split(images, labels)

    # Start training or testing images
    if args.options == "train":
        print("Training Images")
        train_images(train_labels, train_img, args.classifier)
    elif args.options == "test":
        print("Testing Images")
        test_images(test_labels, test_img, args.classifier)
    else:
        raise ValueError("Invalid Options")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Road Marking Classification")
    parser.add_argument('--options', type=str, default="train", choices=["train", "test"], help="Train or test")
    parser.add_argument('--classifier', type=str,
                        default="svm-linear",
                        choices=["svm-linear", "svm-rbf", "rf", "mlp"],
                        help="SVM, Random Forest, MLP")

    args = parser.parse_args()
    main(args)
