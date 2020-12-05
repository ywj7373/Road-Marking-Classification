import glob
import numpy as np
from PIL import Image
from ipm import reverse_rectification, rectification
from mser import mser
from svm import hog, train_svm, run_svm
import cv2
import argparse
import time

label_dict = {}


# Need to split train and test data for each class
def train_test_split(images, labels, test_size=0.05, shuffle=True, random_state=42):
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
    # TODO: Need to remove glare

    # Inverse Perspective Transform: Rectification
    rectified_img = rectification(img)

    # Maximally Stable Extremal Regions: Get candidate regions
    boxes = mser(rectified_img)

    return rectified_img, boxes


def train_images(labels, images):
    # images = images[:20] # Comment this for full training
    # labels = labels[:20] # Comment this for full training

    # Create trainable hog features
    H = np.zeros((len(images), 3780))
    for idx, image in enumerate(images):
        img = cv2.imread(image, 3)
        rectified_img, boxes = get_candidates(img)

        best_box = None
        # TODO: Find representative box
        for box in boxes:
            (x, y, w, h) = box
            best_box = box # TODO: Fix this
            cv2.rectangle(rectified_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        Image.fromarray(rectified_img).show()
        print(rectified_img.shape)
        time.sleep(0.5)

        # Histogram of Gradients
        if best_box is not None:
            h = hog(rectified_img, best_box)
            H[idx] = np.array(h).flatten()

    # Train SVM
    train_svm(labels, H)


def test_images(labels, images):
    global label_dict

    for idx, image in enumerate(images):
        img = cv2.imread(image)
        correct_label = labels[idx]
        rectified_img, boxes = get_candidates(img)

        # Draw boxes and labels around ROI
        img_clone = rectified_img.copy()
        for box in boxes:
            (x, y, w, h) = box
            cv2.rectangle(img_clone, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Run classifier
            h = hog(rectified_img, box)
            output = int(run_svm(h)[1][0][0])
            output_str = label_dict[output]
            cv2.putText(img_clone, output_str, (x, y-3), 3, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # TODO: Evaluation

        # Reverse Rectification with labels
        Image.fromarray(img_clone).show()
        img_clone = reverse_rectification(img_clone)
        # Image.fromarray(img_clone).show()


def main(args):
    # Load multiple images
    image_path = "RoadMarkingDataset/*.jpg"
    images = sorted(glob.glob(image_path))

    # Load labels (shape = (1443, 1))
    labels = np.zeros((1443, 1)).astype(np.int32)
    global label_dict
    with open("dataset_annotations.txt", 'r') as f:
        for idx, text in enumerate(f.readlines()):
            data_label = text.split(',')[8]  # ex) 'left_turn', '40', ...
            value_exists = False
            for k, v in label_dict.items():
                if v == data_label:
                    labels[idx] = k
                    value_exists = True
            if not value_exists:
                key = len(label_dict)
                label_dict[key] = data_label
                labels[idx] = key

    # Split data into train and test data (95, 5)
    train_img, test_img, train_labels, test_labels = train_test_split(images, labels)

    # Start training or testing images
    if args.options == "train":
        print("Training Images")
        train_images(train_labels, train_img)
    elif args.options == "test":
        print("Testing Images")
        test_images(test_labels, test_img)
    else:
        raise ValueError("Invalid Options")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Road Marking Classification")
    parser.add_argument('--options', type=str, default="train", choices=["train", "test"], help="Train or test")
    args = parser.parse_args()
    main(args)
