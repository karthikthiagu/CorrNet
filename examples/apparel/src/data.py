import os, sys
from random import shuffle

import numpy as np
import cv2


def addNoise(noise_typ, image):
    row, col, ch = image.shape

    if  "gauss" in noise_typ:
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(loc = mean, scale = sigma, size = (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_image = image + gauss
        image = noisy_image

    if "blur" in noise_typ:
        noisy_image = image
        min_dim = min(row, col)
        blur_img = noisy_image
        for i in range(50):
            x_patch, y_patch = np.random.randint(0, min_dim - 5, 2)
            patch_dim = np.random.randint(5, min(min_dim - x_patch, min_dim - y_patch))
            kernel_dim = np.random.randint(2, 7)
            blur_img[x_patch : x_patch + patch_dim, y_patch : y_patch + patch_dim, :] = cv2.blur(\
                                                                                 blur_img[x_patch : x_patch + patch_dim, y_patch : y_patch + patch_dim, :],\
                                                                                 (kernel_dim, kernel_dim))
        noisy_image = blur_img

    if "s&p" in noise_typ:
        s_vs_p = 0.1
        amount_salt   = 0.9
        amount_pepper = 0.9
        if 'blur' not in noise_typ:
            noisy_image = image
        # Salt mode
        num_salt = np.ceil(amount_salt * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount_pepper * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords] = 0



    return noisy_image

# Write data to disk
def writeData(images_labels = [], path = None, resize = False, images_labels_filename = None, noise = False):
    print 'Writing data to disk'
    if not os.path.isdir(path):
        os.mkdir(path)
    # Load images_labels file to write contents of the train/val/test folders and labels
    images_labels_file = open(images_labels_filename, 'w')
    # Dict to help in naming the images
    label_dict = dict()
    # Write images to disk; wirte image paths and corresponding labels to images_labels_filename
    for image_path_input, label in images_labels:
        if label not in label_dict:
            label_dict[label] = 0
        label_dict[label] += 1
        image = cv2.imread(image_path_input)
        image = addNoise('s&p+blur', image) if noise == True else image
        resized_image = cv2.resize(image, (99, 99))
        image_path_output = '%s/%s_%s.jpg' % (path, label, str(label_dict[label]))
        cv2.imwrite(image_path_output, resized_image)
        images_labels_file.write('%s %s\n' % (image_path_output, label))
    images_labels_file.close()

# Prepare data
def prepareData(metadata_filename, train_path, val_path, test_path, noise = False):
    print 'Preparing data'
    # File containing images and labels seperated by ' ' as delimiter
    metadata_file = open(metadata_filename, 'r')
    images_labels = [line.strip().split() for line in metadata_file.readlines()]
    metadata_file.close()
    # Dict to store images corresponding to labels
    label_dict = dict()
    # Populate the dict
    for image, label in images_labels:
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(image)
    # We are going to work with these 10 labels that have at least 4000 images each
    permissible_labels = ['1', '2', '3' , '4', '6', '8', '9', '10', '12', '13']
    # Rename the labels for the above labels
    permissible_keys   = {permissible_labels[i] : str(i) for i in range(len(permissible_labels))}
    # Containers for train, val and test
    train, val, test = [], [], []
    # Populate train, val and test
    for label in permissible_labels:
        shuffle(label_dict[label])
        test  += zip(label_dict[label][     : 1000], [permissible_keys[label]]*1000)
        val   += zip(label_dict[label][1000 : 2000], [permissible_keys[label]]*1000)
        train += zip(label_dict[label][2000 : 4000], [permissible_keys[label]]*2000)
    # Write data to disk; resize the images if required
    '''
    writeData(images_labels = train, path = train_path, resize = True, images_labels_filename = train_path  + '.txt', noise = noise)
    writeData(images_labels = val  , path = val_path  , resize = True, images_labels_filename = val_path + '.txt', noise = noise)
    writeData(images_labels = test , path = test_path , resize = True, images_labels_filename = test_path + '.txt', noise = noise)
    '''
    writeData(images_labels = [tuple(line.strip().split(' ')) for line in open('./data/train.txt').readlines()], path = train_path, resize = True, images_labels_filename = train_path  + '.txt', noise = noise)
    writeData(images_labels = [tuple(line.strip().split(' ')) for line in open('./data/valid.txt').readlines()]  , path = val_path  , resize = True, images_labels_filename = val_path + '.txt', noise = noise)
    writeData(images_labels = [tuple(line.strip().split(' ')) for line in open('./data/test.txt').readlines()] , path = test_path , resize = True, images_labels_filename = test_path + '.txt', noise = noise)

def loadFeatures(features_filename = None, labels_filename = None):
    features_strings = [line.strip()[:-1].split(',') for line in open(features_filename, 'r').readlines()]
    features = []
    for line in features_strings:
        features.append([float(word) for word in line])
    labels = [int(line.strip().split(' ')[1]) for line in open(labels_filename, 'r').readlines()]
    return (np.asarray(features), np.asarray(labels))

def writeLabels(labels_input_filename = None, labels_output_filename = None):
    labels = [int(line.strip().split(' ')[1]) for line in open(labels_input_filename, 'r').readlines()]
    labels_file = open(labels_output_filename, 'w')
    for label in labels:
        labels_file.write('%s\n' % str(label))
    labels_file.close()

if __name__ == '__main__':

    writeLabels(labels_input_filename = sys.argv[1], labels_output_filename = sys.argv[2])
    #    prepareData(metadata_filename = './data/metadata/images_labels.txt',\
    #                train_path = './data/train_noisy',\
    #                val_path   = './data/valid_noisy',\
    #                test_path  = './data/test_noisy',\
    #                noise = True)
