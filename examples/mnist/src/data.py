# Data for MNIST

import os, sys

import gzip
import cPickle as pkl

import numpy as np
import cv2
#import h5py

from random import shuffle

def transformImage(features_image):
    image = features_image.reshape((28, 28))
    row, col = image.shape
    cw_vs_acw = 1 if np.random.randint(2) else -1
    angle = np.random.uniform(10, 30)
    rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((row/2, col/2), cw_vs_acw*angle, 1.0), image.shape)
    tx, ty = np.random.uniform(2, 4, 2)
    tx = tx if np.random.randint(2) else -tx
    ty = ty if np.random.randint(2) else -ty
    trans_rotated_image = cv2.warpAffine(rotated_image, np.float32([[1, 0, tx], [0, 1, ty]]), image.shape)
    transformed_image_features = trans_rotated_image.flatten()
    return transformed_image_features

def addLayeredNoise(data, scale = 0.8):
    for i in range(4):
        layer_data = np.zeros(data.shape)
        layer_data[:, :] = data[:, :]
        np.random.shuffle(layer_data)
        data += scale * layer_data
        scale /= 2
    return data

def addNoiseSingleImage(noise_typ, features_image):
    image = features_image.reshape(28, 28)
    row, col = image.shape

    image = cv2.resize(image, (14, 14))
    image = cv2.resize(image, (1000, 10))
    image = cv2.resize(image, (6, 12))
    image = cv2.resize(image, (28, 28))
    noisy_image = image
    #cv2.imshow('win', image)
    #cv2.waitKey(0)

    if  "gauss" in noise_typ:
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(loc = mean, scale = sigma, size = (row,col))
        gauss = gauss.reshape(row, col)
        noisy_image = image + gauss

    if "s&p" in noise_typ:
        s_vs_p = 0.1
        amount_salt   = 0.3
        amount_pepper = 0.7
        
        # Salt mode
        num_salt = np.ceil(amount_salt * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount_pepper * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords] = 0


    if "blur" in noise_typ:
        min_dim = min(row, col)
        blur_img = noisy_image
        for i in range(20):
            x_patch, y_patch = np.random.randint(0, min_dim - 5, 2)
            patch_dim = np.random.randint(5, min(min_dim - x_patch, min_dim - y_patch))
            kernel_dim = np.random.randint(2, 4)
            blur_img[x_patch : x_patch + patch_dim, y_patch : y_patch + patch_dim] = cv2.blur(\
                                                                                     blur_img[x_patch : x_patch + patch_dim, y_patch : y_patch + patch_dim],\
                                                                                     (kernel_dim, kernel_dim))
        noisy_image = blur_img



    noisy_features_image = noisy_image.flatten()
    return noisy_features_image

        
# Add noise
def addNoise(noise_typ, features_images):

    for features_image in features_images:
        yield addNoiseSingleImage(noise_typ, features_image)

# Load MNIST data
def loadData(noise = False, transform = False, layer = False):
    data = pkl.load(gzip.open('data/mnist.pkl.gz', 'rb')) ###
    train_data, val_data, test_data = data
    
    def layeredNoiseAddKaro(data):
        layered_data = []
        for stage in data:
            layered_images = addLayeredNoise(stage[0])
            labels = stage[1]
            layered_data.append(tuple([layered_images, labels]))
        return tuple(layered_data)

    def noiseAddKaro(data):
        noisy_data = []
        # stage is train, val or test
        for stage in data:
            noisy_features_images = np.zeros(stage[0].shape)
            for image_count, noisy_features_image in enumerate(addNoise('s&p', stage[0])):
                noisy_features_images[image_count, :] = noisy_features_image[:]
            labels = stage[1]
            noisy_data.append(tuple([noisy_features_images, labels]))
        return tuple(noisy_data)   

    def transformKaro(data):
        transformed_data = []
        for stage in data:
            transformed_features_images = np.zeros(stage[0].shape)
            for image_count, features_image in enumerate(stage[0]):
                transformed_features_images[image_count, :] = transformImage(features_image)
            labels = stage[1]
            transformed_data.append(tuple([transformed_features_images, labels]))
        return tuple(transformed_data)

    if layer == True and noise == True:
        return noiseAddKaro(layeredNoiseAddKaro(data))
    elif layer == True:
        return layeredNoiseAddKaro(data)
    if noise == True and transform == True:
        return noiseAddKaro(transformKaro(data))
    elif noise == True:
        return noiseAddKaro(data)
    elif transform == True:
        return transformKaro(data)
    else:
        return (train_data, val_data, test_data)

# Write data to local disk
def writeData(data, base_path = 'data/'):
    for stage_name, stage in data.iteritems():
        if stage_name not in os.listdir(base_path):
            os.mkdir(os.path.join(base_path, stage_name))
        for count, img in enumerate(stage[0]):            
            cv2.imwrite('data/%s/%s_%s.jpg' % (stage_name ,str(stage[1][count]), str(count)), img.reshape((28, 28)) * 255)

# Write data to text files
def writeDataText(data, base_path = 'data/'):
    for stage_name, stage in data.iteritems():
        features_file = open('%s_features.txt' % os.path.join(base_path, stage_name), 'w')
        labels_file   = open('%s_labels.txt'   % os.path.join(base_path, stage_name), 'w')
        for count, img in enumerate(stage[0]):
            features_file.write('%s\n' % ' '.join([str(word) for word in img]))
            labels_file.write('%s\n'   % str(stage[1][count]))
        features_file.close()
        labels_file.close()

def readData(base_path = '/home/karthik/mnist/models/exp_1/gaussian_sp_blur/SRC/matpic'):
    return_views = []
    for view in [1, 2]:
        for stage in ['train', 'valid', 'test']:
            return_views.append(tuple([np.load(os.path.join(base_path, '%s/view%s.npy' % (stage, str(view)))),\
                                    np.load(os.path.join(base_path, '%s/labels.npy' % stage)).flatten()]))
    return return_views

def readWeights(base_path = None):
    return {'W_left' : np.load(os.path.join(base_path, 'W_left.npy')), 'W_right' : np.load(os.path.join(base_path, 'W_right.npy')),\
            'b' : np.load(os.path.join(base_path, 'b.npy'))}

def loadCaffeData(data, path):
    file_size = 1000
    data_dict = {'train' : (data[0], data[3]), 'val' : (data[1], data[4]), 'test' : (data[2], data[5])}
    for stage, data in data_dict.iteritems():
        metadata_file = open('%s/%s.txt' % (path, stage), 'w')
        print 'creating %s hdf5 files' % stage
        if stage not in os.listdir(path):
            os.mkdir(os.path.join(path, stage))
        stage_path = os.path.join(path, stage)
        number_of_files = data[0][0].shape[0]/file_size
        for file_num in xrange(number_of_files):
            filename = '%s/%s_%s.h5' % (stage_path, stage, str(file_num))
            metadata_file.write('%s\n' % os.path.join(os.getcwd(), filename))
            dataset = h5py.File(filename, 'w')
            dataset.create_dataset(name = 'x_1',\
                                   shape = data[0][0][file_num * file_size : (file_num + 1) * file_size].shape,\
                                   dtype = data[0][0].dtype,\
                                   data =  data[0][0][file_num * file_size : (file_num + 1) * file_size])
            dataset.create_dataset(name = 'x_2',\
                                   shape = data[1][0][file_num * file_size : (file_num + 1) * file_size].shape,\
                                   dtype = data[1][0].dtype,\
                                   data =  data[1][0][file_num * file_size : (file_num + 1) * file_size])           
            dataset.create_dataset(name = 'y',\
                                   shape = data[0][1][file_num * file_size : (file_num + 1) * file_size].shape,\
                                   dtype = data[0][1].dtype,\
                                   data =  data[0][1][file_num * file_size : (file_num + 1) * file_size])
            dataset.close()
        print 'Created %s hdf5 files for %s' % (str(file_num + 1), stage)
        metadata_file.close()

    print 'Created Caffe Data'     

def loadPairViews(noise = False, data = None):
    if data == None:
        train_data, val_data, test_data = loadData(noise = False)
    else:
        train_data_1, val_data_1, test_data_1, train_data_2, val_data_2, test_data_2 = data 
    data_stage = {'train' : (train_data_1, train_data_2), 'val' : (val_data_1, val_data_2), 'test' : (test_data_1, test_data_2)}

    view1, view2 = ({}, {})
    for stage in data_stage.keys():
        view1_features, view1_labels, view2_features, view2_labels = ([], [], [], [])
        data = data_stage[stage]
        data_dict_1 = {}
        data_dict_2 = {}
        for count, label in enumerate(data[0][1]):
            if label not in data_dict_1:
                data_dict_1[label] = [ 0, [] ]
                data_dict_2[label] = [ 0, [] ]
            data_dict_1[label][0] += 1
            data_dict_1[label][1].append(data[0][0][count])
            data_dict_2[label][0] += 1
            data_dict_2[label][1].append(data[1][0][count])
   
        for label in data_dict_1.keys():
            partition = data_dict_1[label][0]/2
            view1_features += data_dict_1[label][1][ : partition]
            view2_features += data_dict_2[label][1][partition : 2*partition]
            view1_labels += [label]*partition
            view2_labels += [label]*partition

        shuffler = zip(view1_features, view1_labels, view2_features, view2_labels)
        shuffle(shuffler)
        view1_features, view1_labels, view2_features, view2_labels = zip(*shuffler)
        
        view1_features = np.asarray(view1_features)
        view1_labels = np.asarray(view1_labels)
        view2_features = np.asarray(view2_features)
        view2_labels = np.asarray(view2_labels)
       
        if noise == True:
            # Randomly chose view1 or view2; Add noise to that image
            for i in range(view1_features.shape[0]):
                choice = np.random.randint(2)
                choice = 0
                if choice == 0:
                    view1_features[i, :] = addNoiseSingleImage('', view1_features[i, :])
                else:
                    view2_features[i, :] = addNoiseSingleImage('gauss', view2_features[i, :])

        view1[stage] = tuple([np.asarray(view1_features), np.asarray(view1_labels)])
        view2[stage] = tuple([np.asarray(view2_features), np.asarray(view2_labels)])
    
        

    view1 = tuple([view1['train'], view1['val'], view1['test']])
    view2 = tuple([view2['train'], view2['val'], view2['test']])

    return tuple([view1, view2])


# Program entry point
if __name__ == '__main__':

    #loadCaffeData(readData(), './data/caffe/mmae')
    train, val, test = loadData(noise = True, transform = False, layer = False)
    #writeData({'train' : train, 'valid' : val, 'test' : test})
    writeDataText({'train_view2' : train, 'valid_view2' : val, 'test_view2' : test}, sys.argv[1])
