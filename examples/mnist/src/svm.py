import sys

from random import shuffle
import numpy as np
from sklearn.svm import SVC

from sklearn.externals import joblib

from data import *

import time

class svm:

# Initialize classifier
    def __init__(self):   
        self.clf = SVC()
        self.train_data = None
        self.val_data   = None
        self.test_data  = None

    def setData(self, data):
        self.train_data, self.val_data, self.test_data = data

    def trainModel(self, train_data):
        # Train classifier
        self.clf.fit(train_data[0], train_data[1])

    def saveModel(self, filename):
        joblib.dump(self.clf, '%s' % filename)

    def loadModel(self, filename):
        self.clf = joblib.load('%s' % filename)

    def testModel(self, data):
        return np.sum(self.clf.predict(data[0]) == data[1])/float(data[1].shape[0])*100

    def tuneHyperparameters(self, best_model_filename):
        best_acc = -np.inf
        self.clf.C = 2**(0)
        for c_iter in range(4):
            self.clf.gamma = 2**(-4)
            for gamma_iter in range(6):
                start = time.clock()
                print 'Training SVM for C = %s, gamma = %s' % (str(self.clf.C), str(self.clf.gamma))
                self.trainModel(self.train_data)
                acc = self.testModel(self.val_data)
                if acc > best_acc:
                    print 'Better model found with accuracy = %s %%. Previous best = %s %%' % (str(acc), str(best_acc))
                    best_acc = acc
                    best_C   = self.clf.C
                    best_gamma = self.clf.gamma
                    self.saveModel(best_model_filename)
                self.clf.gamma *= 2**(0.5)
                print 'Time taken = %s seconds' % str(time.clock() - start)
            self.clf.C *= 2**(0.5)
        self.loadModel(best_model_filename)
        print 'Test accuracy on best model = %s %%' % str(self.testModel(self.test_data))
        
    # N = Number of data points
    def getCrossValidationAccuracy(self, view1_data, view2_data, cross_val_num = 5):
        N = view1_data[1].shape[0]
        N_split = int(N/float(cross_val_num))
        view1_accuracies = []
        view2_accuracies  = []
        for split_index in range(cross_val_num):
            print 'Training fold %s' % str(split_index + 1)
            view2_test_data = view2_data[0][split_index*N_split : (split_index + 1)*N_split]
            view2_test_labels = view2_data[1][split_index*N_split : (split_index + 1)*N_split]
            view1_test_data = view1_data[0][split_index*N_split : (split_index + 1)*N_split]
            view1_test_labels = view1_data[1][split_index*N_split : (split_index + 1)*N_split]

            train_data = np.concatenate((view1_data[0][ : split_index*N_split], view1_data[0][(split_index + 1)*N_split : ]), axis = 0)
            train_labels = np.concatenate((view1_data[1][ : split_index*N_split], view1_data[1][(split_index + 1)*N_split : ]), axis = 0)
            self.trainModel(tuple([train_data, train_labels]))

            view1_accuracies.append(self.testModel(tuple([view1_test_data, view1_test_labels])))
            view2_accuracies.append(self.testModel(tuple([view2_test_data, view2_test_labels])))

        print 'Average accuracy on 5 folds for view - 1 = %s' % str(np.mean(view1_accuracies))
        print 'Average accuracy on 5 folds for view - 2 = %s' % str(np.mean(view2_accuracies))
        
def shuffleIfNeedBe():
    zipped_data = zip(train_data, labels)
    shuffle(zipped_data)
    train_data, labels = zip(*zipped_data)
    train_data = np.asarray(train_data)
    labels = np.asarray(labels)

def trainViews(target_path, exp_number):

    print 'Loading data'
    train_view1_data, val_view1_data, test_view1_data, train_view2_data, val_view2_data, test_view2_data = loadRepresentationData(target_path)
    #########################################################
    svmobj = svm()
    print 'Training for view - 1'
    svmobj.setData((tuple([train_view1_data[0][:10000], train_view1_data[1][:10000]]), val_view1_data, test_view1_data))
    svmobj.tuneHyperparameters('models/exp_4/TGT_%s/best_model_exp_%s_view_1.pkl' % (str(exp_number), str(exp_number)))
    #svmobj.loadModel('models/exp_3/models/best_model_exp_3_view_1_baseline.pkl') #% (str(exp_number), str(exp_number)))
    print 'Single view accuracy for view 1 = %s' % str(svmobj.testModel(test_view1_data))
    print 'Cross  view accuracy for view 1 = %s' % str(svmobj.testModel(test_view2_data))
    ##########################################################
    svmobj = svm()
    print 'Training for view - 2'
    svmobj.setData((tuple([train_view2_data[0][:10000], train_view2_data[1][:10000]]), val_view2_data, test_view2_data))
    svmobj.tuneHyperparameters('models/exp_4/TGT_%s/best_model_exp_%s_view_2.pkl' % (str(exp_number), str(exp_number)))
    #svmobj.loadModel('models/exp_2/models/best_model_exp_2_view_2_baseline.pkl')# % (str(exp_number), str(exp_number)))
    print 'Single view accuracy for view - 2 = %s' % str(svmobj.testModel(test_view2_data))
    print 'Cross  view accuracy for view - 2 = %s' % str(svmobj.testModel(test_view1_data))


if __name__ == '__main__':

    '''
    if len(sys.argv) < 3:
        print 'Enter target path for projected data, and the experiment number'
        exit()
    _, target_path, exp_number = sys.argv
    '''
    #trainViews(target_path, exp_number)
    
    #train_view1_data, val_view1_data, test_view1_data, train_view2_data, val_view2_data, test_view2_data = loadRepresentationData(target_path)
    #train_view1_data, val_view1_data, test_view1_data, train_view2_data, val_view2_data, test_view2_data = readDataText(base_path = target_path)
    #test_view1_data, test_view2_data = loadRepresentationDataPairs(target_path)
    
    test_view1_data = (np.load(sys.argv[1] + 'view1.npy'), np.load(sys.argv[1] + 'labels.npy').flatten())
    test_view2_data = (np.load(sys.argv[1] + 'view2.npy'), np.load(sys.argv[1] + 'labels.npy').flatten())
    svmobj = svm()
    print 'View - 1 self and cross accuracies'
    svmobj.getCrossValidationAccuracy(test_view1_data, test_view2_data)
    print 'View - 2 self and cross accuracies'
    svmobj.getCrossValidationAccuracy(test_view2_data, test_view1_data)

