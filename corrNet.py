__author__ = 'Sarath'

import os, json, sys
import time
import pickle

from Optimization import *
from Initializer import *
from NNUtil import *


import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class CorrNet(object):

    def init(self, network, x_left = None, x_right = None, load = False):

        self.network = network

        numpy_rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # Random number generators
        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        # Optimization metadata
        self.optimizer = get_optimizer(network['optimization'], network['learning_rate'])  # Optimizer-object, with containers for weights, learning rate (shared)

        # Initialization
        self.Initializer = Initializer(self.numpy_rng)

        # Set inputs
        if x_left != None:
            self.x_left = x_left
        else:
            self.x_left = T.matrix('x_left')
        if x_right != None:
            self.x_right = x_right
        else:
            self.x_right = T.matrix('x_right')
        self.y_proj = T.matrix('y_proj')

        self.params = {}
        preload_params = {}
        # Initialize/Load params
        for layer_name, layer in network['layers'].iteritems():
            # Initialize/Load params
            for param in layer['params']:
                if param + '.npy' in os.listdir(network['params_folder']):
                    print 'loaded %s' % param
                    preload_params = os.path.join(network['params_folder'], param)
                else:
                    preload_params = None
                if layer_name == 'layer_1' and load == False:
                    preload_params = None
                # Encode weights
                if 'W' in param and 'prime' not in param:
                    self.params[param] = self.Initializer.fan_based_sigmoid(param, preload_params,\
                                                                            layer['n_in_%s' % param[-1]], layer['n_out'])
                    self.optimizer.register_variable(param, layer['n_in_%s' % param[-1]], layer['n_out'])
                # Decode weights
                if 'W_prime' in param:
                    self.params[param] = self.params[param[0] + 'W_' + param[-1]].T
                # Decod Biases
                if 'b_prime' in param:
                    self.params[param] = self.Initializer.zero_vector(param, preload_params,\
                                                                     layer['n_in_%s' % param[-1]])
                    self.optimizer.register_variable(param, 1, layer['n_in_%s' % param[-1]])
                # Encode bias
                if 'b' in param and 'prime' not in param:
                    self.params[param] = self.Initializer.zero_vector(param, preload_params,\
                                                                     layer['n_out'])
                    self.optimizer.register_variable(param, 1, layer['n_out'])

        self.proj_from_left = theano.function([self.x_left], self.project_from_left())
        self.proj_from_right = theano.function([self.x_right], self.project_from_right())
        self.recon_left = theano.function([self.y_proj], self.reconstruct_left())
        self.recon_right = theano.function([self.y_proj], self.reconstruct_right())

    def train(self):

        y1 = self.x_left
        y2 = self.x_right
        for layer_count in reversed(range(self.network['num_layers'])):
            if layer_count + 1 == 1:
                y3 = activation(T.dot(y1, self.params[str(layer_count + 1) + 'W_1']) +\
                                T.dot(y2, self.params[str(layer_count + 1) + 'W_2']) +\
                                self.params[str(layer_count + 1) + 'b'],\
                                self.network['activation'])
            y1 = activation(T.dot(y1, self.params[str(layer_count + 1) + 'W_1']) + self.params[str(layer_count + 1) + 'b'], self.network['activation'])
            y2 = activation(T.dot(y2, self.params[str(layer_count + 1) + 'W_2']) + self.params[str(layer_count + 1) + 'b'], self.network['activation'])
        
        [z1_left, z2_left, z3_left] = [y1, y2, y3]
        [z1_right, z2_right, z3_right] = [y1, y2, y3]
        for layer_count in range(self.network['num_layers']):
            z1_left = activation(T.dot(z1_left, self.params[str(layer_count + 1) + 'W_prime_1']) + self.params[str(layer_count + 1) + 'b_prime_1'],\
                                self.network['activation'])
            z1_right = activation(T.dot(z1_right, self.params[str(layer_count + 1) + 'W_prime_2']) + self.params[str(layer_count + 1) + 'b_prime_2'],\
                                self.network['activation'])
            z2_left = activation(T.dot(z2_left, self.params[str(layer_count + 1) + 'W_prime_1']) + self.params[str(layer_count + 1) + 'b_prime_1'],\
                                self.network['activation'])
            z2_right = activation(T.dot(z2_right, self.params[str(layer_count + 1) + 'W_prime_2']) + self.params[str(layer_count + 1) + 'b_prime_2'],\
                                self.network['activation'])
            z3_left = activation(T.dot(z3_left, self.params[str(layer_count + 1) + 'W_prime_1']) + self.params[str(layer_count + 1) + 'b_prime_1'],\
                                self.network['activation'])
            z3_right = activation(T.dot(z3_right, self.params[str(layer_count + 1) + 'W_prime_2']) + self.params[str(layer_count + 1) + 'b_prime_2'],\
                                self.network['activation'])

        y1_mean = T.mean(y1, axis=0)
        y1_centered = y1 - y1_mean
        y2_mean = T.mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = T.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0)+1e-8)
        corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0)+1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr/corr_dr
        
        # Mean is for the batch
        recon_loss = {'view1_self'  : T.mean(loss(z1_left, self.x_left, self.network['loss_function'])),\
                      'view1_cross' : T.mean(loss(z1_right, self.x_right, self.network['loss_function'])),\
                      'view2_self'  : T.mean(loss(z2_right, self.x_right, self.network['loss_function'])),\
                      'view2_cross' : T.mean(loss(z2_left, self.x_left, self.network['loss_function'])),\
                      'view12_left' : T.mean(loss(z3_left, self.x_left, self.network['loss_function'])),\
                      'view12_right': T.mean(loss(z3_right, self.x_right, self.network['loss_function']))}
       
        # L1, L2 and L3 are the reconstruction losses, L4 is the correlation loss
        L1 = recon_loss['view1_self']  + recon_loss['view1_cross']
        L2 = recon_loss['view2_self']  + recon_loss['view2_cross']
        L3 = recon_loss['view12_left'] + recon_loss['view12_right']
        L4 = T.mean(T.sum(corr)) * self.network['lambda']

        # Sum correlation across all dimensions; mean is for the batch
        only_corr = T.mean(T.sum(corr))

        # Mean cost for this batch
        cost = L1 + L2 + L3 - L4
        
        self.param_values = []
        self.param_names  = []
        for param_name, param_value in self.params.iteritems():
            if 'W_prime' in param_name:
                continue
            self.param_values.append(param_value)
            self.param_names.append(param_name)
        gradients = T.grad(cost, self.param_values)
        updates = []
        for p,g,n in zip(self.param_values, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        #return cost, updates, T.mean(L1 + L2 + L3), T.mean(only_corr)
        return cost, updates, only_corr,\
               recon_loss['view1_self'], recon_loss['view1_cross'],\
               recon_loss['view2_self'], recon_loss['view2_cross'],\
               recon_loss['view12_left'], recon_loss['view12_right']

    def project_from_left(self):
        y1 = self.x_left
        for layer_count in reversed(range(self.network['num_layers'])):
            y1 = activation(T.dot(y1, self.params[str(layer_count + 1) + 'W_1']) + self.params[str(layer_count + 1) + 'b'], self.network['activation'])
        return y1

    def project_from_right(self):
        y2 = self.x_right
        for layer_count in reversed(range(self.network['num_layers'])):       
            y2 = activation(T.dot(y2, self.params[str(layer_count + 1) + 'W_2']) + self.params[str(layer_count + 1) + 'b'], self.network['activation'])
        return y2

    def reconstruct_left(self):
        z1 = self.y_proj
        for layer_count in range(self.network['num_layers']):
            z1 = activation(T.dot(z1, self.params[str(layer_count + 1) + 'W_prime_1']) + self.params[str(layer_count + 1) + 'b_prime_1'], self.network['activation'])
        return z1

    def reconstruct_right(self):
        z2 = self.y_proj
        for layer_count in range(self.network['num_layers']):
            z2 = activation(T.dot(z2, self.params[str(layer_count + 1) + 'W_prime_2']) + self.params[str(layer_count + 1) + 'b_prime_2'], self.network['activation'])
        return z2

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):
        print 'saving matrices'
        for p,nm in zip(self.param_values, self.param_names):
            numpy.save(os.path.join(self.network['params_folder'], nm), p.get_value(borrow=True))

def projectCorrNet(network):
    
    src_folder = os.path.join(network['data_folder'], 'matpic/')
    tgt_folder = network['params_folder']

    model = CorrNet()
    model.init(network, None, None, True)

    folder_project = os.path.join(tgt_folder, 'project')
    folder_reconstruct = os.path.join(tgt_folder, 'reconstruct')
    if not os.path.exists(folder_project):
        os.makedirs(folder_project)
    if not os.path.exists(folder_reconstruct):
        os.makedirs(folder_reconstruct)

    for stage in ['train', 'valid', 'test']:
        for view in ['view1', 'view2']:
            input_view = numpy.load(os.path.join(src_folder, '%s/%s.npy' % (stage, view)))
            projected_view = model.proj_from_left(input_view)
            reconstruct_left = model.recon_left(projected_view)
            reconstruct_right = model.recon_right(projected_view)
            numpy.save(os.path.join(tgt_folder, "project/%s-%s" % (stage, view)), projected_view)
            numpy.save(os.path.join(tgt_folder, 'reconstruct/%s-%s-left' % (stage, view)), reconstruct_left)
            numpy.save(os.path.join(tgt_folder, 'reconstruct/%s-%s-right' % (stage, view)), reconstruct_right)
        labels = numpy.load(os.path.join(src_folder, "%s/labels.npy" % stage))
        numpy.save(os.path.join(tgt_folder, "project/%s-labels" % stage), labels)

#  Entry point
def trainCorrNet(network):
    
    training_epochs = network['epochs']
    batch_size = network['batch_size']

    index = T.lscalar()
    x_left = T.matrix('x_left')
    x_right = T.matrix('x_right')

    model = CorrNet()
    model.init(network, x_left, x_right, load = network['load'])
    matpic_folder = os.path.join(network['data_folder'], 'matpic/')
    matpic1_folder = os.path.join(network['data_folder'], 'matpic1/')

    x_left_size = network['layers']['layer_' + str(network['num_layers'])]['n_in_1']
    x_right_size = network['layers']['layer_' + str(network['num_layers'])]['n_in_2']
    train_set_x_left = theano.shared(numpy.asarray(numpy.zeros((1000, x_left_size)), dtype=theano.config.floatX), borrow=True)
    test_set_x_left = theano.shared(numpy.asarray(numpy.load(matpic_folder + 'valid/view1.npy'), dtype = theano.config.floatX), borrow = True)

    train_set_x_right = theano.shared(numpy.asarray(numpy.zeros((1000, x_right_size)), dtype=theano.config.floatX), borrow=True)
    test_set_x_right = theano.shared(numpy.asarray(numpy.load(matpic_folder + 'valid/view2.npy'), dtype = theano.config.floatX), borrow = True)
    
    common_cost, common_updates, corr_val,\
    recon_view1_self, recon_view1_cross,\
    recon_view2_self, recon_view2_cross,\
    recon_view12_left, recon_view12_right = model.train()

    outputs = [ common_cost, corr_val,\
                recon_view1_self, recon_view1_cross,\
                recon_view2_self, recon_view2_cross,\
                recon_view12_left, recon_view12_right ]
    reconstruction_loss = outputs[2] + outputs[3] + outputs[4] + outputs[5] + outputs[6] + outputs[7]
    simple_outputs = [common_cost, corr_val, reconstruction_loss]

    mtrain = theano.function([index], simple_outputs, updates=common_updates, givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size]),(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])
    mtest_common = theano.function([], simple_outputs, givens=[(x_left, test_set_x_left),(x_right, test_set_x_right)])
    mtest_uncommon = theano.function([], outputs, givens=[(x_left, test_set_x_left),(x_right, test_set_x_right)]) 

    oldtc = float("inf")

    recon_list = ['view1_self', 'view1_cross', 'view2_self', 'view2_cross', 'view12_left', 'view12_right']
    for epoch in xrange(training_epochs):
        
        if (epoch + 1) % model.network['step_size'] == 0:
            model.set_lr_rate(numpy.asarray(model.get_lr_rate()*model.network['step_value'], dtype = float32))
        print "In epoch ", str(epoch + 1)
        c = []
        recon_loss_container = []
        corr_container = []
        left_cost_container = []
        ipfile = open(os.path.join(matpic1_folder, "train/ip.txt"),"r")
        for count, line in enumerate(ipfile):
            if count == 10:
                #print 'looking at only the first 20,000 points'
                break
            next = line.strip().split(",")
            if(next[0]=="xy"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                    denseTheanoloader(next[2]+"_right",train_set_x_right, "float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                    sparseTheanoloader(next[2]+"_right",train_set_x_right, "float32", 1000, n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    outputs = mtrain(batch_index)
                    c.append(outputs[0])
                    recon_loss_container.append(sum(outputs[2:]))
                    corr_container.append(outputs[1])
            elif(next[0]=="x"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_left(batch_index))
            elif(next[0]=="y"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_right",train_set_x_right,"float32")
                else:
                    sparseTheanoloader(next[2]+"_right",train_set_x_right,"float32",1000,n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_right(batch_index))

        print 'For train, Reconstruction loss = %s, correlation = %s' % (numpy.mean(recon_loss_container), numpy.mean(corr_container))
        test_outputs = mtest_common()
        #print 'Recostruction loss = %s, correlation = %s, test_cost = %s on validation data' % (str(test_recon), str(test_corr), str(test_cost))
        print '#######################'
        print 'Overall cost = %s, Correlation = %s, Reconstruction loss = %s' % (str(test_outputs[0]), str(test_outputs[1]), str(sum(test_outputs[2:])))
        #for loss_type, loss in zip(recon_list, test_outputs[2:]):
        #    print '%s : %s' % (loss_type, str(loss))
        print '#######################\n'           
        ipfile.close()
        if (epoch + 1) % model.network['step_size'] == 0:
            print 'dropping learning rate by a factor of %s after epoch %s' % (str(model.network['step_size']), str(epoch + 1))
            model.set_lr_rate(numpy.asarray(model.get_lr_rate()*model.network['step_value'], dtype = float32)) 
        # save the parameters for every 5 epochs
        if((epoch+1)%5==0):
            print 'saving models'
            model.save_matrices()

    model.save_matrices()
    ##### TESTING MODEL ######
    outputs = mtest_uncommon()
    print 'Final validation results'
    print 'Reconstruction loss = %s, Correlation = %s, Overall cost = %s' % (str(sum(outputs[2:])), str(outputs[1]), str(outputs[0]))
    output_file = open(os.path.join(model.network['params_folder'], 'results.txt'), 'w')
    output_file.write('Overall cost = %s, Reconstruction loss = %s, Correlation = %s\n' % (str(outputs[0]), str(sum(outputs[2:])), str(outputs[1])))
    for loss_type, loss in zip(recon_list, outputs[2:]):
        output_file.write('%s = %s\n' % (loss_type, str(loss)))
    output_file.close()
    ##########################

if __name__ == '__main__':
    start = time.time()
    _, state, network_file = sys.argv
    if state == 'train':
        trainCorrNet(json.load(open(network_file, 'r')))
    elif state == 'project':
        projectCorrNet(json.load(open(network_file, 'r')))
    end = time.time()
    print 'Time taken to run the experiment = %s' % str((end - start) / 60.0)
