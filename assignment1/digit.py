################################################################################
#                                                                              #
#                               INTRODUCTION                                   #
#                                                                              #
################################################################################

# In order to help you with the first assignment, this file provides a general
# outline of your program. You will implement the details of various pieces of
# Python code grouped in functions. Those functions are called within the main
# function, at the end of this source file. Please refer to the lecture slides
# for the background behind this assignment.
# You will submit three python files (sonar.py, cat.py, digits.py) and three
# pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain
# trained models for each tasks.
# Good luck!

################################################################################
#                                                                              #
#                                    CODE                                      #
#                                                                              #
################################################################################

import numpy as np
import pickle as pkl
import random

def sigmoid(z):

    return

def softmax(u):

    return

def cross_entropy_loss(yhat, y):

    return

def softmax_cross_entropy_error_back(y, t):

    return

def make_one_hot(d):
    # return a one-hot vector representation of digit d

    return

class Digit_Model:

    def __init__(self, dim_input=None, dim_hidden=None, dim_out=None, weights1=None, weights2=None, bias1=None, bias2=None, activation1=(lambda x: x), activation2=(lambda x: x)):

        self._dim_in = dimension
        self._dim_out= dimension
        self._dim_hid= dimension
        self.w1 = weights or np.random.normal(size=self._dim)
        self.w1 = np.array(self.w)
        self.w2 = weights or np.random.normal(size=self._dim)
        self.w2 = np.array(self.w)
        self.b1 = bias if bias is not None else np.random.normal()
        self.b2 = bias if bias is not None else np.random.normal()
        self._a1 = activation_1
        self._a2 = activation_2


    def __str__(self):
        
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
   
    def __call__(self, x):
        
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat


    def load_model(self, file_path):
        
        with open (file_path, mode ='rb') as f:
            ne = pkl.load(f)
        
        self._dim_in = ne_dim_in
        self._dim_out = ne_dim_out
        self._dim_hid = ne_dim_hid
        self.w1 = ne.w1
        self.w2 = ne.w2
        self.b1 = ne.b1
        self.b2 = ne.b2
        self._a1 = ne._a1
        self._a2 = ne._a2
        

    def save_model(self):
       
        f = open('digit_model.pkl','wb')
        pkl.dump(self, f)
        f.close



class Digit_Data:

    def __init__(self, data_file_name='digit_data.pkl'):
        
        self.index = -1
        with open ('%s' % (data_file_name), mode = 'rb') as f:
            digit_data = pkl.load(f)
        self.samples = [(np.reshape(vector, vector.size), 1) for vector in self.standardize(digit_data['train']['digit'])] + [(np.reshape(vector, vector.size),0) for vector in self.standardize(digit_data['train']['no_digit'])]
        random.shuffle(self.samples)

    def __iter__(self):
        
        return self

    def __next__(self):
        
        self.index += 1
        if self.index == len(self.samples):
            raise StopIteration
        """return self.samples[self.index][0], self.samples[self.index][1]"""

    def shuffle(self):
        
        random.shuffle(self.samples)
        
    def standardize(self, rgb_images):
        
        mean = np.mean(rgb_images, axis=(1, 2), keepdims=True)
        std = np.std(rgb_images, axis=(1, 2), keepdims=True)
        return (rgb_images - mean) / std

class Digit_Trainer:

    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = lrloss


    def accuracy(self, data):
        
        """return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])"""

    def train(self, lr, ne):
    
      
        print ("training model on data...")
    
        accuracy = self.accuracy(self.dataset)
        
        """print ("initial accuracy: %.3f" % (accuracy))"""
        
        costs = []
        accuracies = []
        
        for epoch in range(1, ne+1):
            
            self.dataset.shuffle()
            J = 0
            dw = 0
            for d in self.dataset.samples:
                xi, yi = d
                yhat = self.model(xi)
                J += self.loss(yhat, yi)
                dz = yhat -yi
                dw += xi*dz
            J /= len(self.dataset.samples)
            dw /= len(self.dataset.samples)
            self.model.w = self.model.w - lr*dw
            
            accuracy = self.accuracy(self.dataset)
            
            if epoch%10 ==0:
                """print('-->epoch=%d, accuracy=%.3f' % (epoch, accuracy))"""
            costs.append(J)
            accuracies.append(accuracy)
            
            
        print("training complete")
        costs = list(map(lambda t: np.mean(t), [np.array(costs)[i-10:i+11] for i in range(1, len(costs)-10)]))
        
        data = Digit_Data()
        model = Digit_Model(activation=sigmoid)  # specify the necessary arguments
        trainer = Digit_Trainer(data, model)
        costs, accuracies = trainer.train(0.000001, 500) # experiment with learning rate and number of epochs
        model.save_model()
        

        plt.plot(costs)

        plt.plot(accuracies)

"""def standardize(self, rgb_images):
        
        mean = np.mean(rgb_images, axis=(1, 2), keepdims=True)
        std = np.std(rgb_images, axis=(1, 2), keepdims=True)
        return (rgb_images - mean) / std
print(digit_example.min(), digit_example.max())"""
