from .userErrors import NotFittedError
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
sns.set_style("whitegrid")


class BinaryClassificationModel():
    
    '''
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    compute cost
    cost -- cross-entropy cost
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    Update parameters using gradient descent
    '''
    
    def __init__(self,layer_dims=[],learning_rate=0.05,n_iterations=10000,random_seed=101):
        '''
        Arguments:
        -----------
        layer_dims -- python array (list) containing the dimension of hidden layers in neural network
        learning_rate -- learning rate or step size of gradient descent, default = [] (that means 0 hidden layer and
                         one output layer with 1 neuron)
        n_iterations -- number of iterations, to optimize parameters w and b
        random_seed -- int, RandomState instance, optional (default=101)
                       used for randomly initialize parameter W
                       
        Parameters:
        ------------
        W,b
        
        Hyperparameters:
        ----------------
        postprocessing features that can be tuned to get more better results
    
        layer_dims, learning_rate, n_iterations
        
        example:
        --------
        >> nn_model = DeepNeuralNet([10,5,3], 0.05, 1000, 101)
        >> nn_model.fit(X_train, y_train)
        >> predictions = nn_model.predict(X_test)
        '''
        
        # initialize parameters dictionary
        self.parameters = {}
        
        # public instance member variables
        self.layer_dims    = layer_dims+[1]              # dimesion of layers       
        self.learning_rate = learning_rate               # learning rate used in gradient descent
        self.iterations    = n_iterations                # total number of iterations to optimize parameters w and b
        self.random_seed   = random_seed                 # random seed to generate consistent results 
        self.costs         = []                          # collection of computed cost after each 100 iterations
        
        # private instance member variables 
        self.__L__ = None                                # total number of layers excluding input layers                
        self.__n__ = None                                # python array (list) containing dimension of each layer   
        self.__fitted__ = False                          # check wether fit method is called or not                          
        self.__iterations_x__= []                        # collection of iteration value over 100 iterations
    def ReLU(self,data): 
        '''
        The rectified linear activation function or ReLU 
        piecewise linear function
        
        argument :
        --------
        data -- numpy array
         
        return :
        ------
        the input data directly if it is positive, otherwise, it will output zero.
        '''
        return np.maximum(0,data)
    
    def ReLU_backward(self,data):
        '''
        derivative of ReLU function 
        
        arguments:
        ----------
        data -- numpy array
        
        return:
        ---------
        return numpy array where data is greater then 0 with 1 else 0
        '''
        return np.where(data<=0,0,1)
    
    def sigmoid(self,data):
        '''
        sigmoid activation function
        
        arguments:
        ---------
        data -- numpy array
        
        returns:
        computed sigmoid value of data
        '''
        return 1/(1+np.exp(-data))
        
    def __w_b_Initializer__(self,L,n): 
        '''
        definition : private instance member function
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        for input layer there are no W and b parameters                
        '''
        
        np.random.seed(self.random_seed) 
        
        for i in range(1,L+1):
            w_i = np.random.randn(n[i],n[i-1])*np.sqrt(2/n[i-1])
            b_i = np.zeros((n[i],1))
            
            self.parameters['W'+str(i)] = w_i
            self.parameters['b'+str(i)] = b_i
    

    def fit(self, X, y):
        '''
        Arguments:
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        y : array-like or scaler of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary
        
        '''
        X = X.T
        if isinstance(y,(pd.Series)):
            y = y.values.reshape(1,-1)
        else:
            y = y.reshape(1,-1)
            
        
        self.__fitted__ = True

        n = [X.shape[0]]+self.layer_dims
        L = len(n)-1
        m = X.shape[-1]
        
        self.__n__ = n
        self.__L__ = L
        
        # randomly intitialize the parameters values
        self.__w_b_Initializer__(L,n)
        
        for iteration in range(self.iterations):
            # forward propogation
            forward_cache = {'A0':X}                               # holds forward linear units and linear activation units
            parameters = self.parameters
            
            for i in range(1,L+1):
                # compute linear unit for hidden layer_i
                forward_cache['Z'+str(i)] = np.dot(parameters["W"+str(i)],forward_cache["A"+str(i-1)])+parameters["b"+str(i)]         

                if i==L:
                    forward_cache['A'+str(i)] = self.sigmoid(forward_cache['Z'+str(i)])            # compute sigmoid activation unit at output layer
                else:
                    forward_cache['A'+str(i)] = self.ReLU(forward_cache['Z'+str(i)])               # compute relu activation unit at hidden layers

                assert(forward_cache['A'+str(i)].shape == (n[i],m))          # check wether shpae is correct or not after computaitons
                assert(forward_cache['Z'+str(i)].shape == (n[i],m))

                
            # computing cost
            if iteration%100 == 0:
                cost = (-1/m)*(np.dot(np.log(forward_cache["A"+str(L)]),y.T)+np.dot(np.log(1-forward_cache["A"+str(L)]),(1-y).T))
                cost = float(np.squeeze(cost))
                
                self.costs.append(cost)
                self.__iterations_x__.append(iteration)
                
                if iteration%1000==0:
                    print(f"cost after the iteration {iteration}:{cost}")
            
            # backward propagation
            gradients = {}
            backward_cache = {}
            
            for i in range(L,0,-1):
                if i==L:
                    dZ_i = forward_cache['A'+str(i)]-y
                else:
                    dZ_i = np.dot(parameters["W"+str(i+1)].T,backward_cache['dZ'+str(i+1)])*(self.ReLU_backward(forward_cache['Z'+str(i)]))
                
                assert(dZ_i.shape == (n[i],m))
                backward_cache['dZ'+str(i)] = dZ_i
                
                gradients['dW'+str(i)] = (1/m)*np.dot(backward_cache['dZ'+str(i)], forward_cache['A'+str(i-1)].T)
                gradients['db'+str(i)] = (1/m)*np.sum(backward_cache['dZ'+str(i)], axis=1, keepdims=True)
                
                assert(gradients['dW'+str(i)].shape == (n[i], n[i-1]))
                assert(gradients['db'+str(i)].shape == (n[i], 1))
                
                    
            # updating parameteres value
            for i in range(1,L+1):
                parameters['W'+str(i)] = parameters['W'+str(i)]-self.learning_rate*gradients['dW'+str(i)]
                parameters['b'+str(i)] = parameters['b'+str(i)]-self.learning_rate*gradients['db'+str(i)]

                assert(parameters['W'+str(i)].shape == (n[i],n[i-1]))
                assert(parameters['b'+str(i)].shape == (n[i],1))  
                
        self.parameters = parameters
    
    def costPlot(self,kind=1):
        '''plot cost vs iteration
        
        Arguments:
            kind -- int, default (kind=1), available choices for kind are {1, 2}
        '''
        if not self.__fitted__:
            raise NotFittedError()
        else:
            plt.figure(figsize=(12,4))
            if kind in range(1,3):
                if kind==1:
                    plt.plot(self.__iterations_x__,self.costs)
                else:
                    plt.plot(self.__iterations_x__[1:],self.costs[1:])
            else:
                warnings.warn("provide a valid argument value for the argument kind")
                plt.plot(self.__iterations_x__,self.costs)
        
    def predict(self, X):
        '''
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
        '''
        X = X.T
        if not self.__fitted__:
            raise NotFittedError()
        else:
            L,m,n = self.__L__,X.shape[-1],self.__n__

            a_i = X # for i==0 

            for i in range(1,L+1):
                z_i = np.dot(self.parameters['W'+str(i)],a_i)+self.parameters['b'+str(i)]

                if i==L:
                    a_i = self.sigmoid(z_i)
                else:
                    a_i = self.ReLU(z_i)

                assert(a_i.shape == (n[i],m))
                assert(z_i.shape == (n[i],m))


            predictions = a_i>0.5
            predictions = predictions.astype(int)
            predictions = np.squeeze(predictions.T,axis=1)
            return(predictions)
    
    def modelAccuracy(self,X_test,y_true):
        '''
        Arguments:
        ---------
        X_test -- array_like or sparse matrix, shape (n_samples, n_features)
                  Samples.
        
        y_true -- scaler of shape (n_samples,)
        '''
        if not self.__fitted__:
            raise NotFittedError()
        else:
            predictions = self.predict(X_test)
            if isinstance(y_true,(pd.Series)):
                y_true = y_true.values
            Y = y_true
            print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
            return float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)