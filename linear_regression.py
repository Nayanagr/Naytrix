# ---------------------------------Importing Required Modules---------------------------------------------------========#
import numpy as np


# --------------------------------Defining the Functionality------------------------------------------------------------#
class LinearRegression:
    '''
    Required input as numpy array
    '''
    #
    # def __int__(self):
    #     """Initiate weight and bias as beta"""
    #     self.beta = None
    #
    # def fit(self, x, y):
    #     # To add 1's column in each row.
    #     x_bias = np.c_[np.ones(x.shape[0]), x]
    #     # Use the Normal Equation
    #     self.beta = np.linalg.inv(x_bias.T @ x_bias)@x_bias.T@y
    #
    # def predict(self,x):
    #     x_b = np.c_[np.ones(x.shape[0]),x]
    #     return x_b@self.beta
    def __init__(self,l_r=0.01,itr=1000):
        self.l_r = l_r
        self.itr = itr
        self.weights= None
        self.bias = None
        self.data = {}
        self.checkpointlist = [self.itr, int(round(self.itr/10)), int(round(self.itr/5)), int(round(self.itr/2))]

    def fit(self,x,y):
        '''Fits the value of x and y array.'''
        n_rows , n_indep =x.shape
        self.weights= np.zeros(n_indep)
        self.bias = 0

        for i in range(1,self.itr+1):
            y_predict = np.dot(x, self.weights) + self.bias

            dw = (1/(2*n_rows)) * np.dot(x.T, (y_predict-y))
            db = (1/(2*n_rows)) * np.sum(y_predict-y)

            self.weights -= self.l_r*dw
            self.bias -= self.l_r*db
            if i in self.checkpointlist:
                self.data[i]=[{"Weight":float(self.weights),"Bias":float(self.bias)}]
        print(self.data)

    def predict(self,x):
        '''Returns predicted value'''
        return np.dot(x,self.weights)+self.bias

    def evaluate(self,actual,predicted):
        '''Returns MSE value'''
        mse = np.mean((actual-predicted)**2)
        return mse

