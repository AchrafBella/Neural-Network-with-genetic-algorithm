# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:27:43 2020

@author: Supernova
"""
import numpy as np
from scipy.special import expit

# loss fonction to calculate the error 
# Mean Squared Error (MSE)
def mse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

class Neural_Network():
    def __init__(self, input_nodes, output_nodes, size):
        #les parametres de notre réseau 
        self.input_nodes  = input_nodes
        self.output_nodes = output_nodes
        self.size = size
        
        self.weights = list()   
        # fonction activation : sigmoid
        self.sigmoid = lambda x : expit(x)
        pass
        
    def __Population(self):
        self.weights = [(2*np.random.random((self.input_nodes, self.output_nodes))-1).tolist() for _ in range(self.size)]
        return self.weights

    def __Fitness(self, x, y):
        liste = list()
        error = list()
        for elm in self.weights:
            output = self.sigmoid(np.dot(x,elm))
            liste.append(output)
            elm += [mse(output, y)]
            error.append(mse(output, y))
            pass
        return liste, np.mean(error)
    #fonction de la selection des 10 premiers
    #pour determiner les synapes qui sont bonne il faut calculer l'accuracy à
    # travers RMSE nous avons ajouter l'accuracy dans la popualtion des synapes
    # comme le tri la fonction sort( key = lambda accuracy_liste : 
    # accuracy_liste[-1], reverse = True) pour le tri des premiers element selon 
    #l'accuracy comme on a fait le tri nous ne sommes plus besoin de l'accuracy on
    #le supprimer par la fonction del et en retour la top des synapes
    def __Selection(self):
        self.weights = sorted(self.weights, key = lambda liste : liste[-1])
        for elm in self.weights:
            del elm[-1]
            pass
        pass
    #fonction de croissement
    #le choix des parent aletoire
    #la creation des enfant (les nouveaux synapes)

    def __Crossover(self, top):
        top_weight = self.weights[:top]
        parent1, parent2 = top_weight[np.random.randint(top)],  top_weight[np.random.randint(top)]
        while(parent1 == parent2):
            parent2 = top_weight[np.random.randint(top)]
            pass
        neural = (np.mean([parent1, parent2], axis = 0)).tolist()
        del self.weights[-1]
        self.weights.append(neural)
        pass
    
    def __Mutation(self):
        for elm in self.weights:
            split = np.random.randint(3)
            elm[split][0] += (np.random.uniform(-1.0, 1.0, 1))[0]            
            pass
        pass
    
    def fit(self, x, y, epochs, top = 5):
        """
        fonction fit pour entrain notre réseau de neurones 
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        epochs : TYPE
            DESCRIPTION.
        top : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        """
        self.__Population()
        for e in range(epochs):
            output, error = self.__Fitness(x, y)
            self.__Selection()
            self.__Crossover(top)
            self.__Mutation()
            if(e % 5 == 0):
                print('epoch ==>', e, 'loss ==>',error)
            pass
        return output
pass
    
if __name__ == '__main__':
    # données d'entrainement 
    x = np.array([ [0,0,1],
                          [1,1,1],
                          [1,0,1],
                          [0,1,1]])
    
    y = np.array([[0,1,1,0]]).T
    
    n = Neural_Network(3, 1, 500)
    t = n.fit(x, y,150, 100)
    print(*t[0:5], sep = '\n')
