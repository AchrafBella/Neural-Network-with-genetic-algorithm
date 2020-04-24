""" École Centrale de Lyon
    UE INF S8 Algorithmes collaboratifs et applications 2019-2020
    BE - Réseau de neurones utilisant algorithmes génétiques.
    
    Module pour fonctions d’activation

    Sigmoid (ou logistique ou encore escalier lissé)

    Rectified linear unit (ReLU)

    Tangente Hyperbolique_______________________________________________________
    La fonction Tangente Hyperbolique f(x) = tanh(x) = 2/(1 + e**(-2*x)) - 1 
    dont la dérivée est 1 - f(x)**2 et l’étendu (l’image) dans [-1; 1].
    + Remarquez bien que la dérivée f' contient sa primitive f (comme pour la 
    sigmoïde).

    Gaussienne__________________________________________________________________
    La fonction Gaussienne f(x) = e**(-x**2) et dont la dérivée sera f'(x) = 
    -2*x*e**(-x**2) = -2 x f(x) et l’étendu dans le demi (0; 1]. Remarquez bien 
    que la dérivée f' contient f mais on a quand même besoin de x !

    Afin de ramener l’image de cette fonction dans [0; 1], on peut diviser la 
    valeur par 2 et ajouter 0.5, ce qui donne [-1/2 + 0.5; 1/2 + 0.5] = [0; 1]

    Arc Tangente________________________________________________________________
    La fonction Arc Tangente f(x) = tan-1(x) dont la dérivée est 1/(1 + x**2) et 
    l’étendu dans [-pi/2 ; pi/2].

    Comme observé ci-dessus, pour ramener l’image de cette fonction dans [0; 1],
    on multiplie les deux bornes par 1/pi et on leur ajoute 1/2 pour obtenir
    [0 ; 1]

    https://en.wikipedia.org/wiki/Activation_function

    @author: Achraf Bella
    @author: Bruno Moreira Nabinger
"""

# import numpy as np
from numpy import exp, maximum, minimum, tanh, arctan
from math import pi

# Activation function:  sigmoid
def sigmoid_derive(x, derive = False):
    if(derive == True):
        return x*(1-x)
    else:
        return 1/(1+ exp(-x))

# Activation function: Rectified linear unit (ReLU)
# https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-
# function-in-numpy#comment62519312_32109519
def ReLU(x, derive = False):
    return maximum(x, 0, x)

# Activation function: tangente hyperbolique
def TanH(x, derive = False):
    return tanh(x)

# Activation function: Gaussian
def Gaussian(x, derive = False):
    return ((exp(-(x**2)))/2 + 0.5)

# Activation function: arctangente
def ArcTan(x, derive = False):
    return (1/pi * arctan(x) + 0.5)