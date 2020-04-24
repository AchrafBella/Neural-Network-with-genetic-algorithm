# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:57:19 2020

@author: Supernova
"""
from Neural_Network import Neural, Neural_pro
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
import math
import warnings

warnings.filterwarnings("ignore")
np.random.seed(1)

#processing data 
#convert function to turn a string to list of int element
def convert(string):
    liste = []
    for elm in string:
        liste.append(int(elm))
    return liste

#upload data for train
training_data_file = open('train.txt', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#preprocessing data 
x_train = []
y_train = []

for elm in training_data_list:
    elm = elm.replace('\n','').replace(',','').replace(' ','')
    x = convert(elm)
    x_train.append(x[:48])
    target = np.zeros(10)
    target[x[-1]] = 1
    y_train.append(target)
    pass

#upload data for test 
testing_data_file = open('test.txt', 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

#preprocessing data 
test = []
for elm in testing_data_list:
    elm = elm.replace('\n','').replace(',','').replace(' ','')
    test.append((convert(elm)))
    pass

# change the type of the data from list to array 
x_train = np.array(x_train)
y_train = np.array(y_train)
test    = np.array(test)
print("\n")
print(" Input_ data  :",x_train.shape)
print(" Output_data  :",y_train.shape)
print(" Testing_data :",test.shape)




def printInformation(fonction_activation, liste_alphas, liste_scores):
    s = " Activation function "+str(fonction_activation)+"\n"
    s += "___________________________________________________"+"\n"
    i = 0
    for elm in liste_alphas:
        s += "training test with alpha : {0:4f}, score : {1:2d}".\
                format(elm, liste_scores[i])+"/20"+"\n"
        i +=1
        pass
    return s


#test with activation function 
""" ====================== activation functions ====================== """
# Sigmoid functin and his derive
sigmoid  = lambda x:scipy.special.expit(x)
sigmoid_derive = lambda x : x*(1.0 - x)    

# Integral_Sigmoid functin and his derive 
Integral_Sigmoid = lambda x : np.log( 1.0 + np.exp(x) )
sigmoid  = lambda x:scipy.special.expit(x)

#activation function tan
thn = lambda x : np.tanh(x) 
thn_der = lambda x : 1+x**2

#fonction d'activation arctang
arctang     = lambda x : np.arctan(x)
arctang_der = lambda x: 1/ (math.pi * (1+x*x) +0.5)
"""

alphas = [1, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.14]
#alphas = [0.00003, 0.005, 0.002, 0.001,0.1,0.2,0.3,0.5]

score1 = []
score2 = []
score3 = []
score4 = []

score11 = []
score22 = []
score33 = []
score44 = []

print(' neural network with one hidden layer')

for alpha in alphas:
    n1 = Neural(48, 32, 10, alpha, arctang, arctang_der)
    n1.fit(x_train, y_train, 2500)
    n1.plot_Neural_error(alpha, " arctang ")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n1.query(elm[:48]))
        if( x == y):
            cpt+=1
    score1.append(cpt)
    pass
print(printInformation("arctang", alphas, score1))

for alpha in alphas:
    n2 = Neural(48, 32, 10, alpha, Integral_Sigmoid, sigmoid)
    n2.fit(x_train, y_train, 2500)
    n2.plot_Neural_error(alpha, "Intergral_sigmoid")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n2.query(elm[:48]))
        if( x == y):
            cpt+=1
    score2.append(cpt)
    pass
print(printInformation("integral_sigmoid", alphas, score2))

for alpha in alphas:
    n3 = Neural(48, 32, 10, alpha, thn, thn_der)
    n3.fit(x_train, y_train, 2500)
    n3.plot_Neural_error(alpha, "tangante")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n3.query(elm[:48]))
        if( x == y):
            cpt+=1
    score3.append(cpt)
    pass

print(printInformation("thang", alphas, score3))


for alpha in alphas:
    n4 = Neural(48, 32, 10, alpha, sigmoid, sigmoid_derive)
    n4.fit(x_train, y_train, 2500)
    n4.plot_Neural_error(alpha, " Sigmoid ")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n4.query(elm[:48]))
        if( x == y):
            cpt+=1
    score4.append(cpt)
    pass
print(printInformation("Sigmoid", alphas, score4))

print('neural network with two hidden layers')

for alpha in alphas:
    n11 = Neural_pro(48, 60, 16, 10, alpha, arctang, arctang_der)
    n11.fit(x_train, y_train, 2500)
    n11.plot_Neural_error(alpha, " arctang ")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n11.query(elm[:48]))
        if( x == y):
            cpt+=1
    score11.append(cpt)
    pass
print(printInformation("arctang", alphas, score11))


for alpha in alphas:
    n22 = Neural_pro(48, 60, 16, 10, alpha, Integral_Sigmoid, Integral_Sigmoid)
    n22.fit(x_train, y_train, 2500)
    n22.plot_Neural_error(alpha, " Integral_Sigmoid ")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n22.query(elm[:48]))
        if( x == y):
            cpt+=1
    score22.append(cpt)
    pass
print(printInformation("Integral_Sigmoid", alphas, score22))

for alpha in alphas:
    n33 = Neural_pro(48, 60, 16, 10, alpha, thn, thn_der)
    n33.fit(x_train, y_train, 2500)
    n33.plot_Neural_error(alpha, " tangante ")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n33.query(elm[:48]))
        if( x == y):
            cpt+=1
    score33.append(cpt)
    pass
print(printInformation("tangante", alphas, score33))

alphas = [0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.14]

dropouts = [0.6,0.7]

for alpha in alphas:
    print("alpha ",alpha)
    for drop in dropouts:
        n44 = Neural_pro(48, 60, 16, 10, alpha, sigmoid, sigmoid_derive)
        print("drop out ",drop)
        n44.fit(x_train, y_train, 2500, True, drop)
        #n44.plot_Neural_error(alpha, " Sigmoid ")
        cpt = 0
        for elm in test:
            x = elm[-1]
            y = np.argmax(n44.query(elm[:48]))
            if( x == y):
                cpt+=1
        print(cpt,"/20")
        score44.append(cpt)
        pass
#print(printInformation("Sigmoid", alphas, score44))





dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

scored1 = []
scored2 = []

for alpha in alphas:
    n33 = Neural_pro(48, 32, 16, 10, alpha, thn, thn_der)
    n33.fit(x_train, y_train, 2500)
    n33.plot_Neural_error(alpha, " tangante ")
    cpt = 0
    for elm in test:
        x = elm[-1]
        y = np.argmax(n33.query(elm[:48]))
        if( x == y):
            cpt+=1
    scored1.append(cpt)
    pass
print(printInformation("tangante", alphas, scored1))


for alpha in alphas:
    n33 = Neural_pro(48, 32, 16, 10, alpha, thn, thn_der)
    for drop in dropouts:
        print('drop out with ',drop)
        n33.fit(x_train, y_train, 2500, True, drop)
        n33.plot_Neural_error(alpha, " tangante ")
        cpt = 0
        for elm in test:
            x = elm[-1]
            y = np.argmax(n33.query(elm[:48]))
            if( x == y):
                cpt+=1
        scored2.append(cpt)
    print(printInformation("tangante", alphas, scored2))

"""









