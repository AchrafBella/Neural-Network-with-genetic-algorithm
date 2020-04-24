# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:34:05 2020

@author: Supernova
"""
import numpy as np
np.random.seed(1)
# loss fonction to calculate the error 
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val

# fonction activation : sigmoid
def sigmoid_derive(x, derive = False):
    if( derive == True):
        return x*(1-x)
    else:
        return 1/(1+ np.exp(-x))
    
# données d'entrainement 
x_train = np.array([[0,1,1],[0,1,1],[1,0,1],[1,1,1]])
y_train = np.array([[0,0,1,1]]).T

# creation de la population des synapses 
def population(size_population):
    population = [(2*np.random.random((3,1))-1).tolist() for _ in range(size_population)]
    return population

# fonction fitness
# param : pop == popualtion
# la fontion tolist() pour transformer une array a une liste
def fitness(pop, Input):
    output = []
    for elm in pop:
        output.append( np.abs((sigmoid_derive(np.dot(Input, elm)).tolist() )))
    return output

#fonction de la selection des 10 premiers
#pour determiner les synapes qui sont bonne il faut calculer l'accuracy à travers RMSE
#nous avons ajouter l'accuracy dans la popualtion des synapes comme le tri 
#la fonction sort( key = lambda accuracy_liste : accuracy_liste[-1], reverse = True) pour le tri des premiers element selon l'accuracy
# comme on a fait le tri nous ne sommes plus besoin de l'accuracy on le supprimer par la fonction del et en retour la top des synapes
def selection(pop, output, real_output):
    i = 0
    for elm in output:
        pop[i] += [rmse(elm, real_output)]
        i += 1
    pop.sort( key = lambda accuracy_liste : accuracy_liste[-1], reverse = True)
    for elm in pop:
        del elm[-1]
    #print("Vamos com muita calma nessa hora, Selection!: pop,", pop, sep="\n")
    return pop

#fonction de croissement
#le choix des parent aletoire
#la creation des enfant (les nouveaux synapes)

def crossover(pop, top):
    #les top premiers selon l'accuracy
    top_10 = pop[:top]
    # le split ou le crossover point
    split = np.random.randint(4)
    parent1 = top_10[np.random.randint(top)]
    parent2 = top_10[np.random.randint(top)]
    child1 = parent1[0:split] + parent2[split:4]
    child2 = parent2[0:split] + parent1[split:4]
    # j'ai supprimer les 10 premiers dans la liste qui contient les 20 synapes (la population)
    del pop[:top]
    # j'ai sypprimer les parent et j'ai ajouter les nouveaux enfant
    if( parent1 == parent2):
        top_10.remove(parent1)
    else:
        top_10.remove(parent1)
        top_10.remove(parent2)
    top_10.append(child1)
    top_10.append(child2)
    # puis j'ai ajouter la nouvelle liste de top 10 dans la liste de la population
    pop.extend(top_10)
    return pop

#mutation
def mutation(pop):
    for elm in pop:
        split = np.random.randint(3)
        elm[split][0] += (np.random.uniform(-1.0, 1.0, 1))[0]
    return pop

if __name__ == '__main__':
    generation = 4
    m = None
    if(m == None):
        p = population(50)
    else:
        p=m
    for i in range(generation):
        f = fitness(p, x_train)
        s = selection(p, f, y_train)
        print("s : ",*s, sep="\n")
        print("\n ")
        c = crossover(s, 10)
        m = mutation(c)
        print('generation : ',i)
    for elm in f:
        print(elm,'\n') 