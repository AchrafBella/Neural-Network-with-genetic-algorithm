# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:34:05 2020

@author: Supernova
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# np.random.seed(1)

from  copy import deepcopy

# https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-
# position-of-figure-windows-with-matplotlib
def move_matplotlib_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)

    Move figure's upper left corner created by matplotlib.pyplot.figure() to 
    pixel (x, y)
    
    Parameters:
        figure: figure created by matplotlib.pyplot.figure()
        x (int): Coordonné x de la figure
        y (int): Coordonné y de la figure
    Returns:
        None
    """
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


# loss fonction to calculate the error 
# Mean Squared Error (MSE)
def mse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    #rmse_val = np.sqrt(mean_of_differences_squared)
    return mean_of_differences_squared # rmse_val

# fonction activation : sigmoid
def activation_sigmoid_derive(x, derive = False):
    if( derive == True):
        return x*(1-x)
    else:
        return 1/(1+ np.exp(-x))

# creation de la population des synapses 
def create_population(size_population):
    population = [(2*np.random.random((3,1))-1).tolist() 
        for _ in range(size_population)]
    return population

# fonction fitness
# param : population == popualtion
# la fontion tolist() pour transformer une array a une liste
def fitness(population, Input, activationFunction = activation_sigmoid_derive):
    output = []
    for elm in population:
        output.append(activationFunction(np.dot(Input, elm)).tolist() )
    return output

#fonction de la selection
#pour determiner les synapes qui sont bonne il faut calculer l'accuracy à
# travers RMSE nous avons ajouter l'accuracy dans la popualtion des synapes
# comme le tri la fonction sort( key = lambda accuracy_liste : 
# accuracy_liste[-1], reverse = True) pour le tri des premiers elements selon 
# l'accuracy comme on a fait le tri nous ne sommes plus besoin de l'accuracy on
# le supprimer par la fonction del et en retour la top des synapes
def selection(population, output, real_output):
    error = []
    i = 0
    for elm in output:
        population[i] += [mse(elm, real_output)]
        i += 1
    population.sort( key = lambda accuracy_liste : accuracy_liste[-1])
    for elm in population:
        error.append(elm[-1])
        del elm[-1]
    return population, error

#fonction de croissement
#le choix des parent aletoire
#la creation des enfant (les nouveaux synapes)
def crossover(population, crossoverSize, crossoverPoint,
              crossoverPointRandom = True):
    #crossoverSize : les top premiers selon l'accuracy
    topPopulation = population[:crossoverSize]
    # le split ou le crossover point
    if (crossoverPointRandom == True):
        split = np.random.randint(crossoverPoint)
    else:
        split = crossoverPoint
    parent1 = topPopulation[np.random.randint(crossoverSize)]
    parent2 = topPopulation[np.random.randint(crossoverSize)]
    child1 = deepcopy(parent1[0:split] + parent2[split:4])
    child2 = deepcopy(parent2[0:split] + parent1[split:4])
    # On supprime les derniers éléments
    population.pop(-1)
    population.pop(-1)
    # On ajoute les nouveaux enfants
    population.append(child1)
    population.append(child2)
    return population

def crossover_Dichotomie(population, crossoverSize):
    #crossoverSize : les top premiers selon l'accuracy
    topPopulation = population[:crossoverSize]

    parent1 = topPopulation[np.random.randint(crossoverSize)]
    parent2 = topPopulation[np.random.randint(crossoverSize)]
    parent3 = topPopulation[np.random.randint(crossoverSize)]
    parent4 = topPopulation[np.random.randint(crossoverSize)]
    child1 = []
    child1.append([(parent1[0][0] + parent2[0][0])/2])
    child1.append([(parent1[1][0] + parent2[1][0])/2])
    child1.append([(parent1[2][0] + parent2[2][0])/2])
    child2 = []
    child2.append([(parent3[0][0] + parent4[0][0])/2])
    child2.append([(parent3[1][0] + parent4[1][0])/2])
    child2.append([(parent3[2][0] + parent4[2][0])/2])

    # On supprime les derniers éléments, 
    population.pop(-1)
    population.pop(-1)
    # On ajoute les nouveaux enfants
    population.append(child1)
    population.append(child2)
    return population

#mutation
def mutation(population):
    for i in range(1, len(population)-2):
        split = np.random.randint(3)
        (population[i])[split][0] += (np.random.uniform(-1.0, 1.0, 1))[0]
    return population

def mutation3(population):
    for i in range(1, len(population)-2):
        (population[i])[0][0] += (np.random.uniform(-1.0, 1.0, 1))[0]
        (population[i])[1][0] += (np.random.uniform(-1.0, 1.0, 1))[0]
        (population[i])[2][0] += (np.random.uniform(-1.0, 1.0, 1))[0]
    return population

# données d'entrainement 
x_train = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y_train = np.array([[0,0,1,1]]).T

if __name__ == '__main__':
    meanErrorPopulation = []
    errorBestMemberPopulation = []
    errorWorstMemberPopulation = []
    generation = 400
    populationSize = 50
    m = None

    for i in range(generation):
        if(m == None):
            p = create_population(populationSize)
        else:
            p = m
        f = fitness(p, x_train)
        s, errorPopulation = selection(p, f, y_train)

        # print("s : ",*s, sep="\n")
        # print("\n ")
        # c = crossover_Dichotomie(p, 10)
        c = crossover(s, 10, 4, True)
        m = mutation(c)

        meanErrorPopulation.append(np.mean(errorPopulation))
        errorBestMemberPopulation.append(errorPopulation[0])
        errorWorstMemberPopulation.append(errorPopulation[populationSize-1])
        # if (i%10 == 0):
            # print('generation : ', i)
    
    best = fitness(s, x_train)
    print(*best[0:3], sep = '\n')

    # ——— La partie courbe de l’erreur ———–
    figEvolutionErreur = plt.figure("Evolution de l'erreur d'apprentissage")
    ax = figEvolutionErreur.add_subplot(1, 1, 1)
    ax.plot(range(generation), errorBestMemberPopulation,
            color='tab:blue', label='Error de la meilleur candidate')
    ax.plot(range(generation), meanErrorPopulation,
            color='tab:orange', label='Error moyenne de la poulation')
    ax.plot(range(generation), errorWorstMemberPopulation,
            color='tab:red', label='Error de la pire candidate')


    # set the limits
    ax.set_xlim([-0.1, generation])
    # ax.set_ylim([0, 1])

    ax.set_xlabel('Generation')
    ax.set_ylabel('Erreur')

    ax.set_title("Evolution de l'erreur d'apprentissage")
    ax.grid(True, linestyle='-.')
    ax.legend()

    move_matplotlib_figure(figEvolutionErreur, 10, 10)

    # display the plot
    plt.show()

    

    #for elm in best[0:3]:
    #    print(elm,'\n') 