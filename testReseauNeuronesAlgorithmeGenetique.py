# -*- coding: utf-8 -*-
""" École Centrale de Lyon
    UE INF S8 Algorithmes collaboratifs et applications 2019-2020
    BE - Réseau de neurones utilisant algorithmes génétiques.
    La méthode de Descente de Gradient est montrée aussi à titre d'exemple   

    @author: Achraf Bella
    @author: Bruno Moreira Nabinger
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)

from reseauNeuronesAlgorithmeGenetique import create_population, fitness, \
    selection, crossover, crossover_Dichotomie, mutation, mutation3
import activationFunctions

# https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-
# position-of-figure-windows-with-matplotlib
def move_figure(figure, x, y):
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
        figure.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        figure.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        figure.canvas.manager.window.move(x, y)

def creeCourbesEvolutionErreur(figure, title,  numberOfGenerations, 
    errorBestMemberPopulation, meanErrorPopulation, errorWorstMemberPopulation):
    """Crée Courbes Evolution Erreur

    Move figure's upper left corner created by matplotlib.pyplot.figure() to 
    pixel (x, y)
    
    Parameters:
        figure: figure created by matplotlib.pyplot.figure()
        title (string): titre des courbes, par exemple "Evolution de l'erreur 
            d'apprentissage : "
        numberOfGenerations (int): nombre de générations utilisé dans la 
            méthode algorithme génétique
        errorBestMemberPopulation (list[floats]): liste que contient l'erreur de
            la meuilleure candidate de chaque génération
        meanErrorPopulation (list[floats]): liste que contient l'erreur moyenne
            de la population de chaque génération
        errorWorstMemberPopulation (list[floats]): liste que contient l'erreur 
            de la pire candidate de chaque génération
    Returns:
        None
    """
    # ——— La partie courbe de l’erreur ———–
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(range(numberOfGenerations), errorBestMemberPopulation,
            color='tab:blue', label='Error de la meilleure candidate')
    ax.plot(range(numberOfGenerations), meanErrorPopulation,
            color='tab:orange', label='Error moyenne de la population')
    ax.plot(range(numberOfGenerations), errorWorstMemberPopulation,
            color='tab:red', label='Error de la pire candidate')


    # set the limits
    ax.set_xlim([-0.1, 220])#generation])
    # ax.set_ylim([0, 1])

    ax.set_xlabel('Generation')
    ax.set_ylabel('Erreur')

    ax.set_title(title)
    ax.grid(True, linestyle='-.')
    ax.legend()

    # # display the plot
    # plt.show()

def printInfosReseauNeuroneAlgorithmeGenetique(
        generation, populationSize,
        crossoverSize, crossoverPoint, crossoverPointRandom,
        activationFunctionName):
    print("Fonction reseauNeuroneAlgorithmeGenetique arguments: \n" \
        "\tNombre de générations : {} Taille de la population : {}\n" \
        "\tNombre d'éléments prises en compte pour le croissement : {}\n" \
        "\tSplit ou crossover Point : {}\t Crossover Point aléatoire : {}\n" \
        "\tFonction d'ativation: {}" \
        .format(generation, populationSize, \
            crossoverSize, crossoverPoint, crossoverPointRandom, \
            activationFunctionName))

def reseauNeuroneAlgorithmeGenetique(
        x_train, y_train, 
        generation = 400, populationSize = 50,
        crossoverSize = 10, crossoverPoint = 4, crossoverPointRandom = True,
        activationFunction = activationFunctions.sigmoid_derive, 
        tolerance = 1.0e-08):
    meanErrorPopulation = []
    errorBestMemberPopulation = []
    errorWorstMemberPopulation = []
    m = None
    turnToleranceAchieved = None
    
    for i in range(generation):
        if(m == None):
            p = create_population(populationSize)
        else:
            p=m
        f = fitness(p, x_train, activationFunction)
        s, errorPopulation = selection(p, f, y_train)

        # print("s : ",*s, sep="\n")
        # print("\n ")
        # c = crossover_Dichotomie(s, crossoverSize)
        c = crossover(s, crossoverSize, crossoverPoint, crossoverPointRandom)
        m = mutation3(c)
        # m = mutation(c)

        meanErrorPopulation.append(np.mean(errorPopulation))
        errorBestMemberPopulation.append(errorPopulation[0])
        errorWorstMemberPopulation.append(errorPopulation[populationSize-1])

        if (errorPopulation[0] < tolerance and turnToleranceAchieved is None):
            turnToleranceAchieved = i
        # if (i%10 == 0):
            # print('generation : ', i)
    
    best = fitness(s, x_train)
    # print(*best[0:3], sep = '\n')

    return best[0], s[0], turnToleranceAchieved, errorBestMemberPopulation, \
         meanErrorPopulation, errorWorstMemberPopulation
    

if __name__ == '__main__':
    print("Machine epsilon (float):")
    print(np.finfo(float).eps)

    tolerance = 1.0e-04
    print("Tolerance : ", tolerance)



    # données d'entrainement 
    inputData = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    outputData = np.array([[0,0,1,1]]).T
    print("inputData: ", *inputData)
    print("outputData: ", *outputData)

    print("_" * 80)
    generation = 600
    populationSize = 50
    crossoverSize = 10
    crossoverPoint = 4
    crossoverPointRandom = True
    printInfosReseauNeuroneAlgorithmeGenetique(generation, populationSize,
        crossoverSize, crossoverPoint, crossoverPointRandom,
        "Sigmoide")

    output, weights, turnToleranceAchieved, errorBestMemberPopulation, \
        meanErrorPopulation, errorWorstMemberPopulation = \
        reseauNeuroneAlgorithmeGenetique(inputData, outputData, 
                                         generation, populationSize,
                                         crossoverSize, crossoverPoint,
                                         crossoverPointRandom,
                                         activationFunctions.sigmoid_derive,
                                         tolerance)

    print("Output =", output)
    print("Weights =", weights)
    print("Turn tolerance achieved =", turnToleranceAchieved)

    figEvolutionErreurSigmoide = \
        plt.figure("Evolution de l'erreur d'apprentissage : Sigmoide")
    creeCourbesEvolutionErreur(figEvolutionErreurSigmoide,
                            "Evolution de l'erreur d'apprentissage : Sigmoide",
                               generation,
                               errorBestMemberPopulation,
                               meanErrorPopulation, 
                               errorWorstMemberPopulation)
    move_figure(figEvolutionErreurSigmoide, 10, 10)
    
    print("_" * 80)
    generation = 600
    populationSize = 50
    crossoverSize = 10
    crossoverPoint = 4
    crossoverPointRandom = True
    printInfosReseauNeuroneAlgorithmeGenetique(generation, populationSize,
        crossoverSize, crossoverPoint, crossoverPointRandom,
        "ReLU")

    output, weights, turnToleranceAchieved, errorBestMemberPopulation, \
        meanErrorPopulation, errorWorstMemberPopulation = \
        reseauNeuroneAlgorithmeGenetique(inputData, outputData, 
                                         generation, populationSize,
                                         crossoverSize, crossoverPoint,
                                         crossoverPointRandom,
                                         activationFunctions.ReLU,
                                         tolerance)
                                         
    print("Output =", output)
    print("Weights =", weights)
    print("Turn tolerance achieved =", turnToleranceAchieved)

    figEvolutionErreurReLU = \
        plt.figure("Evolution de l'erreur d'apprentissage : ReLU")
    creeCourbesEvolutionErreur(figEvolutionErreurReLU, 
                            "Evolution de l'erreur d'apprentissage : ReLU",
                               generation,
                               errorBestMemberPopulation,
                               meanErrorPopulation,
                               errorWorstMemberPopulation)
    move_figure(figEvolutionErreurReLU, 460, 10)
    
    print("_" * 80)
    generation = 600
    populationSize = 50
    crossoverSize = 10
    crossoverPoint = 4
    crossoverPointRandom = True
    printInfosReseauNeuroneAlgorithmeGenetique(generation, populationSize,
        crossoverSize, crossoverPoint, crossoverPointRandom,
        "Tangente Hyperbolique")

    output, weights, turnToleranceAchieved, errorBestMemberPopulation, \
        meanErrorPopulation, errorWorstMemberPopulation = \
        reseauNeuroneAlgorithmeGenetique(inputData, outputData, 
                                         generation, populationSize,
                                         crossoverSize, crossoverPoint,
                                         crossoverPointRandom,
                                         activationFunctions.TanH,
                                         tolerance)

    print("Output =", output)
    print("Weights =", weights)
    print("Turn tolerance achieved =", turnToleranceAchieved)

    figEvolutionErreurTanH = \
        plt.figure("Evolution de l'erreur d'apprentissage : Tan Hyperbolique")
    creeCourbesEvolutionErreur(figEvolutionErreurTanH, 
                            "Evolution de l'erreur d'apprentissage : TanH",
                               generation,
                               errorBestMemberPopulation,
                               meanErrorPopulation, 
                               errorWorstMemberPopulation)
    move_figure(figEvolutionErreurTanH, 860, 10)

    print("_" * 80)
    generation = 600
    populationSize = 50
    crossoverSize = 10
    crossoverPoint = 4
    crossoverPointRandom = True
    printInfosReseauNeuroneAlgorithmeGenetique(generation, populationSize,
        crossoverSize, crossoverPoint, crossoverPointRandom,
        "Gaussienne")

    output, weights, turnToleranceAchieved, errorBestMemberPopulation, \
        meanErrorPopulation, errorWorstMemberPopulation = \
        reseauNeuroneAlgorithmeGenetique(inputData, outputData, 
                                         generation, populationSize,
                                         crossoverSize, crossoverPoint,
                                         crossoverPointRandom,
                                         activationFunctions.Gaussian,
                                         tolerance)

    print("Output =", output)
    print("Weights =", weights)
    print("Turn tolerance achieved =", turnToleranceAchieved)

    figEvolutionErreurGaussienne = \
        plt.figure("Evolution de l'erreur d'apprentissage : Gaussienne")
    creeCourbesEvolutionErreur(figEvolutionErreurGaussienne,
                        "Evolution de l'erreur d'apprentissage : Gaussienne",
                               generation,
                               errorBestMemberPopulation,
                               meanErrorPopulation, 
                               errorWorstMemberPopulation)
    move_figure(figEvolutionErreurGaussienne, 10, 310)

    print("_" * 80)
    generation = 600
    populationSize = 50
    crossoverSize = 10
    crossoverPoint = 4
    crossoverPointRandom = True
    printInfosReseauNeuroneAlgorithmeGenetique(generation, populationSize,
        crossoverSize, crossoverPoint, crossoverPointRandom,
        "Arc Tangente")

    output, weights, turnToleranceAchieved, errorBestMemberPopulation, \
        meanErrorPopulation, errorWorstMemberPopulation = \
        reseauNeuroneAlgorithmeGenetique(inputData, outputData, 
                                         generation, populationSize,
                                         crossoverSize, crossoverPoint,
                                         crossoverPointRandom,
                                         activationFunctions.ArcTan,
                                         tolerance)

    print("Output =", output)
    print("Weights =", weights)
    print("Turn tolerance achieved =", turnToleranceAchieved)

    figEvolutionErreurArcTan = \
        plt.figure("Evolution de l'erreur d'apprentissage : Arc Tangente")
    creeCourbesEvolutionErreur(figEvolutionErreurArcTan,
                            "Evolution de l'erreur d'apprentissage : ArcTan",
                               generation,
                               errorBestMemberPopulation,
                               meanErrorPopulation, 
                               errorWorstMemberPopulation)
    move_figure(figEvolutionErreurArcTan, 610, 310)


    # display the plots
    plt.show()
    