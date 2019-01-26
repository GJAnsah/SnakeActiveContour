import random
from deap import creator, base, tools, algorithms

import numpy as np



def eval(individual):
    return sum(individual)

def mutSet(individual):
    size = len(individual)
    _x = random.randint(1, size)
    _y = random.randint(1, 2)
    individual.args[1][_x][_y] += random.uniform(-10,10)
    return individual

def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


def genetic_algorithm( initInd):
    #init:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    # register individuals
    toolbox.register("individual", tools.initRepeat, creator.Individual, sn)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # register crossover
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("evaluate", eval)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    random.seed(64)

    #run GA
    pop = toolbox.population(n=300)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)


    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats,
                        halloffame=hof)

    return pop, stats, hof


if __name__ == "__main__":

    s = np.linspace(0, 2 * np.pi, 600)
    x = 130 + 80 * np.cos(s)
    y = 250 + 57 * np.sin(s)

    V = np.array([x, y]).T
    V2 = V[::50]

    x0, y0 = V2[:, 0].astype(np.float), V2[:, 1].astype(np.float)

    # store snake progress
    sn = np.array([x0, y0]).T
    genetic_algorithm(sn)