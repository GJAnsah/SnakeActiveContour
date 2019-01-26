import random
from deap import creator, base, tools, algorithms

import numpy as np



def eval(individual):
    A = 0
    individual = individual[0]
    for i in range(len(individual[0])-1):
        #x[i]* y[i+1] - x[i+1]*y[i]
        tmp1 = individual[i][0] * individual[i+1][1]
        tmp2 = individual[i+1][0] * individual[i][1]
        A = A + (tmp1 - tmp2)
    A = abs(A)
    return 0.5*A,

def mutSet(individual, prop):
    size = len(individual)
    _x = random.randint(1, size)
    _y = random.randint(1, 2)
    individual[0][_x][_y] += random.uniform(-10,10)
    return individual

def cxTwoPointCopy(ind1, ind2):
    #ind1 = ind1[0]
    #ind2 = ind2[0]
    size = len(ind1[0])
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[0][cxpoint1:cxpoint2], ind2[0][cxpoint1:cxpoint2] \
        = ind2[0][cxpoint1:cxpoint2].copy(), ind1[0][cxpoint1:cxpoint2].copy()
    ind1.fitness.values = eval(ind1)
    ind2.fitness.values = eval(ind2)
    return ind1, ind2

def initIndividual():
    range_x = (100,150)
    range_y = (200, 270)
    range_r = (50,80)
    _x = random.randint(min(range_x), max(range_x))
    _y = random.randint(min(range_y), max(range_y))
    _r = random.randint(min(range_r), max(range_r))
    s = np.linspace(0, 2 * np.pi, 600)
    x = _x + _r * np.cos(s) #130 + 80 * np.cos(s)
    y = _y + _r * np.sin(s) #250 + 57 * np.sin(s)
    V = np.array([x, y]).T
    V2 = V[::50]
    x0, y0 = V2[:, 0].astype(np.float), V2[:, 1].astype(np.float)
    # store snake progress
    sn = np.array([x0, y0]).T
    return sn

def genetic_algorithm( ):
    random.seed(86)
    #init:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)


    toolbox = base.Toolbox()
    # register individuals
    toolbox.register("init_ind", initIndividual)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.init_ind, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # register crossover
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("evaluate", eval)
    toolbox.register("mutate", mutSet, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    random.seed(64)

    #run GA
    pop = toolbox.population(n=300)

    for p in pop:
        p.fitness.values = eval(p)


    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats)
                        #,
                       # halloffame=hof)

    return pop, stats, hof



if __name__ == "__main__":

    genetic_algorithm()