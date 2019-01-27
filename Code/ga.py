import random, math
import numpy as np
import skimage as skimage
import scipy as scipy
from skimage.color import rgb2gray
from skimage import data
from scipy import ndimage
from deap import creator, base, tools, algorithms

#% matplotlib
#inline

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

alpha = 0.5  # controls continuity energy impact
beta =  0.9 # controls curvatur energy impact
gamma = 1  #controls external engery impact

wLine = 0
wEdge = 1
imagepath = 'pic3.png'

image = cv2.imread(imagepath)
#image = data.astronaut()
image = rgb2gray(image)

floatimage = skimage.img_as_float(image)

edgeImage = np.sqrt(scipy.ndimage.sobel(floatimage, axis=0, mode='reflect') ** 2 +
                    scipy.ndimage.sobel(floatimage, axis=1, mode='reflect') ** 2)
edgeImage = (edgeImage - edgeImage.min()) / (edgeImage.max() - edgeImage.min())

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(edgeImage, cmap=plt.cm.gray)

plt.show()

externalEnergy = wLine * floatimage + wEdge * edgeImage

externalEnergyInterpolation = scipy.interpolate.RectBivariateSpline(np.arange(externalEnergy.shape[1]),
                                                                    np.arange(externalEnergy.shape[0]),
                                                                    externalEnergy.T, kx=2, ky=2, s=0)

def _externalEnergy(individual):
    x,y = individual[0][:,0], individual[0][:,1]
    energies = externalEnergyInterpolation(x,y,dy=1,grid=False)
    return (sum(energies))

def _areaEnergy(individual):
    A = 0.0
    for i in range(len(individual[0]) - 1):
        # x[i]* y[i+1] - x[i+1]*y[i]
        tmp_1 = individual[0][i][0] * individual[0][i + 1][1]
        tmp_2 = individual[0][i + 1][0] * individual[0][i][1]
        A = A + (tmp_1 - tmp_2)
    A = abs(A)
    return 0.5 * A


def _continuityEnergy(individual):
    x,y = individual[0][:,0], individual[0][:,1]
    n = len(x)
    cE = 0.0
    for i in range(n-1):
        nextp = np.array((x[i+1],y[i+1]))
        currp = np.array((x[i],y[i]))
        cE = cE + np.linalg.norm(nextp - currp)**2
    return (cE)


def _smoothnessEnergy(individual):
    x,y = individual[0][:,0], individual[0][:,1]
    n = len(x)
    sE = 0.0
    for i in range(1,n-1):
        nextp = np.array((x[i+1],y[i+1]))
        currp = np.array((x[i],y[i]))
        prevp = np.array((x[i-1],y[i-1]))
        sE = sE + np.sum((nextp -2 * currp + prevp)**2)
    return (sE)


def eval(individual):
    sE = _smoothnessEnergy(individual)
    cE = _continuityEnergy(individual)
    aE = _areaEnergy(individual)
    eE = _externalEnergy(individual)   
    return (alpha * cE + beta * sE + gamma * eE,)


def mutSet(prop, individual):
    size = len(individual[0])
    _x = random.randint(0, size - 1)
    _y = random.randint(0, 1)
    individual[0][_x][_y] += np.int(random.uniform(-2, 2))
    return individual,


def cxTwoPointCopy(ind1, ind2):
    # ind1 = ind1[0]
    # ind2 = ind2[0]
    size = len(ind1[0])
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(0, size - 2)
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
    range_x = (100, 150)
    range_y = (200, 270)
    range_r = (50, 80)
    _x = random.randint(min(range_x), max(range_x))
    _y = random.randint(min(range_y), max(range_y))
    _r = random.randint(min(range_r), max(range_r))
    s = np.linspace(0, 2 * np.pi, 50)
    x = _x + _r * np.cos(s)  # 130 + 80 * np.cos(s)
    y = _y + _r * np.sin(s)  # 250 + 57 * np.sin(s)
    V = np.array([x, y]).T
    #V2 = V[::50]

    x0, y0 = V[:, 0].astype(np.int), V[:, 1].astype(np.int)

    # store snake progress
    sn = np.array([x0, y0]).T
    return sn


def genetic_algorithm():
    random.seed(64)
    # init:
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
    toolbox.register("mutate", mutSet, 0.05)
    toolbox.register("select", tools.selTournament, tournsize=7)
    #random.seed(64)

    # run GA
    pop = toolbox.population(n=100)

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

    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=150, stats=stats, halloffame=hof)
    best = np.vstack([hof[0][0], hof[0][0][0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap=plt.cm.gray)
    circle1 = plt.Circle((125, 240), 80, color='r',fill=False)
    ax.add_artist(circle1)
    ax.plot(best[:, 0], best[:, 1], '-b', lw=3)

    plt.show()

    return pop, stats, hof


if __name__ == "__main__":
    genetic_algorithm()
