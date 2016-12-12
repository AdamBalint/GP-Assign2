import operator
import math
import random
from random import shuffle
#import GP_Graph as gpg
import scipy as sp

#import KFold

import numpy as np


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from scipy.misc import imread, imsave
from scipy.ndimage.filters import sobel, laplace, gaussian_laplace



'''

Read in data

'''
data = []

print("Reading and generating images...")
gt_img = imread("Images/gt-img1.png", flatten="False", mode="RGB")
images = []
images.append(imread("Images/g-img1.png", flatten="False", mode="RGB"))
images.append(sobel(images[0], axis=0))
images.append(sobel(images[0], axis=1))
mag = np.hypot(images[-1], images[-2])  # magnitude
mag *= 255.0 / np.max(mag)  # normalize (Q&D)
imsave('sobel-both.jpg', mag)
images.append(laplace(images[0]))
imsave('laplace.jpg', images[-1])
images.append(gaussian_laplace(images[0], 0.05))
mean_13 = []
mean_17 = []
mean_21 = []
std_13 = []
std_17 = []
std_21 = []
print("Reading in pre-calculated means...")

f = open("Images/Means/Test2-img-0-13.txt", 'r')
for row in f.readlines():
    mean_13.append([float(x) for x in row.split()])
f.close()

f = open("Images/Means/Test2-img-0-17.txt", 'r')
for row in f.readlines():
    mean_17.append([float(x) for x in row.split()])
f.close()

f = open("Images/Means/Test2-img-0-21.txt", 'r')
for row in f.readlines():
    mean_21.append([float(x) for x in row.split()])
f.close()

f = open("Images/Deviation/Test2-img-0-13.txt", 'r')
for row in f.readlines():
    std_13.append([float(x) for x in row.split()])
f.close()

f = open("Images/Deviation/Test2-img-0-17.txt", 'r')
for row in f.readlines():
    std_17.append([float(x) for x in row.split()])
f.close()

f = open("Images/Deviation/Test2-img-0-21.txt", 'r')
for row in f.readlines():
    std_21.append([float(x) for x in row.split()])
f.close()

print("Setting up data for use...")

for y in range(len(gt_img)):
    row = []
    for x in range(len(gt_img[y])):
        #print(images[0][y][x])
        tmp = [int(gt_img[y][x]), int(images[0][y][x]), mean_13[y][x], mean_17[y][x], mean_21[y][x], std_13[y][x], std_17[y][x], std_21[y][x],
        int(images[2][y][x])%255, int(images[1][y][x])%255, int(images[3][y][x])%255, int(images[4][y][x])%255]

        # sobels:mean_17[y][x],
        #print (tmp)
        row.append(tmp)
    data.append(row)

points = []

print("Reading in training points...")
#f = open("img_1_datapoints.txt")
f = open("Rand_datapoints4.txt")

for line in f.readlines()[1:]:
    points.append([int(l) for l in line.split()])
    #print (x) for x in line.split()[:2])
#print(points)

print(points[0])

#print (data)

# Definition of the protected div
def protectedDiv(left, right):
    # if the number is close to 0, treat is as 0 to prevent infinity
    try:
        if (right < 0.000000000001):
            return 1
        return left / right
    except ZeroDivisionError:
        return 1
    except RuntimeWarning:
        print("Left: " + str(left) + "\tRight: " + str(right))
        return 1

# defines the absolute square root
def abs_sqrt(num):
    return math.sqrt(abs(num))

def cbrt(num):
    return sp.special.cbrt(num)

# defines sin using radians
def sin(num):
    try:
        n2 = math.radians(num)
        return math.sin(n2)
    except ValueError:
        print("Infinity Warning")
        print("Num: " + str(num))

# defines cosine using radians
def cos(num):
    try:
        n2 = math.radians(num)
        return math.cos(n2)
    except ValueError:
        print("Infinity Warning")
        print("Num: " + str(num))

# Defines the modulo operator
def modulo(n1, n2):
    if (n2 == 0):
        return n1
    return n1%n2

# Defines distance calculation using 4 inputs
def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# Defines an if statement comparing to 0
def ifStatement(n1, n2, n3):
    if (n1 < 0):
        return n2
    else:
        return n3

# defines an if statement comparing 2 numbers
def compStatement(n1, n2, n3, n4):
    if (n1 < n2):
        return n3
    else:
        return n4

# calculates the determinant of a 2x2 matrix made from 4 inputs
def det2x2(n11, n12, n21, n22):
    return (n11*n22-n12*n21)

def average(n1, n2):
    return (n1+n2)/2

# allows to set the test number to use, and the base output name
test_num = 6;
name = "Experiment-" + str(test_num)




# set up GP parameters
# Original Value, Original Mean x 3, Value Sobel vertical, value sobel horizontal, edge detect, blotchy value
pset = gp.PrimitiveSet("MAIN", 11)
# add the addition operator. Has 2 inputs
# sets up the rest of the operators to use
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)

# include other operators based on test number
if (test_num > 1):
    pset.addPrimitive(sin,1)
if (test_num > 3):
    pset.addPrimitive(cbrt, 1)
if (test_num > 4):
    pset.addPrimitive(round, 1)
if (test_num > 5):
    pset.addPrimitive(ifStatement,3)
if (test_num > 2):
    pset.addPrimitive(math.tanh,1)

# unused operators
#pset.addPrimitive(cos,1)
pset.addPrimitive(abs_sqrt, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
#pset.addPrimitive(math.floor, 1)
#pset.addPrimitive(math.ceil, 1)
pset.addPrimitive(modulo, 2) # seems to be pretty good
#pset.addPrimitive(dist,4)
#pset.addPrimitive(compStatement,4)
#pset.addPrimitive(det2x2,4)
pset.addPrimitive(average,2)

# add the posibility of a constant from -1 to 1
pset.addEphemeralConstant("const", lambda: random.uniform(-1, 1))

pset.renameArguments(ARG0="oValue")
pset.renameArguments(ARG1="oAvg13")
pset.renameArguments(ARG2="oAvg17")
pset.renameArguments(ARG3="oAvg21")
pset.renameArguments(ARG4="oStd13")
pset.renameArguments(ARG5="oStd17")
pset.renameArguments(ARG6="oStd21")
pset.renameArguments(ARG7="sobelVal_v")
pset.renameArguments(ARG8="sobelVal_h")
pset.renameArguments(ARG9="edgeVal")
pset.renameArguments(ARG10="blotchVal")

# Individual will have intensity, 13x13, 17x17, 21x21 means for each filter
# evaluation will take the arrays, and an x and y value







# state that it is a maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# set the individuals language and fitness type
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# set tree generation parameters
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
# Set up individuals and populations and set up the compilation process
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# define the evaluation function
# counts the number of correct classifications
# Method must return an array
def evalFunc(individual, all_data, train_points):
    func = toolbox.compile(expr=individual)
    total = 1.0

    for point in train_points:
        tmp = [float(x) for x in all_data[point[1]][point[0]][1:]]
        if (int((0 if (func(*tmp)) <= 0 else 1) == point[2])):
            if (point[2] == 0):
                total += 0.1
            else:
                total += 0.9
        #print("Result: " + str(int((0 if (func(*tmp)) <= 0 else 1)) + "\tAns: " + str(point[2]))


    return total,

# define the evaluation for the testing
# Identical to above, but does more logging
def testEval(individual, all_data, train_points):
    tp, tn, fp, fn = 0,0,0,0
    func = toolbox.compile(expr=individual)
    total = 0

    im = []

    for y in range(len(gt_img)):
        #print("In row: " + str(y))
        t = []
        for x in range(len(gt_img[y])):
            tmp = [float(a) for a in all_data[y][x][1:]]
            res = 0 if (func(*tmp)) <= 0 else 1

            if (res == 1):
                if (all_data[y][x][0] > 0):
                    tp += 1
                    t.append([125,0,0,255])
                else:
                    fp += 1
                    t.append([0,125,0,255])
            else:
                if (all_data[y][x][0] > 0):
                    fn += 1
                    t.append([0,0,125,255])
                else:
                    tn += 1
                    t.append([0,0,0,255])
        im.append(t)
        #total += res

    for i in train_points:
        im[i[1]][i[0]] = (255, 255, 0, 255)
    imsave("Small res5.png", im)
    print("Image saved")
    return [(tp+tn)/(tp+tn+fp+fn), tp, tn, fp, fn]

# register the evaluation function
toolbox.register("evaluate", evalFunc, all_data=data, train_points=points)

# set up GP parameters
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Set up logging
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# register methods for calculating various statistics
mstats.register("mean", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

# open file summary
f_avg = open('Logs/'+name+'-avg.txt', 'w')
avg_vals = []


# run the test 20 times
for i in range(1):
    # split the data each time and register the evaluation function using the correct data
    #folds, test_data = splitData(all_data, fold_k)
    #toolbox.register("evaluate", evalFunc, data)

    # generate the populations
    pop = toolbox.population(n=1250)
    # holds the n best individuals
    hof = tools.HallOfFame(1)
    print ("Run: " + str(i))
    pop, logs = algorithms.eaSimple(pop, toolbox, 0.9, 0.2, 60, stats=mstats, halloffame=hof, verbose=True)
    # Open file to log the results
    f = open('Logs/'+name+'-logs-' + str(i) +'.txt', 'w')
    f.write(str(logs))
    f.close()
    expr = hof[0]
    # Print and store the testing results for the best solution

    #print("fitness: " + str(testEval(expr, test_data)))
    avg_vals.append("\t".join(str(s) for s in testEval(expr, data, points)))

# write results to the file
f_avg.write(str("\n".join(str(s) for s in avg_vals)))
f_avg.close()
