import operator
import math
import random
from random import shuffle

import numpy as np
import types

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from scipy.misc import imread, imsave
from scipy.ndimage.filters import sobel, laplace, gaussian_laplace
import array

gt_img = imread("Images/gt-img1.png", flatten="True", mode="RGB")

f = open("img_1_datapoints.txt", 'r')
points = f.readlines()



images = []

images.append(imread("Images/g-img1.png", flatten="True", mode="RGB"))

img_width = len(images[0][0])
img_height = len(images[0])

for i in range(len(images[0])):
    print(images[0][i])


images.append(sobel(images[0], axis=0))
images.append(sobel(images[0], axis=1))
images.append(laplace(images[0]))
images.append(gaussian_laplace(images[0], 0.05))

for i in range(len(images)):
    imsave("Images/Means/img-" + str(i) + ".jpg", images[i])

avg_17x17 = []
avg_13x13 = []
avg_21x21 = []

def getMean(img, x, y, size):
    tot = 0
    count = 0
    #print (img[y])
    for j in range(y-(int(size/2)), y+(int(size/2))+1):
        for i in range(x-(int(size/2)), x+(int(size/2))+1):
            if not ((i < 0) or (j < 0) or (i >= img_width) or (j >= img_height)):
                #print (img[j][i])
                tot += (img[j][i])%255
                count += 1
    return tot/count

def genMeanImg (img, size):
    avg_img = []
    for y in range(len(img)):
        row = []
        for x in range(len(img[y])):
            row.append(getMean(img, x, y, size))
        avg_img.append(row)
    return avg_img

for i in range(len(images)):
    print("Calculating means:")
    print("Calculating 13x13")
    a = open("Images/Means/img-" + str(i) + "-13.txt", 'w')
    tmp = genMeanImg(images[i], 13)
    #print(tmp)
    #print(tmp[0])
    print(tmp[0][0])
    avg_13x13.append(tmp)
    for j in range(len(tmp)):
        a.write("\t".join(str(v) for v in tmp[j]) +"\n")
        a.flush()
    a.close()

    print("Calculating 17x17")
    a = open("Images/Means/img-" + str(i) + "-17.txt", 'w')
    tmp = genMeanImg(images[i], 17)
    avg_17x17.append(tmp)
    for j in range(len(tmp)):
        a.write("\t".join(str(v) for v in tmp[j]) +"\n")
        a.flush()
    a.close()

    print("Calculating 21x21")
    a = open("Images/Means/img-" + str(i) + "-21.txt", 'w')
    tmp = genMeanImg(images[i], 21)
    avg_21x21.append(tmp)
    for j in range(len(tmp)):
        a.write("\t".join(str(v) for v in tmp[j]) +"\n")
        a.flush()
    a.close()
