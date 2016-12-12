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

#for i in range(len(images[0])):
#    print(images[0][i])


images.append(sobel(images[0], axis=0))
images.append(sobel(images[0], axis=1))
images.append(laplace(images[0]))
images.append(gaussian_laplace(images[0], 0.05))

for i in range(len(images)):
    imsave("Images/Means/Test2-img-" + str(i) + ".jpg", images[i])

avg_17x17 = []
avg_13x13 = []
avg_21x21 = []
std_17x17 = []
std_13x13 = []
std_21x21 = []


def getMean(img, x, y, size):
    tot = 0
    count = 0
    #print (img[y])
    off = (size - 1)//2
    start = end = 0

    # find start of row
    if (x - off < 0):
        start = 0
    else:
        start = x-off

    # find end of row
    if (x+off+1 > len(img[0])):
        end = len(img[0])
    else:
        end = x+off+1

    yStart = y
    if (y - off < 0):
        yStart = 0
    else:
        yStart = y - off

    yEnd = y
    if (y + off+1 > len(img)):
        yEnd = len(img)
    else:
        yEnd = y + off+1

    vals = []

    for i in range(yStart, yEnd):
        vals.extend(img[i][start:end])

    return np.mean(np.array(vals)%255), np.std(np.array(vals)%255)



    '''
    for j in range(y-(int(size/2)), y+(int(size/2))+1):
        for i in range(x-(int(size/2)), x+(int(size/2))+1):
            if not ((i < 0) or (j < 0) or (i >= img_width) or (j >= img_height)):
                #print (img[j][i])
                tot += (img[j][i])%255
                count += 1

    return tot/count
    '''



def genMeanImg (img, size):
    avg_img = []
    std_img = []
    for y in range(len(img)):
        row = []
        row_std = []
        for x in range(len(img[y])):
            avg, std = getMean(img, x, y, size)
            row.append(avg)
            row_std.append(std)
        avg_img.append(row)
        std_img.append(row_std)
    return avg_img, std_img

for i in range(len(images)):
    print("Calculating means:")
    print("\tCalculating 13x13")
    a = open("Images/Means/Test2-img-" + str(i) + "-13.txt", 'w+')
    tmp, tmp2 = genMeanImg(images[i], 13)

    #print (tmp2)
    avg_13x13.append(tmp)
    std_13x13.append(tmp2)
    print("\t\tWriting Average")
    for j in range(len(tmp)):
        a.write("\t".join(str(v) for v in tmp[j]) +"\n")
        a.flush()
    a.close()
    print("\t\tWriting Standard Deviation");
    b = open("Images/Deviation/Test2-img-" + str(i) + "-13.txt", 'w+')
    for j in range(len(tmp2)):
        b.write("\t".join(str(v) for v in tmp2[j]) +"\n")
        b.flush()
    b.close()

    print("Calculating 17x17")
    a = open("Images/Means/Test2-img-" + str(i) + "-17.txt", 'w+')
    b = open("Images/Deviation/Test2-img-" + str(i) + "-17.txt", 'w+')
    tmp, tmp2 = genMeanImg(images[i], 17)

    avg_17x17.append(tmp)
    std_17x17.append(tmp2)
    for j in range(len(tmp)):
        a.write("\t".join(str(v) for v in tmp[j]) +"\n")
        b.write("\t".join(str(v) for v in tmp2[j]) +"\n")
        a.flush()
        b.flush()
    a.close()
    b.close()

    print("Calculating 21x21")
    a = open("Images/Means/Test2-img-" + str(i) + "-21.txt", 'w+')
    b = open("Images/Deviation/Test2-img-" + str(i) + "-21.txt", 'w+')
    tmp, tmp2 = genMeanImg(images[i], 21)

    avg_21x21.append(tmp)
    std_21x21.append(tmp2)
    for j in range(len(tmp)):
        a.write("\t".join(str(v) for v in tmp[j]) +"\n")
        b.write("\t".join(str(v) for v in tmp2[j]) +"\n")
        a.flush()
        b.flush()
    a.close()
    b.close()
