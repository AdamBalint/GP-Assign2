#import cv
import random
from scipy.misc import imread
img = imread("Images/gt-img1.png", flatten="True", mode="RGB")

black = red = 0

f = open("Datapoints.txt", 'w')

f.write("x\ty\tres\n")

res = []

hand_picked_r = [   (117,56),(516,13),(267,237),(215,93),
                    (355,795),(650,323),(699,57),(632,127),(341,253)]
hand_picked_b = [   (322,83),(403,149),(643,276),(591,454),(131,1045),
                    (632,131),(604,144),(686,43),(511,1),(375,1087),
                    (361,1077),(664,962),(693,973),(643,1014),(115,378),
                    (82,628),(518,892),(137,1022),(907,949),(287,491),
                    (339,455),(302,409)]
picked_black = []
picked_black.extend(hand_picked_b)

hp_black_set = {}
hp_black_set = set(hand_picked_b);

for i in range(len(hand_picked_b)):
    print(str(hand_picked_b[i]) + "\t" + str(img[hand_picked_b[i][1]][hand_picked_b[i][0]]))


all_black = []
for i in range(len(img)):
    for j in range(len(img[i])):
        if (img[i][j] == 0):
            black += 1
            all_black.append((j,i))
        else:
            red += 1

while(len(picked_black) != 90):
    pick = all_black[random.randint(0,len(all_black)-1)]
    if (pick not in hp_black_set):
        picked_black.append(pick)
        hp_black_set.add(pick)



print ("Black: " + str(black) + "\t" + str(black / (black+red)))
print ("Red: " + str(red) + "\t" + str(red / (black+red)))

for i in range(len(hand_picked_r)):
    f.write(str(hand_picked_r[i][0]) + "\t" + str(hand_picked_r[i][1]) + "\t1\n")
f.flush()
for i in range(len(picked_black)):
    f.write(str(picked_black[i][0]) + "\t" + str(picked_black[i][1]) + "\t0\n")
f.flush()

f.close()
# 10-4 : professors here
# 11-12:30 : classtime
# 12-30:2 : talk time
# 2-4 : Extra time
# Extra time : lunch, meetings with profs (Ross(10) and Ombuki(when she comes in)), work (if I have any),
#              meeting with academic advisor about jobs that came in
#              figuring out grad school stuff, dealing with school paperwork
#              Talking to TA person and anyone who needs help(marking) with AI assignments,
#              meeting with head TA (sometimes) and organizing CSC events
#              managing the CSC, talking with people at the school (networking - very important!)
#              Doing as much assignments as I can so I can discuss with people if I have any problems
#              Just hanging out with people
#              Talking to chair of department (sometimes) and building connections with the profs
#              in the department (reference letters needed in future)
#              Eat lunch
