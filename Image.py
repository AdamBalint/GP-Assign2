from scipy.misc import imread, imsave
from scipy.ndimage.filters import sobel, laplace, gaussian_laplace


print(imread("Images/c-img1.png"))

img = []
r = [255,0,0,255]
g = [0,255,0,255]
b = [0,0,255,255]
w = [255,255,255,255]
img.append([r,g])
img.append([b,w])

imsave("Test_save.png", img)
