# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#The following code takes the image Image.jpg and does compression using singular value decomposition with k singular values. The output of this code is a file with a compressed image named CompressedImage.jpg.

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from PIL import Image
import scipy.misc

image = np.array(Image.open('Image'))   #Import Image
row, col, _ = image.shape
print "The number of pixels of this image is: ", row, "x", col #Number of pixels of original image
image_bytes = image.nbytes  #Calculate original space occupied
print "The space (in bytes) needed to store this image is", image_bytes

red = image[:, :, 0]   
green = image[:, :, 1] 
blue = image[:, :, 2]
# Full Singular Value Decomposition 
red_U, red_d, red_V = np.linalg.svd(red, full_matrices=True)  
green_U, green_d, green_V = np.linalg.svd(green, full_matrices=True)
blue_U, blue_d, blue_V = np.linalg.svd(blue, full_matrices=True)

k=50 #Store only k rows (You can change the value of k to change the number of singular values of the compression)
red_U_k = red_U[:, 0:k]
red_V_k = red_V[0:k, :]
green_U_k = green_U[:, 0:k]
green_V_k = green_V[0:k, :]
blue_U_k = blue_U[:, 0:k]
blue_V_k = blue_V[0:k, :]

red_d_k = red_d[0:k]
green_d_k = green_d[0:k]
blue_d_k = blue_d[0:k]

#Calculate data necessary to recover original approximation

red_c=np.mean(red, axis=0)
np.outer(np.ones([1,col]),red_c)
red_E=red-red_c
red_V_t= red_V.transpose()
red_HatV= red_V_t[:,0:k]
red_Y=np.dot(red_E,red_HatV)

green_c=np.mean(green, axis=0)
np.outer(np.ones([1,col]),green_c)
green_E=green-green_c
green_V_t= green_V.transpose()
green_HatV= green_V_t[:,0:k]
green_Y=np.dot(green_E,green_HatV)

blue_c=np.mean(blue, axis=0)
np.outer(np.ones([1,col]),blue_c)
blue_E=blue-blue_c
blue_V_t= blue_V.transpose()
blue_HatV= blue_V_t[:,0:k]
blue_Y=np.dot(blue_E,blue_HatV)


compressed_bytes = sum([matrix.nbytes for matrix in 
                        [red_Y, red_HatV, red_c, green_Y, green_HatV, green_c, blue_Y, blue_HatV, blue_c]])


storage_ratio = compressed_bytes/image_bytes

#Recover approximation of original image 

red_E1=np.dot(red_Y,red_HatV.transpose())
red_cmat=np.outer(np.ones([1,row]),red_c)
red_aprox=red_E1+red_cmat

green_E1=np.dot(green_Y,green_HatV.transpose())
green_cmat=np.outer(np.ones([1,row]),green_c)
green_aprox=green_E1+green_cmat

blue_E1=np.dot(blue_Y,blue_HatV.transpose())
blue_cmat=np.outer(np.ones([1,row]),blue_c)
blue_aprox=blue_E1+blue_cmat

#Reconstruct the three dimensional array  
image_reconstructed = np.zeros((row, col, 3))

image_reconstructed[:, :, 0] = red_aprox
image_reconstructed[:, :, 1] = green_aprox
image_reconstructed[:, :, 2] = blue_aprox

scipy.misc.imsave('CompressedImage.jpg', image_reconstructed)

#Forbenius norm of the difference
error=linalg.norm(image-image_reconstructed)

print k
print compressed_bytes
print storage_ratio
print error 


