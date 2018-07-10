#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:09:11 2018

@author: juanis
"""
import numpy as np
import math
import scipy.misc
from PIL import Image
from numpy import linalg

#The following code takes the image Image.jpg and does compression using Haar Transform with values of k used as the threshold for quantization. The output of this code is a file with a compressed image named CompressedImage.jpg.

#Import and separate RGB components
image = np.array(Image.open('Image.jpg'))   #Import Image
image = image / 255.0
image = image 
red = image[:, :, 0]   
green = image[:, :, 1] 
blue = image[:, :, 2]

#Convert matrices into one dimension that is 2^k
r=max(red.shape)
s=int(pow(2,math.ceil(math.log(r,2))))
r2 = np.zeros((s, s))
r2[:red.shape[0], :red.shape[1]] = red

g2 = np.zeros((s, s))
g2[:green.shape[0], :green.shape[1]] = green

b2 = np.zeros((s, s))
b2[:blue.shape[0], :blue.shape[1]] = blue


#Haar Matrix calculation (Dim n)
N = s
p = np.array([0, 0])
q = np.array([0, 1])
n = int(np.log2(N))

for i in np.arange(1, n):
    p = np.concatenate((p, i * np.ones(2**i)))
    t = np.arange(1, 2**i + 1)
    q = np.concatenate((q, t))

Hr = np.zeros([N, N])
Hr[0,:] = 1;

for i in np.arange(1, N):
    P = p[i]; # probably is actually (0, i)
    Q = q[i]; # probably is actually (0, i)    
    for j in np.arange(N * (Q - 1) / (2**P), N * (( Q - 0.5) / (2**P)), dtype = int):
        Hr[i, j] = 2**(P/2)    
    for j in np.arange(N * (( Q - 0.5) / 2**P), (N * (Q / 2**P)), dtype = int):
        Hr[i, j] = -(2**(P/2))
Hr = Hr * (1/ np.sqrt(N))

#Calculating Haar transform
r2T=np.matmul(np.matmul(Hr,r2),np.transpose(Hr))
g2T=np.matmul(np.matmul(Hr,g2),np.transpose(Hr))
b2T=np.matmul(np.matmul(Hr,b2),np.transpose(Hr))

#Quantization (For different threshold change the values of k)
k = 2
vecr2T=np.absolute(np.reshape(r2T,s*s))
mir=np.min(vecr2T[np.nonzero(vecr2T)])
mar=np.max(vecr2T[np.nonzero(vecr2T)])

rquantT= np.asarray(r2T)
low= np.absolute(rquantT) < k # Where values are low
rquantT[low] = 0
nonzeror=np.sum(1 - low.astype(int))

vecg2T=np.absolute(np.reshape(g2T,s*s))
mig=np.min(vecg2T[np.nonzero(vecg2T)])
mag=np.max(vecg2T[np.nonzero(vecg2T)])

gquantT= np.asarray(g2T)
low= np.absolute(gquantT) < k # Where values are low
gquantT[low] = 0
nonzerog=np.sum(1 - low.astype(int))

vecb2T=np.absolute(np.reshape(b2T,s*s))
mib=np.min(vecb2T[np.nonzero(vecb2T)])
mab=np.max(vecb2T[np.nonzero(vecb2T)])

bquantT= np.asarray(b2T)
low= np.absolute(bquantT) < k # Where values are low
bquantT[low] = 0
nonzerob=np.sum(1 - low.astype(int))


#Calculating Haar Inverse transform and Reconstructing the Image
rquant= np.matmul(np.matmul(np.transpose(Hr),rquantT),Hr)
gquant=np.matmul(np.matmul(np.transpose(Hr),gquantT),Hr)
bquant=np.matmul(np.matmul(np.transpose(Hr),bquantT),Hr)

image_reconstructed = np.zeros((red.shape[0], red.shape[1], 3))

image_reconstructed[:, :, 0] = rquant[:red.shape[0], :red.shape[1]]
image_reconstructed[:, :, 1] = gquant[:green.shape[0], :green.shape[1]]
image_reconstructed[:, :, 2] = bquant[:blue.shape[0], :blue.shape[1]]

scipy.misc.imsave('ImageCompressed.jpg', image_reconstructed * 255.0)

#Error Calculation
error1=linalg.norm(red-rquant[:red.shape[0], :red.shape[1]])
error2=linalg.norm(green-gquant[:green.shape[0], :green.shape[1]])
error3=linalg.norm(blue-bquant[:blue.shape[0], :blue.shape[1]])
error=(error1+error2+error3)/3

frob1=linalg.norm(red)
frob2=linalg.norm(green)
frob3=linalg.norm(blue)
frob=(frob1+frob2+frob3)/3

relerror=error/frob

#Sparseness ratio

spr=float(nonzeror)/(low.shape[0]*low.shape[1])
spg=float(nonzerog)/(low.shape[0]*low.shape[1])
spb=float(nonzerob)/(low.shape[0]*low.shape[1])


spRatio=(spr+spg+spb)/3

print k
print spRatio
print relerror


